# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K2 gated RMSNorm in CuTe DSL: two thread-block-cluster kernels.

Row kernel (decode batches, ``M * 4 <= #SMs``): one cluster of C CTAs per row,
8 bf16 columns per thread, C = 8 or 4 so the batch fits one wave. rstd is
factored out of the W_down contraction, so a row needs a single DSMEM exchange
of 17 partial sums (sum of squares + 16 z partials): every block pushes its
sums into the same slot of every peer (st.shared::cluster), one split cluster
barrier, then each block sums the slots locally.

Tile kernel (everything else): one cluster of C CTAs per 16-row tile, each CTA
owning H / C columns (C = 8 for H = 7168: 7 warps x 128 columns). The rank-16
projections run on tensor cores (mma.sync m16n8k16 bf16, hi/lo split of the
fp32 operand): phase 1 computes sum(h^2) and z' = (h * w_norm) @ W_down^T per
row over this CTA's columns, reduces them over the warps and pushes the 17
partials per row to every peer; phase 2 forms rstd and z = silu(z' * rstd),
gate = z @ W_up^T and the output. The MMA k / n indices are permuted onto the
8 consecutive columns each thread owns, so fragments fall out of 16-byte
vector loads without shuffles; W_up is pre-arranged into B-fragment order on
the host (``make_wu_frag``). h is kept in smem (fp32) between the phases and
written back to the residual in phase 2. Weights are read once per 16 rows
instead of once per row.

Weight slices are loaded before griddepcontrol_wait() so they overlap the
producer kernel under programmatic dependent launch. B200, H = 7168, per
launch: 3.7 us at M <= 8 (per-row Triton 9.9), 6.1 us at M = 64 (10.0),
9.3 us at M = 400 (28.9), 149 us at M = 8192 (533). Compiled per
(kernel, H, C, residual, eps), M is symbolic; the first call of a variant
costs ~1 s, so warmup() runs them all at model load.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

RANK = 16
VEC = 8
PARTS = RANK + 1
MAX_CLUSTER = 8
TILE_ROWS = 16
TILE_WARP_COLS = 128
TILE_GROUPS = TILE_WARP_COLS // 32
TILE_MIN_BLOCKS = 2


@functools.cache
def _cute_available() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from quack.compile_utils import make_fake_tensor  # noqa: F401
    except ImportError as e:  # noqa: BLE001
        logger.info("CuTe DSL unavailable for the A.X-K2 gated RMSNorm: %r", e)
        return False
    return True


def make_wu_frag(w_up: torch.Tensor) -> torch.Tensor:
    """W_up [H, 16] -> [H // 32, 32 lanes, 16] in mma.m16n8k16 B-fragment order.

    Entry (G, lane, 4t + v) = W_up[32G + 8(g//2) + 2t + (g%2), 2q + (v&1) + 8(v>>1)]
    with g = lane // 4, q = lane % 4: the four values lane (g, q) needs for
    n8 tile t of 32-column group G, so that the C fragment of tiles 0..3 lands
    on the lane's 8 consecutive columns 8q..8q+7."""
    H = w_up.shape[0]
    assert H % 32 == 0 and w_up.shape[1] == RANK
    groups = H // 32
    lane = torch.arange(32)
    g, q = lane // 4, lane % 4
    t = torch.arange(4)
    v = torch.arange(4)
    col = 8 * (g // 2)[:, None] + 2 * t[None, :] + (g % 2)[:, None]  # [32, t]
    rank = 2 * q[:, None] + (v & 1)[None, :] + 8 * (v >> 1)[None, :]  # [32, v]
    cols = 32 * torch.arange(groups)[:, None, None, None] + col[None, :, :, None]
    cols = cols.expand(groups, 32, 4, 4)
    ranks = rank[None, :, None, :].expand(groups, 32, 4, 4)
    out = w_up.detach()[cols.to(w_up.device), ranks.to(w_up.device)]
    return out.reshape(groups, 32, RANK).contiguous()


def _build_kernel_classes():
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.math as cmath
    from cuda.bindings.driver import CUstream
    from cutlass import BFloat16, Float32, Int32, const_expr
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def dsmem_store_f32(smem_ptr: cute.Pointer, rank, value, *, loc=None, ip=None):
        """Store one fp32 at this smem offset in CTA `rank` of the cluster.

        The DSL's dsmem tensors do not support load/store, hence the PTX."""
        addr = cutlass.Int32(smem_ptr.toint(loc=loc, ip=ip))
        llvm.inline_asm(
            T.i32(),
            [
                addr.ir_value(loc=loc, ip=ip),
                cutlass.Int32(rank).ir_value(loc=loc, ip=ip),
                Float32(value).ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;\n\tst.shared::cluster.f32 [$0], $3;",
            "=r,r,r,f",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )

    class _Common:
        @cute.jit
        def _load8(self, t: cute.Tensor):
            r = cute.make_rmem_tensor_like(t)
            cute.autovec_copy(t, r)
            return r.load().to(Float32)

        @cute.jit
        def _load8_bf(self, t: cute.Tensor):
            r = cute.make_rmem_tensor_like(t)
            cute.autovec_copy(t, r)
            return r.load()

        @cute.jit
        def _store8(self, vals, t: cute.Tensor):
            r = cute.make_rmem_tensor_like(t)
            r.store(vals.to(BFloat16))
            cute.autovec_copy(r, t)

        @cute.jit
        def _sum8(self, v):
            return v.reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )

        @cute.jit
        def _sigmoid(self, v):
            return Float32(1.0) / (Float32(1.0) + cmath.exp(-v))

        @cute.jit
        def _sigmoid_fast(self, v):
            return cmath.rcp(Float32(1.0) + cmath.exp(-v, fastmath=True), approx=True)

    class GatedRMSNormRow(_Common):
        """grid = M * C blocks of T threads, cluster C per row."""

        def __init__(
            self,
            hidden: int,
            cluster: int,
            threads: int,
            has_residual: bool,
            eps: float,
            use_pdl: bool,
        ):
            assert hidden % (cluster * VEC) == 0
            self.H = hidden
            self.C = cluster
            self.T = threads
            self.cols = hidden // cluster
            self.nv = self.cols // VEC
            assert self.nv <= threads
            self.num_warps = threads // 32
            self.has_residual = has_residual
            self.eps = eps
            self.use_pdl = use_pdl
            # 512/1024-thread blocks would spill with W_up resident across the
            # barrier; they load it after the exchange instead.
            self.early_wu = threads <= 256

        @cute.jit
        def __call__(
            self,
            gX: cute.Tensor,
            gRes: cute.Tensor,
            gWn: cute.Tensor,
            gWd: cute.Tensor,
            gWu: cute.Tensor,
            gOut: cute.Tensor,
            stream: CUstream,
        ):
            m = cute.size(gX, mode=[0])
            self.kernel(gX, gRes, gWn, gWd, gWu, gOut).launch(
                grid=[m * self.C, 1, 1],
                block=[self.T, 1, 1],
                cluster=[self.C, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=1,
            )

        def _load_wu(self, gWu_v, v_idx):
            """W_up rows of this thread's 8 columns, two 8-vectors each (plain
            Python at trace time, so the list is built in one place)."""
            return [
                (
                    self._load8(gWu_v[v_idx * VEC + i, (None, 0)]),
                    self._load8(gWu_v[v_idx * VEC + i, (None, 1)]),
                )
                for i in range(VEC)
            ]

        @cute.kernel
        def kernel(
            self,
            gX: cute.Tensor,
            gRes: cute.Tensor,
            gWn: cute.Tensor,
            gWd: cute.Tensor,
            gWu: cute.Tensor,
            gOut: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            wid = cute.arch.warp_idx()
            C: cutlass.Constexpr = self.C
            rank = cutlass.Int32(0)
            if const_expr(C > 1):
                rank = cute.arch.block_idx_in_cluster()
            row = bidx // C
            # Idle threads (nv < T) load a valid vector and contribute zero.
            t_eff = tidx
            mask = Float32(1.0)
            if tidx >= self.nv:
                t_eff = cutlass.Int32(self.nv - 1)
                mask = Float32(0.0)
            v_idx = rank * self.nv + t_eff

            # 8-wide vector views: (M, (8, H/8)), (16, (8, H/8)), (H, (8, 2)),
            # ((8, H/8))
            gX_v = cute.logical_divide(gX, (None, VEC))
            gRes_v = cute.logical_divide(gRes, (None, VEC))
            gOut_v = cute.logical_divide(gOut, (None, VEC))
            gWd_v = cute.logical_divide(gWd, (None, VEC))
            gWu_v = cute.logical_divide(gWu, (None, VEC))
            gWn_v = cute.logical_divide(gWn, (VEC,))

            smem = cutlass.utils.SmemAllocator()
            warp_part = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.num_warps, PARTS), stride=(PARTS, 1)),
                byte_alignment=16,
            )
            xchg = smem.allocate_tensor(
                Float32,
                cute.make_layout((MAX_CLUSTER, PARTS), stride=(PARTS, 1)),
                byte_alignment=16,
            )
            zr = smem.allocate_tensor(
                Float32, cute.make_layout((PARTS,)), byte_alignment=16
            )

            # ---- weights first: independent of the producer kernel ----
            wn = self._load8(gWn_v[((None, v_idx),)])
            wd = [self._load8(gWd_v[r, (None, v_idx)]) for r in range(RANK)]
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # ---- x (+ residual, written back) ----
            h = self._load8(gX_v[row, (None, v_idx)])
            if const_expr(self.has_residual):
                h = h + self._load8(gRes_v[row, (None, v_idx)])
                if tidx < self.nv:
                    self._store8(h, gRes_v[row, (None, v_idx)])
            h = h * mask
            xw = h * wn

            # ---- local partials: sum(h^2) and sum_i Wd[r, i] * w[i] * h[i] ----
            part = cute.make_rmem_tensor((PARTS,), Float32)
            part[RANK] = self._sum8(h * h)
            for r in cutlass.range_constexpr(RANK):
                part[r] = self._sum8(wd[r] * xw)

            wu = self._load_wu(gWu_v, v_idx) if self.early_wu else None

            # ---- block reduction ----
            for k in cutlass.range_constexpr(PARTS):
                part[k] = cute.arch.warp_reduction_sum(part[k])
            with cute.arch.elect_one():
                for k in cutlass.range_constexpr(PARTS):
                    warp_part[wid, k] = part[k]
            cute.arch.sync_threads()
            if tidx < PARTS:
                s = Float32(0.0)
                for w in cutlass.range_constexpr(self.num_warps):
                    s = s + warp_part[w, tidx]
                if const_expr(C > 1):
                    slot = xchg.iterator + (rank * PARTS + tidx)
                    for b in cutlass.range_constexpr(C):
                        dsmem_store_f32(slot, b, s)
                else:
                    xchg[0, tidx] = s

            # ---- cluster exchange (push model, one split barrier) ----
            if const_expr(C > 1):
                cute.arch.cluster_arrive()
            wu = wu if self.early_wu else self._load_wu(gWu_v, v_idx)
            if const_expr(C > 1):
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()

            # ---- rstd and z once per row (17 threads), then the gate ----
            if tidx < PARTS:
                s = Float32(0.0)
                ss = Float32(0.0)
                for b in cutlass.range_constexpr(C):
                    s = s + xchg[b, tidx]
                    ss = ss + xchg[b, RANK]
                rstd = cmath.rsqrt(ss / Float32(self.H) + Float32(self.eps))
                val = s * rstd
                zv = val * self._sigmoid(val)
                if tidx == RANK:
                    zv = rstd
                zr[tidx] = zv
            cute.arch.sync_threads()

            if tidx < self.nv:
                rstd = zr[RANK]
                z = zr.load()
                out = cute.make_rmem_tensor((VEC,), Float32)
                assert wu is not None
                for i in cutlass.range_constexpr(VEC):
                    lo, hi = wu[i]
                    g = Float32(0.0)
                    for r in cutlass.range_constexpr(VEC):
                        g = g + lo[r] * z[r]
                    for r in cutlass.range_constexpr(VEC):
                        g = g + hi[r] * z[VEC + r]
                    out[i] = xw[i] * rstd * self._sigmoid(g)
                self._store8(out.load(), gOut_v[row, (None, v_idx)])

    class GatedRMSNormTile(_Common):
        """grid = ceil(M / 16) * C blocks of (H / C / 128) warps, cluster C
        per 16-row tile; thread (g = lane // 4, q = lane % 4) owns rows g and
        g + 8 and columns 8q..8q+7 of each 32-column group of its warp."""

        def __init__(
            self,
            hidden: int,
            cluster: int,
            has_residual: bool,
            eps: float,
            use_pdl: bool,
        ):
            assert hidden % (cluster * TILE_WARP_COLS) == 0, (hidden, cluster)
            self.H = hidden
            self.C = cluster
            self.cols = hidden // cluster
            self.W = self.cols // TILE_WARP_COLS
            assert 1 <= self.W <= 32
            self.T = self.W * 32
            self.has_residual = has_residual
            self.eps = eps
            self.use_pdl = use_pdl

        @cute.jit
        def __call__(
            self,
            gX: cute.Tensor,
            gRes: cute.Tensor,
            gWn: cute.Tensor,
            gWd: cute.Tensor,
            gWuF: cute.Tensor,
            gOut: cute.Tensor,
            stream: CUstream,
        ):
            m = cute.size(gX, mode=[0])
            op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(op)
            tiles = cute.ceil_div(m, TILE_ROWS)
            self.kernel(gX, gRes, gWn, gWd, gWuF, gOut, tiled_mma).launch(
                grid=[tiles * self.C, 1, 1],
                block=[self.T, 1, 1],
                cluster=[self.C, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
                # 2 CTAs per SM: ptxas keeps this at ~127 registers without
                # spills; 3+ spill and run slower.
                min_blocks_per_mp=TILE_MIN_BLOCKS,
            )

        def _frag(self, shape, dtype):
            return cute.make_rmem_tensor(cute.make_layout(shape), dtype)

        @cute.kernel
        def kernel(
            self,
            gX: cute.Tensor,
            gRes: cute.Tensor,
            gWn: cute.Tensor,
            gWd: cute.Tensor,
            gWuF: cute.Tensor,
            gOut: cute.Tensor,
            tiled_mma: cute.TiledMma,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            wid = cute.arch.warp_idx()
            lane = cute.arch.lane_idx()
            C: cutlass.Constexpr = self.C
            W: cutlass.Constexpr = self.W
            NT: cutlass.Constexpr = self.T
            G: cutlass.Constexpr = TILE_GROUPS
            rank = Int32(0)
            if const_expr(C > 1):
                rank = cute.arch.block_idx_in_cluster()
            tile = bidx // C
            m = cute.size(gX, mode=[0])
            g = lane // 4
            q = lane % 4
            r0 = tile * TILE_ROWS + g
            r1 = r0 + 8
            # Rows past M load row M - 1 (valid memory) and skip their stores.
            r0c = r0
            r1c = r1
            if r0 >= m:
                r0c = m - 1
            if r1 >= m:
                r1c = m - 1
            col0 = rank * self.cols + wid * TILE_WARP_COLS

            gX_v = cute.logical_divide(gX, (None, VEC))
            gRes_v = cute.logical_divide(gRes, (None, VEC))
            gOut_v = cute.logical_divide(gOut, (None, VEC))
            gWd_v = cute.logical_divide(gWd, (None, VEC))
            gWn_v = cute.logical_divide(gWn, (VEC,))
            gWuF_v = cute.logical_divide(gWuF, (None, None, VEC))

            smem = cutlass.utils.SmemAllocator()
            # h (fp32, so y matches the row kernel bit for bit) between the
            # phases; chunk-major so consecutive threads hit consecutive slots
            # (no bank conflicts).
            stash = smem.allocate_tensor(
                Float32,
                cute.make_layout((NT, 2 * G, VEC), stride=(VEC, NT * VEC, 1)),
                byte_alignment=16,
            )
            part = smem.allocate_tensor(
                Float32,
                cute.make_layout((W, TILE_ROWS, RANK), stride=(TILE_ROWS * 20, 20, 1)),
                byte_alignment=16,
            )
            ssp = smem.allocate_tensor(
                Float32,
                cute.make_layout((W, 2, 32), stride=(64, 32, 1)),
                byte_alignment=16,
            )
            xchg = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (MAX_CLUSTER, TILE_ROWS, PARTS),
                    stride=(TILE_ROWS * PARTS, PARTS, 1),
                ),
                byte_alignment=16,
            )
            zs = smem.allocate_tensor(
                Float32,
                cute.make_layout((TILE_ROWS, PARTS), stride=(PARTS, 1)),
                byte_alignment=16,
            )
            zsm = smem.allocate_tensor(
                Float32,
                cute.make_layout((TILE_ROWS, RANK), stride=(RANK, 1)),
                byte_alignment=16,
            )
            rstd_s = smem.allocate_tensor(
                Float32, cute.make_layout((TILE_ROWS,)), byte_alignment=16
            )

            # 8-wide vector index of this thread's columns in group j.
            vcol = [(col0 + 32 * j) // VEC + q for j in range(G)]
            # ---- weights first: independent of the producer kernel ----
            wd = [
                (
                    self._load8_bf(gWd_v[g, (None, vc)]),
                    self._load8_bf(gWd_v[8 + g, (None, vc)]),
                )
                for vc in vcol
            ]
            wn = [self._load8_bf(gWn_v[((None, vc),)]) for vc in vcol]
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            h0s = [self._load8(gX_v[r0c, (None, vc)]) for vc in vcol]
            h1s = [self._load8(gX_v[r1c, (None, vc)]) for vc in vcol]
            if const_expr(self.has_residual):
                h0s = [
                    h0s[j] + self._load8(gRes_v[r0c, (None, vcol[j])]) for j in range(G)
                ]
                h1s = [
                    h1s[j] + self._load8(gRes_v[r1c, (None, vcol[j])]) for j in range(G)
                ]

            # ---- phase 1: sum(h^2) and z' on tensor cores ----
            # A fragment (k16 step ks): a0,a1 = (row g, k 2q..2q+1), a2,a3 =
            # (row g+8, same k), a4..a7 the same for k 2q+8..2q+9. k pair
            # (2q, 2q+1) is this thread's local columns 4ks..4ks+1 and pair
            # (2q+8, 2q+9) its columns 4ks+2..4ks+3; B uses the same k map,
            # so b0..b3 are W_down[8t + g, local 4ks..4ks+3].
            acc = [self._frag(((2, 2), 1, 1), Float32) for _ in range(2)]
            for t in cutlass.range_constexpr(2):
                acc[t].fill(0.0)
            ss0 = Float32(0.0)
            ss1 = Float32(0.0)
            for j in cutlass.range_constexpr(G):
                h0 = h0s[j]
                h1 = h1s[j]
                st0 = cute.make_rmem_tensor((VEC,), Float32)
                st0.store(h0)
                cute.autovec_copy(st0, stash[tidx, j, None])
                st1 = cute.make_rmem_tensor((VEC,), Float32)
                st1.store(h1)
                cute.autovec_copy(st1, stash[tidx, G + j, None])
                ss0 = ss0 + self._sum8(h0 * h0)
                ss1 = ss1 + self._sum8(h1 * h1)
                wnf = wn[j].to(Float32)
                y0 = h0 * wnf
                y1 = h1 * wnf
                y0h = y0.to(BFloat16)
                y1h = y1.to(BFloat16)
                y0l = (y0 - y0h.to(Float32)).to(BFloat16)
                y1l = (y1 - y1h.to(Float32)).to(BFloat16)
                for ks in cutlass.range_constexpr(2):
                    for hl in cutlass.range_constexpr(2):
                        ya0 = (y0h, y0l)[hl]
                        ya1 = (y1h, y1l)[hl]
                        a = self._frag(((2, 2, 2), 1, 1), BFloat16)
                        a[0] = ya0[4 * ks]
                        a[1] = ya0[4 * ks + 1]
                        a[2] = ya1[4 * ks]
                        a[3] = ya1[4 * ks + 1]
                        a[4] = ya0[4 * ks + 2]
                        a[5] = ya0[4 * ks + 3]
                        a[6] = ya1[4 * ks + 2]
                        a[7] = ya1[4 * ks + 3]
                        for t in cutlass.range_constexpr(2):
                            b = self._frag(((2, 2), 1, 1), BFloat16)
                            wdt = wd[j][t]
                            b[0] = wdt[4 * ks]
                            b[1] = wdt[4 * ks + 1]
                            b[2] = wdt[4 * ks + 2]
                            b[3] = wdt[4 * ks + 3]
                            cute.gemm(tiled_mma, acc[t], a, b, acc[t])

            # ---- partials to smem, reduce over warps, push to the cluster ----
            # C fragment: c0,c1 = (row g, n 2q..2q+1), c2,c3 = (row g+8, same).
            for t in cutlass.range_constexpr(2):
                for v in cutlass.range_constexpr(4):
                    part[wid, g + 8 * (v >> 1), 8 * t + 2 * q + (v & 1)] = acc[t][v]
            ssp[wid, 0, lane] = ss0
            ssp[wid, 1, lane] = ss1
            cute.arch.sync_threads()
            NRED: cutlass.Constexpr = TILE_ROWS * PARTS
            for i in cutlass.range_constexpr((NRED + NT - 1) // NT):
                idx = tidx + i * NT
                if idx < NRED:
                    row = idx // PARTS
                    k = idx % PARTS
                    acc_s = Float32(0.0)
                    if k < RANK:
                        for w in cutlass.range_constexpr(W):
                            acc_s = acc_s + part[w, row, k]
                    else:
                        half = row // 8
                        gg = row % 8
                        for w in cutlass.range_constexpr(W):
                            for qq in cutlass.range_constexpr(4):
                                acc_s = acc_s + ssp[w, half, 4 * gg + qq]
                    if const_expr(C > 1):
                        slot = xchg.iterator + (rank * NRED + idx)
                        for pb in cutlass.range_constexpr(C):
                            dsmem_store_f32(slot, pb, acc_s)
                    else:
                        xchg[0, row, k] = acc_s
            if const_expr(C > 1):
                cute.arch.cluster_arrive()
            # W_up fragments (B-fragment order, see make_wu_frag) overlap the barrier.
            wu = [
                (
                    self._load8_bf(gWuF_v[(col0 + 32 * j) // 32, lane, (None, 0)]),
                    self._load8_bf(gWuF_v[(col0 + 32 * j) // 32, lane, (None, 1)]),
                )
                for j in range(G)
            ]
            if const_expr(C > 1):
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()
            for i in cutlass.range_constexpr((NRED + NT - 1) // NT):
                idx = tidx + i * NT
                if idx < NRED:
                    row = idx // PARTS
                    k = idx % PARTS
                    acc_s = Float32(0.0)
                    for pb in cutlass.range_constexpr(C):
                        acc_s = acc_s + xchg[pb, row, k]
                    zs[row, k] = acc_s
            cute.arch.sync_threads()
            NZ: cutlass.Constexpr = TILE_ROWS * RANK
            for i in cutlass.range_constexpr((NZ + NT - 1) // NT):
                idx = tidx + i * NT
                if idx < NZ:
                    row = idx // RANK
                    k = idx % RANK
                    rstd = cmath.rsqrt(
                        zs[row, RANK] / Float32(self.H) + Float32(self.eps)
                    )
                    val = zs[row, k] * rstd
                    zsm[row, k] = val * self._sigmoid(val)
                    if k == 0:
                        rstd_s[row] = rstd
            cute.arch.sync_threads()

            # ---- phase 2: gate on tensor cores, output ----
            # A = z (k = rank, one k16 step), B = W_up fragments; with the n
            # permutation of make_wu_frag, c0,c1 of n8 tile t are this
            # thread's local columns 2t, 2t+1 of the group.
            rstd0 = rstd_s[g]
            rstd1 = rstd_s[8 + g]
            zv = [
                zsm[g, 2 * q],
                zsm[g, 2 * q + 1],
                zsm[8 + g, 2 * q],
                zsm[8 + g, 2 * q + 1],
                zsm[g, 2 * q + 8],
                zsm[g, 2 * q + 9],
                zsm[8 + g, 2 * q + 8],
                zsm[8 + g, 2 * q + 9],
            ]
            ah = self._frag(((2, 2, 2), 1, 1), BFloat16)
            al = self._frag(((2, 2, 2), 1, 1), BFloat16)
            for v in cutlass.range_constexpr(8):
                hi = zv[v].to(BFloat16)
                ah[v] = hi
                al[v] = (zv[v] - hi.to(Float32)).to(BFloat16)
            wn2 = [self._load8_bf(gWn_v[((None, vc),)]) for vc in vcol]
            for j in cutlass.range_constexpr(G):
                wnj = wn2[j].to(Float32)
                accs = [self._frag(((2, 2), 1, 1), Float32) for _ in range(4)]
                for t in cutlass.range_constexpr(4):
                    accs[t].fill(0.0)
                    b = self._frag(((2, 2), 1, 1), BFloat16)
                    wuv = wu[j][t // 2]
                    for v in cutlass.range_constexpr(4):
                        b[v] = wuv[4 * (t % 2) + v]
                    cute.gemm(tiled_mma, accs[t], ah, b, accs[t])
                    cute.gemm(tiled_mma, accs[t], al, b, accs[t])
                h0 = self._load8_bf(stash[tidx, j, None])
                h1 = self._load8_bf(stash[tidx, G + j, None])
                if const_expr(self.has_residual):
                    if r0 < m:
                        self._store8(h0, gRes_v[r0, (None, vcol[j])])
                    if r1 < m:
                        self._store8(h1, gRes_v[r1, (None, vcol[j])])
                out0 = cute.make_rmem_tensor((VEC,), Float32)
                out1 = cute.make_rmem_tensor((VEC,), Float32)
                for t in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(2):
                        i = 2 * t + e
                        out0[i] = (
                            h0[i] * rstd0 * wnj[i] * self._sigmoid_fast(accs[t][e])
                        )
                        out1[i] = (
                            h1[i] * rstd1 * wnj[i] * self._sigmoid_fast(accs[t][2 + e])
                        )
                if r0 < m:
                    self._store8(out0.load(), gOut_v[r0, (None, vcol[j])])
                if r1 < m:
                    self._store8(out1.load(), gOut_v[r1, (None, vcol[j])])

    return GatedRMSNormRow, GatedRMSNormTile, CUstream


class CuteGatedRMSNorm:
    """Compile cache + launcher for the two cluster kernels."""

    def __init__(self) -> None:
        self._compiled: dict[tuple, object] = {}
        self._row_cls = None
        self._tile_cls = None
        self._CUstream = None

    def _ensure(self):
        if self._row_cls is None:
            self._row_cls, self._tile_cls, self._CUstream = _build_kernel_classes()

    @staticmethod
    def threads_for(nv: int) -> int:
        return 128 if nv <= 128 else 256 if nv <= 256 else 512 if nv <= 512 else 1024

    @staticmethod
    def row_supported(H: int) -> bool:
        return H % VEC == 0 and H // VEC <= 1024

    @staticmethod
    def tile_cluster(H: int) -> int | None:
        """Cluster size of the tile kernel for this H: 8 when H is a multiple
        of 1024 (A.X-K2's 7168), else 4; None when neither divides H into
        warps of 128 columns. Cluster 2 would need up to 28 warps per CTA
        and is not offered."""
        for c in (MAX_CLUSTER, 4):
            if H % (c * TILE_WARP_COLS) == 0 and H // (c * TILE_WARP_COLS) <= 32:
                return c
        return None

    @staticmethod
    def pick_row_cluster(H: int, M: int, sms: int) -> int:
        """Largest power-of-two cluster (<= 8) that divides H into 8-wide
        vectors and keeps M rows in one wave."""
        c = MAX_CLUSTER
        while c > max(1, sms // max(M, 1)):
            c >>= 1
        while c > 1 and H % (c * VEC) != 0:
            c >>= 1
        return c

    @classmethod
    def kind(cls, H: int, M: int, sms: int) -> str:
        """Row kernel while a cluster of >= 4 CTAs per row fits in one wave,
        tile kernel above that. Clusters are scheduled per GPC, which leaves
        ~10 % of the SMs idle: on B200 (148 SMs) the row kernel takes 3.7-5.6 us
        up to M = 33 and 10 us from M = 34, the tile kernel 6.3 us throughout."""
        if cls.tile_cluster(H) is None or sms * 9 // 10 >= M * 4:
            return "row"
        return "tile"

    def _stream(self):
        from vllm.utils.torch_utils import current_stream

        assert self._CUstream is not None
        return self._CUstream(current_stream().cuda_stream)

    def _fakes(self, H: int, tile: bool):
        import cutlass
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        bf16 = cutlass.BFloat16
        m = cute.sym_int()
        w_up_shape = (H // 32, 32, RANK) if tile else (H, RANK)
        return (
            make_fake_tensor(bf16, (m, H), divisibility=VEC),
            make_fake_tensor(bf16, (m, H), divisibility=VEC),
            make_fake_tensor(bf16, (H,), divisibility=VEC),
            make_fake_tensor(bf16, (RANK, H), divisibility=VEC),
            make_fake_tensor(bf16, w_up_shape, divisibility=VEC),
            make_fake_tensor(bf16, (m, H), divisibility=VEC),
        )

    def compile_row(
        self, H: int, cluster: int, has_residual: bool, eps: float, use_pdl: bool
    ):
        self._ensure()
        import cutlass.cute as cute

        key = ("row", H, cluster, has_residual, eps, use_pdl)
        if key in self._compiled:
            return self._compiled[key]
        assert self._row_cls is not None
        threads = self.threads_for(H // cluster // VEC)
        kernel = self._row_cls(H, cluster, threads, has_residual, eps, use_pdl)
        self._compiled[key] = cute.compile(
            kernel, *self._fakes(H, False), self._stream(), options="--enable-tvm-ffi"
        )
        logger.debug("compiled CuTe gated RMSNorm row kernel H=%d C=%d", H, cluster)
        return self._compiled[key]

    def compile_tile(self, H: int, has_residual: bool, eps: float, use_pdl: bool):
        self._ensure()
        import cutlass.cute as cute

        cluster = self.tile_cluster(H)
        assert cluster is not None
        key = ("tile", H, cluster, has_residual, eps, use_pdl)
        if key in self._compiled:
            return self._compiled[key]
        assert self._tile_cls is not None
        kernel = self._tile_cls(H, cluster, has_residual, eps, use_pdl)
        self._compiled[key] = cute.compile(
            kernel, *self._fakes(H, True), self._stream(), options="--enable-tvm-ffi"
        )
        logger.debug("compiled CuTe gated RMSNorm tile kernel H=%d C=%d", H, cluster)
        return self._compiled[key]

    def warmup(self, H: int, eps: float, device: torch.device) -> None:
        """Compile every variant the dispatcher can pick (call before CUDA
        graph capture; each variant takes ~1 s)."""
        use_pdl = torch.cuda.get_device_properties(device).major >= 9
        has_tile = self.tile_cluster(H) is not None
        for has_residual in (False, True):
            for c in (8, 4) if has_tile else (8, 4, 2, 1):
                if H % (c * VEC) == 0 and H // c // VEC <= 1024:
                    self.compile_row(H, c, has_residual, eps, use_pdl)
            if has_tile:
                self.compile_tile(H, has_residual, eps, use_pdl)

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        w_norm: torch.Tensor,
        w_down: torch.Tensor,
        w_up: torch.Tensor,
        w_up_frag: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        M, H = x.shape
        props = torch.cuda.get_device_properties(x.device)
        hopper = props.major >= 9
        has_residual = residual is not None
        res_arg = residual if residual is not None else x
        out = torch.empty_like(x)
        kind = self.kind(H, M, props.multi_processor_count) if hopper else "row"
        if kind == "tile":
            fn = self.compile_tile(H, has_residual, float(eps), hopper)
            fn(x, res_arg, w_norm, w_down, w_up_frag, out, self._stream())
        else:
            cluster = (
                self.pick_row_cluster(H, M, props.multi_processor_count)
                if hopper
                else 1
            )
            fn = self.compile_row(H, cluster, has_residual, float(eps), hopper)
            fn(x, res_arg, w_norm, w_down, w_up, out, self._stream())
        return out


_INSTANCE: CuteGatedRMSNorm | None = None


def cute_gated_rmsnorm() -> CuteGatedRMSNorm | None:
    global _INSTANCE
    if not _cute_available():
        return None
    if _INSTANCE is None:
        _INSTANCE = CuteGatedRMSNorm()
    return _INSTANCE
