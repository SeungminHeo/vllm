# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused A.X-K2 gated RMSNorm Triton kernel.

The kernel keeps the whole pipeline in fp32 and rounds to bf16 once, so it is
compared against an fp32 PyTorch reference with rtol/atol=1e-2, the tolerance
the deepseek_v32 fused-kernel test uses for bf16 norm outputs.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.axk2.gated_rmsnorm import axk2_gated_rmsnorm_triton
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

EPS = 1e-6
RANK = 16

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the gated RMSNorm Triton kernel requires CUDA",
)


@requires_cuda
@pytest.mark.parametrize("hidden_size", [2048, 3072])
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gated_rmsnorm_matches_fp32_reference(
    hidden_size: int, num_tokens: int, with_residual: bool
):
    """The registered op matches fp32 math and adds the residual in place.

    hidden_size=3072 is not a power of two, so it exercises the masked loads.
    """
    set_random_seed(0)
    dev = "cuda"
    x = torch.randn(num_tokens, hidden_size, device=dev, dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    w_norm = torch.randn(hidden_size, device=dev, dtype=torch.bfloat16)
    w_down = torch.randn(RANK, hidden_size, device=dev) / hidden_size**0.5
    w_up = torch.randn(hidden_size, RANK, device=dev) / RANK**0.5
    w_down, w_up = w_down.to(torch.bfloat16), w_up.to(torch.bfloat16)

    h = x.float() if residual is None else x.float() + residual.float()
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * w_norm.float()
    gate = F.silu(y @ w_down.float().T) @ w_up.float().T
    ref = y * torch.sigmoid(gate)

    x_in = x.clone()
    out = torch.ops.vllm.axk2_gated_rmsnorm(x, residual, w_norm, w_down, w_up, EPS)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # The first decoder layer aliases the residual to x, so x must stay intact.
    assert torch.equal(x, x_in)
    if residual is not None:
        torch.testing.assert_close(residual.float(), h, rtol=1e-2, atol=1e-2)


def test_gated_rmsnorm_rejects_residual_needing_a_copy():
    """A reshaped copy of the residual would silently drop its in-place update."""
    hidden_size = 64
    x = torch.randn(4, 2, hidden_size, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, hidden_size, dtype=torch.bfloat16).transpose(0, 1)
    w_norm = torch.ones(hidden_size, dtype=torch.bfloat16)
    w_down = torch.zeros(RANK, hidden_size, dtype=torch.bfloat16)
    w_up = torch.zeros(hidden_size, RANK, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="view"):
        axk2_gated_rmsnorm_triton(x, residual, w_norm, w_down, w_up, EPS)
