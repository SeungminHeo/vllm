# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated residual stream of the A.X-K3 family."""

import torch
import torch.nn.functional as F
from torch import nn

from vllm.models.axk3.gated_residual_ops import (
    gated_residual_read,
    gated_residual_write,
)


class AXK3GatedResidual(nn.Module):
    """Read/write gates over a residual stream widened to several branches.

    The stream is ``[..., num_streams * hidden_size]``, branch-major. ``read``
    replaces the pre-norm: it RMS-normalizes each branch, gates it elementwise
    through a low-rank projection and averages the branches into the block
    input. ``write`` replaces the residual add: the block output goes to every
    branch scaled by one gate per branch.

    Parameter names follow the checkpoint. The gain and both gates are applied
    in fp32, as in the reference implementation. On CUDA the formulas run in
    the fused kernels of ``gated_residual_ops``; the PyTorch versions below
    (``_read`` / ``_write``) define the result and serve the other devices.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        rank: int,
        eps: float,
        has_write: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.eps = eps
        wide = num_streams * hidden_size
        self.gamma = nn.Parameter(torch.ones(wide))
        self.read_down = nn.Linear(wide, rank, bias=False)
        self.read_up = nn.Linear(rank, wide, bias=False)
        self.write_proj: nn.Linear | None = None
        self.write_bias: nn.Parameter | None = None
        if has_write:
            self.write_proj = nn.Linear(wide, num_streams, bias=False)
            self.write_bias = nn.Parameter(
                torch.zeros(num_streams, dtype=torch.float32)
            )

    def _read(self, stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        n, h = self.num_streams, self.hidden_size
        branches = stream.unflatten(-1, (n, h)).float()
        rstd = torch.rsqrt(branches.square().mean(dim=-1, keepdim=True) + self.eps)
        normed = (branches * rstd * self.gamma.view(n, h).float()).to(stream.dtype)
        normed = normed.flatten(-2)

        gate = torch.sigmoid(self.read_up(F.silu(self.read_down(normed) / n)).float())
        block_input = (gate * normed.float()).unflatten(-1, (n, h)).mean(dim=-2)

        write_gate = None
        if self.write_proj is not None:
            write_gate = 2.0 * torch.sigmoid(
                (self.write_proj(normed) / n).float() + self.write_bias
            )
        return block_input.to(stream.dtype), write_gate

    def _write(
        self,
        stream: torch.Tensor,
        block_output: torch.Tensor,
        write_gate: torch.Tensor,
    ) -> torch.Tensor:
        n, h = self.num_streams, self.hidden_size
        scaled = write_gate.unsqueeze(-1) * block_output.float().unsqueeze(-2)
        return (stream.unflatten(-1, (n, h)) + scaled.to(stream.dtype)).flatten(-2)

    def _fused_read(
        self,
        stream: torch.Tensor,
        block_output: torch.Tensor | None = None,
        write_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        return gated_residual_read(
            stream,
            self.gamma,
            self.read_down.weight,
            self.read_up.weight,
            self.write_proj.weight if self.write_proj is not None else None,
            self.write_bias,
            self.num_streams,
            self.eps,
            block_output,
            write_gate,
        )

    def read(self, stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Contract the stream into the block input.

        Returns:
            The block input ``[..., hidden_size]`` and the fp32 write gate
            ``[..., num_streams]`` (``None`` for a read-only module).

        """
        if not stream.is_cuda:
            return self._read(stream)
        _, block_input, write_gate = self._fused_read(stream)
        return block_input, write_gate

    def write(
        self,
        stream: torch.Tensor,
        block_output: torch.Tensor,
        write_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not stream.is_cuda:
            return self._write(stream, block_output, write_gate)
        return gated_residual_write(stream, block_output, write_gate, self.num_streams)

    def write_read(
        self,
        stream: torch.Tensor,
        block_output: torch.Tensor,
        write_gate: torch.Tensor,
        reader: "AXK3GatedResidual",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write this block's output, then read the stream for ``reader``.

        Returns:
            The written stream, then what ``reader.read`` returns for it.

        """
        if not stream.is_cuda:
            stream = self._write(stream, block_output, write_gate)
            return stream, *reader._read(stream)
        return reader._fused_read(stream, block_output, write_gate)
