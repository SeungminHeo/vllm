# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused A.X-K3 gated residual kernels.

The kernels round where the PyTorch formulas of ``AXK3GatedResidual`` round
(bf16 stream, normalized stream and projection outputs, fp32 gates) and differ
from them only by the summation order of the reductions and dot products, so
an output is either equal or off by a bf16 ulp or two. The comparison uses the
tolerance of the A.X-K2 gated RMSNorm test, plus a bound on how many elements
differ at all.
"""

import pytest
import torch

from vllm.models.axk3 import gated_residual_ops as ops
from vllm.models.axk3.gated_residual import AXK3GatedResidual
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the gated residual Triton kernels require CUDA",
)

# (hidden_size, num_streams, rank): the A.X-K3 Flash shape, and one whose hidden
# size is not a power of two so the masked loads are exercised.
_SHAPES = [(4096, 4, 64), (1536, 4, 16)]
# Rows on both sides of the switch from the FMA projection to the GEMM path.
_ROWS = [1, 7, ops.PROJECT_MAX_ROWS, ops.PROJECT_MAX_ROWS + 1, 300]


def _module(hidden_size, num_streams, rank, has_write=True):
    module = AXK3GatedResidual(
        hidden_size, num_streams, rank, eps=1e-6, has_write=has_write
    ).to("cuda", torch.bfloat16)
    with torch.no_grad():
        module.gamma.uniform_(0.3, 4.0)
        module.read_down.weight.normal_(std=0.02)
        module.read_up.weight.normal_(std=0.05)
        if has_write:
            module.write_proj.weight.normal_(std=0.02)
            module.write_bias.data = torch.randn(num_streams, device="cuda")
    return module


def _assert_close(actual, expected, max_mismatch=0.01):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    if expected.numel() >= 1024:
        assert (actual != expected).float().mean().item() <= max_mismatch


@requires_cuda
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("num_rows", _ROWS)
@torch.inference_mode()
def test_gated_residual_kernels_match_formulas(shape, num_rows):
    set_random_seed(0)
    hidden_size, num_streams, _ = shape
    writer, reader = _module(*shape), _module(*shape)
    stream = torch.randn(
        num_rows, num_streams * hidden_size, device="cuda", dtype=torch.bfloat16
    )
    block_output = torch.randn(
        num_rows, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    stream_in = stream.clone()

    block_input, write_gate = writer.read(stream)
    expected_input, expected_gate = writer._read(stream)
    _assert_close(block_input, expected_input)
    assert write_gate.dtype == torch.float32
    torch.testing.assert_close(write_gate, expected_gate, rtol=1e-2, atol=1e-2)

    # The write is the same fp32 arithmetic, rounded at the same two places.
    expected_written = writer._write(stream, block_output, expected_gate)
    written = writer.write(stream, block_output, expected_gate)
    assert torch.equal(written, expected_written)

    next_input, next_gate = reader._read(expected_written)
    written, block_input, write_gate = writer.write_read(
        stream, block_output, expected_gate, reader
    )
    assert torch.equal(written, expected_written)
    _assert_close(block_input, next_input)
    torch.testing.assert_close(write_gate, next_gate, rtol=1e-2, atol=1e-2)
    assert torch.equal(stream, stream_in)


@requires_cuda
@torch.inference_mode()
def test_gated_residual_read_only_and_leading_dims():
    """The final read has no write gate; leading dimensions are kept."""
    set_random_seed(0)
    hidden_size, num_streams, rank = 1536, 4, 16
    module = _module(hidden_size, num_streams, rank, has_write=False)
    stream = torch.randn(
        2, 3, num_streams * hidden_size, device="cuda", dtype=torch.bfloat16
    )

    block_input, write_gate = module.read(stream)
    expected_input, _ = module._read(stream)
    assert write_gate is None
    assert block_input.shape == (2, 3, hidden_size)
    _assert_close(block_input, expected_input)


@requires_cuda
@torch.inference_mode()
def test_gated_residual_rows_do_not_depend_on_the_batch():
    """Up to PROJECT_MAX_ROWS a row's result is the same alone or in a batch."""
    set_random_seed(0)
    hidden_size, num_streams, rank = 4096, 4, 64
    module = _module(hidden_size, num_streams, rank)
    num_rows = min(8, ops.PROJECT_MAX_ROWS)
    stream = torch.randn(
        num_rows, num_streams * hidden_size, device="cuda", dtype=torch.bfloat16
    )

    block_input, write_gate = module.read(stream)
    for i in range(num_rows):
        row_input, row_gate = module.read(stream[i : i + 1])
        assert torch.equal(row_input[0], block_input[i])
        assert torch.equal(row_gate[0], write_gate[i])
