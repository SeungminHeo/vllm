# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official vLLM Benchmark for Gated RMSNorm + Activation Quantization.

Compares:
1. Unfused PyTorch Reference Implementation
2. Fused Gated RMSNorm + Act Quantization Triton Kernel
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

from vllm.models.axk2.common.fused_gated_rmsnorm_quant import (
    fused_gated_rmsnorm_quant,
    fused_gated_rmsnorm_quant_ref,
)
from vllm.platforms import current_platform


@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype
    quant_type: str

    def description(self) -> str:
        return (
            f"N {self.num_tokens:<4} | "
            f"D {self.hidden_size:<4} | "
            f"DT {str(self.dtype).split('.')[-1]:<8} | "
            f"Q {self.quant_type}"
        )


def get_bench_params() -> list[bench_params_t]:
    """Test configurations covering realistic serving and prefill workloads."""
    NUM_TOKENS = [1, 16, 64, 256, 1024, 2048, 4096]
    HIDDEN_SIZES = [2048, 4096, 5120, 8192]
    DTYPES = [torch.bfloat16]
    QUANT_TYPES = ["fp8_per_token", "int8_per_token"]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, DTYPES, QUANT_TYPES)
    return [
        bench_params_t(num_tokens=x[0], hidden_size=x[1], dtype=x[2], quant_type=x[3])
        for x in combinations
    ]


def unfused_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    quant_type: str,
):
    """Unfused baseline: separate PyTorch ops with multiple HBM round trips."""
    return fused_gated_rmsnorm_quant_ref(
        x=x,
        weight=weight,
        z=z,
        residual=residual,
        quant_type=quant_type,
        norm_before_gate=True,
        gate_activation="silu",
    )


def fused_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    quant_type: str,
):
    """Fused Triton implementation: single-pass SRAM/register computation."""
    return fused_gated_rmsnorm_quant(
        x=x,
        weight=weight,
        z=z,
        residual=residual,
        quant_type=quant_type,
        norm_before_gate=True,
        gate_activation="silu",
    )


def bench_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    residual: torch.Tensor,
    quant_type: str,
    label: str,
    sub_label: str,
    fn: Callable,
    description: str,
) -> TMeasurement:
    min_run_time = 0.5

    globals_dict = {
        "x": x,
        "weight": weight,
        "z": z,
        "residual": residual,
        "quant_type": quant_type,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(x, weight, z, residual, quant_type)",
        globals=globals_dict,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(params: bench_params_t, label: str, sub_label: str) -> Iterable[TMeasurement]:
    device = torch.device(
        current_platform.device_type if torch.accelerator.is_available() else "cpu"
    )
    x = torch.randn(
        params.num_tokens, params.hidden_size, dtype=params.dtype, device=device
    )
    z = torch.randn(
        params.num_tokens, params.hidden_size, dtype=params.dtype, device=device
    )
    weight = torch.ones(params.hidden_size, dtype=params.dtype, device=device)
    residual = torch.randn(
        params.num_tokens, params.hidden_size, dtype=params.dtype, device=device
    )

    timers = []

    # 1. Unfused PyTorch Baseline
    timers.append(
        bench_fn(
            x,
            weight,
            z,
            residual,
            params.quant_type,
            label,
            sub_label,
            unfused_impl,
            "1_unfused_pytorch",
        )
    )

    # 2. Fused Triton Kernel
    timers.append(
        bench_fn(
            x,
            weight,
            z,
            residual,
            params.quant_type,
            label,
            sub_label,
            fused_impl,
            "2_fused_triton",
        )
    )

    return timers


def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.colorize()
    compare.print()


def main():
    device = current_platform.device_type if torch.accelerator.is_available() else "cpu"
    if device != "cpu":
        torch.set_default_device(device)
        device_name = torch.accelerator.get_device_name()
    else:
        device_name = "CPU"

    bench_params = get_bench_params()

    print("=" * 80)
    print(f"vLLM Fused Kernel Benchmark on: {device_name}")
    print(f"Testing {len(bench_params)} benchmark configurations...")
    print("=" * 80)

    timers = []
    for bp in tqdm(bench_params, desc="Benchmarking"):
        result_timers = bench(bp, "GatedRMSNorm+Quant", bp.description())
        timers.extend(result_timers)

    print("\n" + "=" * 80)
    print("FINAL PYTORCH BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print_timers(timers)


if __name__ == "__main__":
    main()
