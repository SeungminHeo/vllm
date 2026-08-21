# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K2 (``axk2``) model entry point."""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    raise NotImplementedError("axk2 rocm support is not yet enabled.")
elif current_platform.is_xpu():
    raise NotImplementedError("axk2 does not yet support XPU.")
else:
    from .nvidia.model import (
        AXK2DecoderLayer,
        AXK2ForCausalLM,
        AXK2GatedRMSNorm,
        AXK2Model,
    )

__all__ = [
    "AXK2ForCausalLM",
    "AXK2Model",
    "AXK2DecoderLayer",
    "AXK2GatedRMSNorm",
]
