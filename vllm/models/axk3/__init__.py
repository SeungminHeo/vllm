# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K3 (``axk3``) model entry point."""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    raise NotImplementedError("axk3 rocm support is not yet enabled.")
elif current_platform.is_xpu():
    raise NotImplementedError("axk3 does not yet support XPU.")
else:
    from .nvidia.model import (
        AXK3ForCausalLM,
        AXK3GatedResidualDecoderLayer,
        AXK3GatedResidualModel,
    )

__all__ = [
    "AXK3ForCausalLM",
    "AXK3GatedResidualDecoderLayer",
    "AXK3GatedResidualModel",
]
