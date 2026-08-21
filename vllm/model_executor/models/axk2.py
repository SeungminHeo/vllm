# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only A.X K2 model.
Re-exports from vllm.models.axk2 with padding helpers.
"""

import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.axk2 import (
    AXK2DecoderLayer,
    AXK2ForCausalLM,
    AXK2GatedRMSNorm,
    AXK2Model,
)
from vllm.transformers_utils.configs.axk2 import AXK2Config

_MLA_HEAD_SIZE_TARGETS = {
    576: dict(kv_lora=512, qk_rope=64, qk_nope=None, v_head=None),
}
_MLA_DSA_HEAD_SIZE_TARGETS = {
    576: dict(kv_lora=512, qk_rope=64, qk_nope=128, v_head=128),
}


class PartialRMSNorm(RMSNorm):
    """RMSNorm applied only to the first `actual_size` dimensions."""

    def __init__(self, actual_size: int, full_size: int, eps: float = 1e-6):
        assert full_size >= actual_size, (
            f"full_size ({full_size}) must be >= actual_size ({actual_size})"
        )
        super().__init__(actual_size, eps=eps)
        self.actual_size = actual_size
        self.full_size = full_size
        self.pad_size = full_size - actual_size

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if self.pad_size == 0:
            if residual is None:
                return super().forward(x)
            return super().forward(x, residual)

        assert x.shape[-1] == self.full_size, (
            f"PartialRMSNorm expected last dim == {self.full_size}, got {x.shape[-1]}"
        )

        x_head = x[..., : self.actual_size].contiguous()
        x_tail = x[..., self.actual_size :]

        if residual is None:
            out_head = super().forward(x_head)
            out = torch.cat([out_head, x_tail], dim=-1)
            return out

        assert residual.shape[-1] == self.full_size, (
            f"residual last dim ({residual.shape[-1]}) must match "
            f"full_size ({self.full_size})"
        )
        res_head = residual[..., : self.actual_size].contiguous()
        res_tail = residual[..., self.actual_size :]

        out_head, new_res_head = super().forward(x_head, res_head)
        new_res_tail = x_tail + res_tail

        out = torch.cat([out_head, new_res_tail], dim=-1)
        new_residual = torch.cat([new_res_head, new_res_tail], dim=-1)
        return out, new_residual

    def extra_repr(self) -> str:
        return (
            f"actual_size={self.actual_size}, "
            f"full_size={self.full_size}, "
            f"eps={self.variance_epsilon}"
        )


def _mla_pad_dims(
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    target_head_size: int = 576,
    is_dsa: bool = False,
) -> tuple[int, int, int, int]:
    spec = (
        _MLA_DSA_HEAD_SIZE_TARGETS[target_head_size]
        if is_dsa
        else _MLA_HEAD_SIZE_TARGETS[target_head_size]
    )
    return (
        max(qk_nope_head_dim, spec["qk_nope"] or qk_nope_head_dim),
        max(qk_rope_head_dim, spec["qk_rope"] or qk_rope_head_dim),
        max(v_head_dim, spec["v_head"] or v_head_dim),
        max(kv_lora_rank, spec["kv_lora"] or kv_lora_rank),
    )


def _pick_mla_target(config: AXK2Config) -> tuple[int, bool]:
    if hasattr(config, "index_topk"):
        return 576, True
    return 576, False


def _pad_q_b_proj_native_to_eff(
    weight: torch.Tensor,
    num_heads: int,
    native_qk_nope_head_dim: int,
    native_qk_rope_head_dim: int,
    eff_qk_nope_head_dim: int,
    eff_qk_rope_head_dim: int,
) -> torch.Tensor:
    in_dim = weight.shape[-1]
    native_head = native_qk_nope_head_dim + native_qk_rope_head_dim
    eff_head = eff_qk_nope_head_dim + eff_qk_rope_head_dim
    q3 = weight.view(num_heads, native_head, in_dim)
    out = weight.new_zeros((num_heads, eff_head, in_dim))
    out[:, :native_qk_nope_head_dim] = q3[:, :native_qk_nope_head_dim]
    out[
        :,
        eff_qk_nope_head_dim : eff_qk_nope_head_dim + native_qk_rope_head_dim,
    ] = q3[:, native_qk_nope_head_dim:]
    return out.reshape(num_heads * eff_head, in_dim).contiguous()


def _pad_fused_q_b_proj_native_to_eff(
    weight: torch.Tensor,
    num_heads: int,
    native_qk_nope_head_dim: int,
    native_qk_rope_head_dim: int,
    eff_qk_nope_head_dim: int,
    eff_qk_rope_head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:
    in_dim = weight.shape[-1]
    native_qk = native_qk_nope_head_dim + native_qk_rope_head_dim
    eff_qk = eff_qk_nope_head_dim + eff_qk_rope_head_dim
    native_ph = native_qk + v_head_dim
    eff_ph = eff_qk + v_head_dim
    w = weight.view(num_heads, native_ph, in_dim)
    out = weight.new_zeros((num_heads, eff_ph, in_dim))
    out[:, :native_qk_nope_head_dim] = w[:, :native_qk_nope_head_dim]
    out[:, eff_qk_nope_head_dim : eff_qk_nope_head_dim + native_qk_rope_head_dim] = w[
        :, native_qk_nope_head_dim:native_qk
    ]
    out[:, eff_qk : eff_qk + v_head_dim] = w[:, native_qk:native_ph]
    return out.reshape(num_heads * eff_ph, in_dim).contiguous()


def _pad_kv_b_proj_native_to_eff(
    weight: torch.Tensor,
    num_heads: int,
    native_qk_nope_head_dim: int,
    native_v_head_dim: int,
    native_kv_lora_rank: int,
    eff_qk_nope_head_dim: int,
    eff_v_head_dim: int,
    eff_kv_lora_rank: int,
) -> torch.Tensor:
    native_out = native_qk_nope_head_dim + native_v_head_dim
    eff_out = eff_qk_nope_head_dim + eff_v_head_dim
    w3 = weight.view(num_heads, native_out, native_kv_lora_rank)
    out = weight.new_zeros((num_heads, eff_out, eff_kv_lora_rank))
    out[
        :,
        :native_qk_nope_head_dim,
        :native_kv_lora_rank,
    ] = w3[:, :native_qk_nope_head_dim]
    out[
        :,
        eff_qk_nope_head_dim : eff_qk_nope_head_dim + native_v_head_dim,
        :native_kv_lora_rank,
    ] = w3[:, native_qk_nope_head_dim:]
    return out.reshape(num_heads * eff_out, eff_kv_lora_rank).contiguous()


def _make_padded_weight_loader(native_shape, pad_fn, original_loader):
    native_shape = tuple(native_shape)

    def padded_loader(param, loaded_weight, *args, **kwargs):
        if tuple(loaded_weight.shape) == native_shape:
            loaded_weight = pad_fn(loaded_weight)
        return original_loader(param, loaded_weight, *args, **kwargs)

    return padded_loader


__all__ = [
    "PartialRMSNorm",
    "AXK2GatedRMSNorm",
    "AXK2DecoderLayer",
    "AXK2Model",
    "AXK2ForCausalLM",
    "_MLA_HEAD_SIZE_TARGETS",
    "_MLA_DSA_HEAD_SIZE_TARGETS",
    "_mla_pad_dims",
    "_pick_mla_target",
    "_pad_q_b_proj_native_to_eff",
    "_pad_fused_q_b_proj_native_to_eff",
    "_pad_kv_b_proj_native_to_eff",
]
