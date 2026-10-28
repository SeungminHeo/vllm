# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused grouped top-k for routing without a correction bias.

Groups are ranked by their best expert score, as in the `grouped_topk`
reference; the fused CUDA kernel only implements the biased variant, which
ranks groups by the sum of their two best scores.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_MAX_EXPERTS = 1024
_MAX_TOPK = 32


def _is_power_of_2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def can_use_bias_free_grouped_topk(
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
) -> bool:
    num_experts = gating_output.shape[-1]
    return (
        current_platform.is_cuda()
        and e_score_correction_bias is None
        and scoring_func in ("sigmoid", "softmax")
        and gating_output.ndim == 2
        and gating_output.is_contiguous()
        and gating_output.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and _is_power_of_2(num_experts)
        and num_experts <= _MAX_EXPERTS
        and _is_power_of_2(num_expert_group)
        and num_experts % num_expert_group == 0
        and 0 < topk_group <= num_expert_group
        and 0 < topk <= min(_MAX_TOPK, topk_group * num_experts // num_expert_group)
    )


if current_platform.is_cuda():

    @triton.jit
    def _bias_free_grouped_topk_kernel(
        gating_output_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        routed_scaling_factor,
        NUM_EXPERTS: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        TOPK_GROUP: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SIGMOID: tl.constexpr,
        RENORMALIZE: tl.constexpr,
    ):
        GROUP_SIZE: tl.constexpr = NUM_EXPERTS // NUM_GROUPS
        row = tl.program_id(0).to(tl.int64)
        expert_offsets = tl.arange(0, NUM_EXPERTS)
        logits = tl.load(gating_output_ptr + row * NUM_EXPERTS + expert_offsets)
        logits = logits.to(tl.float32)
        if SIGMOID:
            scores = tl.sigmoid(logits)
        else:
            scores = tl.exp(logits - tl.max(logits, axis=0))
            scores = scores / tl.sum(scores, axis=0)
        # The reference scores in the input dtype; round the same way so that
        # selection matches it.
        scores = scores.to(gating_output_ptr.dtype.element_ty).to(tl.float32)

        group_offsets = tl.arange(0, NUM_GROUPS)
        group_scores = tl.max(tl.reshape(scores, (NUM_GROUPS, GROUP_SIZE)), axis=1)
        in_selected_group = expert_offsets < 0
        for _ in tl.static_range(TOPK_GROUP):
            max_value = tl.max(group_scores, axis=0)
            group_id = tl.min(
                tl.where(group_scores == max_value, group_offsets, NUM_GROUPS), axis=0
            )
            in_selected_group = in_selected_group | (
                expert_offsets // GROUP_SIZE == group_id
            )
            group_scores = tl.where(
                group_offsets == group_id, -float("inf"), group_scores
            )

        current = tl.where(in_selected_group, scores, -float("inf"))
        topk_offsets = tl.arange(0, BLOCK_K)
        selected_weights = tl.zeros([BLOCK_K], dtype=tl.float32)
        selected_ids = tl.zeros([BLOCK_K], dtype=tl.int32)
        for slot in tl.static_range(TOPK):
            max_value = tl.max(current, axis=0)
            expert_id = tl.min(
                tl.where(current == max_value, expert_offsets, NUM_EXPERTS), axis=0
            ).to(tl.int32)
            is_slot = topk_offsets == slot
            selected_weights = tl.where(is_slot, max_value, selected_weights)
            selected_ids = tl.where(is_slot, expert_id, selected_ids)
            current = tl.where(expert_offsets == expert_id, -float("inf"), current)

        if RENORMALIZE:
            selected_weights /= tl.sum(selected_weights, axis=0)
        selected_weights *= routed_scaling_factor

        output_mask = topk_offsets < TOPK
        output_offsets = row * TOPK + topk_offsets
        tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


def bias_free_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert e_score_correction_bias is None
    num_tokens, num_experts = gating_output.shape
    topk_weights = gating_output.new_empty((num_tokens, topk), dtype=torch.float32)
    topk_ids = gating_output.new_empty((num_tokens, topk), dtype=torch.int32)
    if num_tokens > 0:
        _bias_free_grouped_topk_kernel[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
            NUM_EXPERTS=num_experts,
            NUM_GROUPS=num_expert_group,
            TOPK_GROUP=topk_group,
            TOPK=topk,
            BLOCK_K=triton.next_power_of_2(topk),
            SIGMOID=scoring_func == "sigmoid",
            RENORMALIZE=renormalize,
            num_warps=1,
        )
    return topk_weights, topk_ids
