# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class AXK3ReasoningParser(DeepSeekR1ReasoningParser):
    """DeepSeek-R1 thinking format exposed as ``--reasoning-parser axk3``."""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # K3 marks </think> as special, including turns without tools.
        request.skip_special_tokens = False
        return request

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        # The opening <think> is in K3's prompt, not its generated output.
        count = 0
        for token_id in token_ids:
            if token_id == self.end_token_id:
                break
            if token_id != self.start_token_id:
                count += 1
        return count
