# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for A.X K3 (``model_type: axk3``), ``--reasoning-parser axk3``.

The K3 generation prompt ends with ``<|im_start|>assistant\\n<think>``, so the
output starts *inside* the think block and only ``</think>`` is generated. The
split is the RL harness function ``split`` of the K3 grammar (``_split`` below,
copied from it together with the K3 token ids):

* split at the first ``</think>``; the reasoning is kept verbatim (a generated
  ``<think>`` is not removed) and an empty content becomes None;
* no ``</think>`` but a ``<tool_call>``: the content starts at the
  ``<tool_call>`` (implicit end);
* neither: everything is reasoning.

Built-in parsers (deepseek_r1, qwen3, ...) differ from this split on at least
one of: ``<think>`` inside a value, a second ``</think>``, the implicit end.
Use this parser together with ``--tool-call-parser axk3``.

Streaming ends the reasoning only at the first ``</think>`` in the text. The
implicit end cannot be decided while streaming (a ``</think>`` may still
follow), so such turns stream entirely as reasoning. A trailing prefix of
``</think>`` is held back until the stream ends with a K3 terminal token id;
then it is flushed as reasoning (equal to the non-streaming split). Streams
that end without a terminal id (``finish_reason=length``, stop strings) can
lose that tail.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

logger = init_logger(__name__)

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
THINK_START_ID = 163705
THINK_END_ID = 163706
TOOL_CALL_START_ID = 163709
# K3 terminal ids (<|im_end|> 163704, <|endoftext|> 163691) = the RL E set.
K3_EOS_IDS = (163704, 163691)


def _split(text: str | None) -> tuple[str | None, str | None]:
    """(reasoning, content) of a generated turn body, the K3 grammar ``split``
    with ``thinking_enabled=True, implicit_end_on_tool_call=True``."""
    if text is None:
        return None, None
    if THINK_END in text:
        r, _, c = text.partition(THINK_END)
        return r, (c or None)
    i = text.find(TOOL_CALL_START)
    if i >= 0:
        return (text[:i] or None), (text[i:] or None)
    return text, None


_LOGGED: set[str] = set()
_TOKENIZER_OK: dict[int, bool] = {}


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ends ``text``."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


def _ends_with_terminal(delta_token_ids: Sequence[int] | None) -> bool:
    """True if the delta ends with a K3 terminal id (end of a stopped stream)."""
    ids = list(delta_token_ids or ())
    return bool(ids) and int(ids[-1]) in K3_EOS_IDS


def _check_tokenizer(tokenizer: Any) -> None:
    """Fail closed if the tokenizer does not map the K3 think/tool tags to the
    K3 ids (checked once per tokenizer object)."""
    if tokenizer is None:
        return
    key = id(tokenizer)
    if _TOKENIZER_OK.get(key):
        return
    want = {
        THINK_START: THINK_START_ID,
        THINK_END: THINK_END_ID,
        TOOL_CALL_START: TOOL_CALL_START_ID,
    }
    conv = getattr(tokenizer, "convert_tokens_to_ids", None)
    try:
        if conv is not None:
            got = {t: conv(t) for t in want}
        else:
            vocab = tokenizer.get_vocab()
            got = {t: vocab.get(t) for t in want}
    except Exception:  # no way to check: warn only
        if "tok_check" not in _LOGGED:
            _LOGGED.add("tok_check")
            logger.warning(
                "[axk3] could not verify the K3 think/tool token ids on this tokenizer"
            )
        return
    if got != want:
        raise RuntimeError(
            f"[axk3] tokenizer is not the K3 tokenizer: token ids {got} != {want}"
        )
    _TOKENIZER_OK[key] = True


class AXK3ReasoningParser(ReasoningParser):
    """K3 think split = the RL harness ``split``."""

    # Kept explicit for forks whose parser base classes consult it.
    engine_based_streaming = False

    def __init__(self, tokenizer: Any, *args: Any, **kwargs: Any):
        super().__init__(tokenizer, *args, **kwargs)
        _check_tokenizer(tokenizer)
        self.start_token_id = THINK_START_ID
        self.end_token_id = THINK_END_ID
        # Streaming state
        self._sent_reasoning_idx = 0
        self._ended = False
        self._text_mode = False

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def adjust_request(self, request: Any) -> Any:
        """Always decode with ``skip_special_tokens=False``, as the RL harness
        does, also for requests without tools (final answers).
        ``spaces_between_special_tokens`` keeps its default."""
        if hasattr(request, "skip_special_tokens"):
            request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """True if the last think tag in ``input_ids`` is ``</think>``. The K3
        prompt ends with ``<think>``, so a stream starts in the reasoning
        phase."""
        for t in reversed(list(input_ids)):
            if t == THINK_START_ID:
                return False
            if t == THINK_END_ID:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._text_mode:
            # Front-end path: end only when extract_reasoning_streaming found
            # '</think>' in the text (also when it was split into pieces).
            return self._ended
        return THINK_END_ID in list(delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        ids = list(input_ids)
        if THINK_END_ID not in ids[:-1]:
            return []
        return ids[ids.index(THINK_END_ID) + 1 :]

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Usage approximation (not a verdict): tokens before the first
        ``</think>`` id, else before the first ``<tool_call>`` id, else all."""
        ids = list(token_ids)
        if THINK_END_ID in ids:
            return ids.index(THINK_END_ID)
        if TOOL_CALL_START_ID in ids:
            return ids.index(TOOL_CALL_START_ID)
        return len(ids)

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        try:
            reasoning, content = _split(model_output)
            return reasoning, content
        except Exception:  # split does not raise on str; safety net
            if "split" not in _LOGGED:
                _LOGGED.add("split")
                logger.exception(
                    "[axk3] unexpected error in the K3 split; returning the "
                    "whole output as content"
                )
            return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._text_mode = True
        current_text = current_text or ""
        if self._ended:
            return DeltaMessage(content=delta_text) if delta_text else None
        idx = current_text.find(THINK_END)
        if idx >= 0:
            reasoning = current_text[self._sent_reasoning_idx : idx]
            content = current_text[idx + len(THINK_END) :]
            self._sent_reasoning_idx = idx
            self._ended = True
            return DeltaMessage(reasoning=reasoning or None, content=content or None)
        if _ends_with_terminal(delta_token_ids):
            # End of stream: a held-back '</think>' prefix is reasoning too.
            sendable = len(current_text)
        else:
            sendable = len(current_text) - _partial_tag_overlap(current_text, THINK_END)
        if sendable > self._sent_reasoning_idx:
            reasoning = current_text[self._sent_reasoning_idx : sendable]
            self._sent_reasoning_idx = sendable
            return DeltaMessage(reasoning=reasoning)
        return None
