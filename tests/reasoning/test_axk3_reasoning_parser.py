# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the A.X K3 (axk3) reasoning parser.

The K3 generation prompt ends with ``<think>``, so model output starts inside
the think block. The split must equal the K3 grammar ``split`` (the RL harness
function). A character-level stub stands in for the K3 tokenizer (no
model files, no network).
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning import axk3_reasoning_parser as axk3_module
from vllm.reasoning.axk3_reasoning_parser import AXK3ReasoningParser

TOKEN_IDS = {
    "<|endoftext|>": 163691,
    "<|im_start|>": 163702,
    "<|im_end|>": 163704,
    "<think>": 163705,
    "</think>": 163706,
    "<tool_call>": 163709,
    "</tool_call>": 163710,
    "<|parameter|>": 163740,
    "<|/parameter|>": 163741,
}
THINK_START_ID = TOKEN_IDS["<think>"]
THINK_END_ID = TOKEN_IDS["</think>"]
TOOL_CALL_START_ID = TOKEN_IDS["<tool_call>"]
IM_END_ID = TOKEN_IDS["<|im_end|>"]
CALL = "<tool_call>get_weather<|parameter|>city\nTokyo<|/parameter|></tool_call>"


class K3StubTokenizer:
    """Character-level stand-in for the K3 tokenizer: each K3 tag / control
    token is one token with its K3 id, every other character is one token."""

    def __init__(self, token_ids: dict[str, int] | None = None):
        tags = dict(TOKEN_IDS if token_ids is None else token_ids)
        self._tags = sorted(tags, key=len, reverse=True)
        self._vocab = dict(tags)
        for c in "\t\n" + "".join(chr(i) for i in range(32, 127)):
            self._vocab.setdefault(c, ord(c))

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._vocab.get(token)

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(text):
            for tag in self._tags:
                if text.startswith(tag, i):
                    out.append(tag)
                    i += len(tag)
                    break
            else:
                out.append(text[i])
                i += 1
        return out

    def ids(self, piece: str) -> list[int]:
        return [self._vocab[piece]] if piece in self._vocab else []


@pytest.fixture
def k3_tokenizer() -> K3StubTokenizer:
    return K3StubTokenizer()


@pytest.fixture
def parser(k3_tokenizer) -> AXK3ReasoningParser:
    return AXK3ReasoningParser(k3_tokenizer)


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "test-model", "messages": [{"role": "user", "content": "Q"}]}
    )


def stream(parser, pieces, terminal_id=None) -> tuple[str, str]:
    """Feed text pieces; with ``terminal_id`` a final ``("", [terminal_id])``
    delta follows, like vLLM V1 (the stop token is not detokenized)."""
    tok = parser.model_tokenizer
    steps = [(p, tok.ids(p)) for p in pieces]
    if terminal_id is not None:
        steps.append(("", [terminal_id]))
    reasoning, content = [], []
    prev_text: str = ""
    prev_ids: list[int] = []
    for delta, ids in steps:
        cur_text, cur_ids = prev_text + delta, prev_ids + ids
        dm = parser.extract_reasoning_streaming(
            prev_text, cur_text, delta, prev_ids, cur_ids, ids
        )
        if dm is not None:
            reasoning.append(dm.reasoning or "")
            content.append(dm.content or "")
        prev_text, prev_ids = cur_text, cur_ids
    return "".join(reasoning), "".join(content)


SPLIT_CASES = [
    ("Reasoning</think>The rest", "Reasoning", "The rest"),
    ("Reasoning</think>", "Reasoning", None),
    ("</think>content", "", "content"),
    ("multi\nline</think>rest\nthat", "multi\nline", "rest\nthat"),
    # a generated <think> is kept verbatim
    ("<think>R</think>C", "<think>R", "C"),
    # only the first </think> splits
    ("R</think>A</think>B", "R", "A</think>B"),
    # implicit end: no </think>, content starts at <tool_call>
    ("thinking" + CALL, "thinking", CALL),
    (CALL, None, CALL),
    # a call inside the reasoning stays reasoning
    ("x" + CALL + "y</think>final", "x" + CALL + "y", "final"),
    # unclosed think: everything is reasoning
    ("still thinking", "still thinking", None),
]


def test_registered_in_tree():
    assert "axk3" in ReasoningParserManager.list_registered()
    assert ReasoningParserManager.get_reasoning_parser("axk3") is AXK3ReasoningParser


@pytest.mark.parametrize("output, reasoning, content", SPLIT_CASES)
def test_extract_reasoning(parser, output, reasoning, content):
    assert parser.extract_reasoning(output, make_request()) == (reasoning, content)
    assert axk3_module._split(output) == (reasoning, content)


@pytest.mark.parametrize(
    "output, reasoning, content",
    [c for c in SPLIT_CASES if "</think>" in c[0]],
)
def test_streaming_matches_split(parser, output, reasoning, content):
    pieces = parser.model_tokenizer.tokenize(output)
    assert stream(parser, pieces, IM_END_ID) == (reasoning or "", content or "")


def test_streaming_end_tag_split_into_text_pieces(parser):
    got = stream(parser, ["Reason", "</th", "ink", ">", "Content"])
    assert got == ("Reason", "Content")
    assert parser.is_reasoning_end_streaming([], [])


def test_streaming_holds_partial_end_tag_until_terminal(k3_tokenizer):
    pieces = list("still thinking </th")
    got = stream(AXK3ReasoningParser(k3_tokenizer), pieces)
    assert got == ("still thinking ", "")
    got = stream(AXK3ReasoningParser(k3_tokenizer), pieces, IM_END_ID)
    assert got == ("still thinking </th", "")


def test_streaming_implicit_end_is_not_detected(parser):
    # Known limit: without </think> the whole streamed turn is reasoning.
    output = "thinking" + CALL
    pieces = parser.model_tokenizer.tokenize(output)
    assert stream(parser, pieces, IM_END_ID) == (output, "")


def test_reasoning_end_from_token_ids(parser):
    prompt = [1, 2, THINK_START_ID]
    assert not parser.is_reasoning_end(prompt)
    assert parser.is_reasoning_end(prompt + [5, THINK_END_ID, 6])
    assert not parser.is_reasoning_end([THINK_END_ID, 5, THINK_START_ID])
    assert parser.is_reasoning_end_streaming([5, THINK_END_ID], [THINK_END_ID])
    assert not parser.is_reasoning_end_streaming([5, 6], [6])


def test_extract_content_ids(parser):
    assert parser.extract_content_ids([5, THINK_END_ID, 6, 7]) == [6, 7]
    assert parser.extract_content_ids([5, THINK_END_ID]) == []
    assert parser.extract_content_ids([5, 6]) == []


def test_count_reasoning_tokens(parser):
    assert parser.count_reasoning_tokens([5, 6, THINK_END_ID, 7]) == 2
    assert parser.count_reasoning_tokens([5, TOOL_CALL_START_ID, 7]) == 1
    assert parser.count_reasoning_tokens([5, 6, 7]) == 3


def test_tags(parser):
    assert parser.reasoning_start_str == "<think>"
    assert parser.reasoning_end_str == "</think>"


def test_adjust_request_keeps_special_tokens(parser):
    request = make_request()
    assert request.skip_special_tokens
    assert parser.adjust_request(request).skip_special_tokens is False


def test_rejects_a_non_k3_tokenizer(monkeypatch):
    monkeypatch.setattr(axk3_module, "_TOKENIZER_OK", {})
    wrong = dict(TOKEN_IDS)
    wrong["</think>"] = 7
    with pytest.raises(RuntimeError, match="not the K3 tokenizer"):
        AXK3ReasoningParser(K3StubTokenizer(wrong))
