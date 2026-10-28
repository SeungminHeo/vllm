# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.axk3_reasoning_parser import AXK3ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

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
    # Match DeepSeek-R1 when an opening tag is generated.
    ("<think>R</think>C", "R", "C"),
    # only the first </think> splits
    ("R</think>A</think>B", "R", "A</think>B"),
    # A tool tag cannot end reasoning without </think>.
    ("thinking" + CALL, "thinking" + CALL, None),
    (CALL, CALL, None),
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
    reference = DeepSeekR1ReasoningParser(parser.model_tokenizer)
    assert reference.extract_reasoning(output, make_request()) == (reasoning, content)


@pytest.mark.parametrize(
    "output, reasoning, content",
    [c for c in SPLIT_CASES if "</think>" in c[0]],
)
def test_streaming_matches_split(parser, output, reasoning, content):
    pieces = parser.model_tokenizer.tokenize(output)
    # DeepSeek-R1 suppresses standalone think tokens, including a repeated end.
    expected_content = (content or "").replace("</think>", "")
    assert stream(parser, pieces, IM_END_ID) == (reasoning or "", expected_content)


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


def test_tags(parser):
    assert parser.reasoning_start_str == "<think>"
    assert parser.reasoning_end_str == "</think>"


def test_adjust_request_keeps_special_tokens(parser):
    request = make_request()
    assert parser.adjust_request(request).skip_special_tokens is False


def test_token_ids_come_from_tokenizer():
    tokens = dict(TOKEN_IDS)
    tokens["</think>"] = 7
    parser = AXK3ReasoningParser(K3StubTokenizer(tokens))
    assert parser.is_reasoning_end([5, 7])
    assert parser.is_reasoning_end_streaming([5, 7], [7])
    assert parser.extract_content_ids([5, 7, 6]) == [6]


@pytest.mark.parametrize(
    "ids, expected",
    [
        ([5, 6, THINK_END_ID, 7], 2),
        ([THINK_START_ID, 5, 6, THINK_END_ID, 7], 2),
        ([THINK_END_ID, 7], 0),
        ([5, 6], 2),
        ([], 0),
        ([5, THINK_END_ID, THINK_START_ID, 6], 1),
    ],
)
def test_generated_reasoning_is_counted_with_an_implicit_opening_tag(
    parser, ids, expected
):
    assert parser.count_reasoning_tokens(ids) == expected


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_composed_parser_separates_reasoning_from_tool_calls(k3_tokenizer, tool_choice):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice=tool_choice,
    )
    parser_cls = ParserManager.get_parser("axk3", "axk3", enable_auto_tools=True)
    assert parser_cls is not None
    parser = parser_cls(k3_tokenizer, request.tools)
    parser.adjust_request(request)
    assert not request.skip_special_tokens
    assert request.structured_outputs is not None
    assert request.structured_outputs.structural_tag
    output = "Check the weather.</think>" + CALL
    reasoning, content, calls = parser.parse(output, request, enable_auto_tools=True)
    assert reasoning == "Check the weather."
    assert content is None
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == '{"city": "Tokyo"}'

    streaming = parser_cls(k3_tokenizer, request.tools)
    pieces = k3_tokenizer.tokenize(output)
    deltas = [
        streaming.parse_delta(
            piece,
            k3_tokenizer.ids(piece),
            request,
            prompt_token_ids=[THINK_START_ID],
            finished=i == len(pieces) - 1,
        )
        for i, piece in enumerate(pieces)
    ]
    assert "".join(d.reasoning or "" for d in deltas if d) == reasoning
    assert "".join(d.content or "" for d in deltas if d) == ""
    arguments = []
    for delta in deltas:
        if delta:
            for call in delta.tool_calls:
                assert call.function is not None
                arguments.append(call.function.arguments or "")
    assert "".join(arguments) == calls[0].arguments


@pytest.mark.parametrize(
    "tail",
    [" plain <", " plain <tool_", "<tool_call>ping"]
    + [CALL[:i] for i in range(1, len(CALL))],
)
def test_composed_parser_flushes_content_without_completing_truncated_calls(
    k3_tokenizer, tail
):
    parser_cls = ParserManager.get_parser("axk3", "axk3", enable_auto_tools=True)
    assert parser_cls is not None
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    parser = parser_cls(k3_tokenizer, request.tools)
    expected = tail if "<tool_call>" not in tail else None
    assert parser.parse("R</think>" + tail, request, enable_auto_tools=True) == (
        "R",
        expected,
        [],
    )
    deltas = []
    for piece in k3_tokenizer.tokenize("R</think>" + tail):
        deltas.append(
            parser.parse_delta(
                piece,
                k3_tokenizer.ids(piece),
                request,
                prompt_token_ids=[THINK_START_ID],
                finished=False,
            )
        )
    deltas.append(parser.parse_delta("", [], request, finished=True))
    assert "".join(d.content or "" for d in deltas if d) == (expected or "")
    args = "".join(
        tc.function.arguments or ""
        for d in deltas
        if d
        for tc in d.tool_calls
        if tc.function
    )
    if args:
        with pytest.raises(json.JSONDecodeError):
            json.loads(args)
    assert parser.parse_delta("", [], request, finished=True) is None


@pytest.fixture(params=["byte_bpe", "checkpoint"])
def hf_k3_tokenizer(request):
    from transformers import TokenizersBackend

    if request.param == "checkpoint":
        path = os.environ.get("AXK3_TOKENIZER")
        if not path:
            pytest.skip("Set AXK3_TOKENIZER to a local AX-K3 tokenizer checkpoint")
        return TokenizersBackend.from_pretrained(path, local_files_only=True)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(
        ["weather Tokyo hello before after"],
        trainers.BpeTrainer(
            vocab_size=280,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(TOKEN_IDS),
        ),
    )
    return TokenizersBackend(
        tokenizer_object=raw,
        eos_token="<|im_end|>",
        additional_special_tokens=list(TOKEN_IDS),
    )


def test_token_ids_match_grammar_and_incremental_detokenization(hf_k3_tokenizer):
    from xgrammar import GrammarCompiler, GrammarMatcher, TokenizerInfo

    tok = hf_k3_tokenizer
    parser_cls = ParserManager.get_parser("axk3", "axk3", enable_auto_tools=True)
    assert parser_cls is not None
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    parser_cls(tok, request.tools).adjust_request(request)
    city = '서울 😀 "quote"\nnext'
    native = "before" + CALL.replace("Tokyo", city) + "after"
    assert request.structured_outputs is not None
    grammar = GrammarCompiler(
        TokenizerInfo.from_huggingface(tok)
    ).compile_structural_tag(request.structured_outputs.structural_tag)
    matcher = GrammarMatcher(grammar)
    for token_id in tok.encode(native, add_special_tokens=False):
        assert matcher.accept_token(token_id)
    assert matcher.accept_token(tok.eos_token_id)
    output = "Reasoning</think>" + native
    ids = tok.encode(output, add_special_tokens=False)
    for batch_size in (1, 7, len(ids)):
        parser = parser_cls(tok, request.tools)
        tokens, prefix, read = None, 0, 0
        pending_text, pending_ids, deltas = "", [], []
        for i, token_id in enumerate(ids):
            new, text, prefix, read = detokenize_incrementally(
                tok,
                ids[: i + 1],
                tokens,
                prefix,
                read,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            tokens = (tokens or []) + new
            pending_text += text
            pending_ids.append(token_id)
            if len(pending_ids) == batch_size or i == len(ids) - 1:
                deltas.append(
                    parser.parse_delta(
                        pending_text,
                        pending_ids,
                        request,
                        prompt_token_ids=[tok.convert_tokens_to_ids("<think>")],
                        finished=False,
                    )
                )
                pending_text, pending_ids = "", []
        deltas.append(parser.parse_delta("", [], request, finished=True))
        assert "".join(d.reasoning or "" for d in deltas if d) == "Reasoning"
        assert "".join(d.content or "" for d in deltas if d) == "beforeafter"
        calls = [tc for d in deltas if d for tc in d.tool_calls]
        assert len([tc.id for tc in calls if tc.id]) == 1
        assert all(tc.index == 0 for tc in calls)
        assert json.loads(
            "".join(tc.function.arguments or "" for tc in calls if tc.function)
        ) == {"city": city}


@pytest.mark.parametrize("mode", ["required", "none", "json"])
def test_hidden_reasoning_grammar_allows_thinking_before_constrained_output(
    k3_tokenizer, mode
):
    """Hiding reasoning must not constrain its tokens with the answer grammar."""
    from xgrammar import Grammar
    from xgrammar.testing import _is_grammar_accept_string

    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        include_reasoning=False,
        tool_choice=mode if mode != "json" else "none",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        if mode != "json"
        else None,
        response_format={"type": "json_object"} if mode == "json" else None,
    )
    cls = ParserManager.get_parser(
        tool_parser_name="axk3" if mode != "json" else None,
        reasoning_parser_name="axk3",
        enable_auto_tools=True,
    )
    assert cls is not None
    cls(k3_tokenizer, request.tools).adjust_request(request)
    assert request.structured_outputs is not None
    grammar = Grammar.from_structural_tag(request.structured_outputs.structural_tag)
    answer = CALL if mode == "required" else '{"answer": 2}'
    assert _is_grammar_accept_string(grammar, "Think first.</think>" + answer)
    assert not _is_grammar_accept_string(grammar, answer)
    if mode == "required":
        assert not _is_grammar_accept_string(grammar, "Think.</think>plain answer")
    elif mode == "none":
        assert not _is_grammar_accept_string(grammar, "Think.</think>" + CALL)
    else:
        assert not _is_grammar_accept_string(grammar, "Think.</think>not JSON")
