# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
import random
from unittest.mock import MagicMock

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.axk3_tool_parser import AXK3ToolParser

TOOLS = [
    {
        "name": "write_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {
                        "overwrite": {"type": "boolean"},
                        "mode": {"type": "string"},
                    },
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "typed",
        "parameters": {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "f": {"type": "number"},
                "b": {"type": "boolean"},
                "arr": {"type": "array", "items": {"type": "string"}},
                "obj": {"type": "object"},
                "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "untyped": {"description": "no type given"},
            },
        },
    },
    {"name": "ping", "parameters": {"type": "object", "properties": {}}},
]


def _request(tool_choice="auto"):
    return ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=[{"type": "function", "function": t} for t in TOOLS],
        tool_choice=tool_choice,
    )


def _call(name, *params):
    body = "".join(f"<|parameter|>{k}\n{v}<|/parameter|>" for k, v in params)
    return f"<tool_call>{name}{body}</tool_call>"


@pytest.fixture
def parser():
    return AXK3ToolParser(MagicMock())


def _both(parser_factory, text, request):
    """Non-streaming result, then the same text streamed char by char and in
    random chunks; all three must agree exactly."""
    content, calls = run_tool_extraction(
        parser_factory(), text, request, streaming=False
    )
    for chunks in (list(text), _random_chunks(text, 0), _random_chunks(text, 1)):
        p = parser_factory()
        s_content, s_calls = run_tool_extraction(
            p, chunks, request, streaming=True, assert_one_tool_per_delta=False
        )
        tail = p.finish_streaming()
        if tail:
            s_content = (s_content or "") + (tail.content or "")
        assert p.get_remaining_unstreamed_args() == ""
        assert [c.function.name for c in s_calls] == [c.function.name for c in calls]
        # byte-identical argument strings, not merely equal JSON
        assert [c.function.arguments for c in s_calls] == [
            c.function.arguments for c in calls
        ]
        if calls:
            assert (s_content or None) == (content or None) or (
                (s_content or "").strip() == "" and content is None
            )
    return content, calls


def _random_chunks(text, seed):
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 12)
        out.append(text[i : i + n])
        i += n
    return out


def test_no_tool_call(parser):
    content, calls = run_tool_extraction(parser, "안녕하세요", _request())
    assert content == "안녕하세요" and calls == []


def test_prompt_example():
    text = _call(
        "write_file",
        ("path", "hello.py"),
        ("content", 'print("안녕하세요")\n'),
        ("options", '{"overwrite": true, "mode": "0644"}'),
    )
    content, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, _request())
    assert content is None
    assert len(calls) == 1 and calls[0].function.name == "write_file"
    assert json.loads(calls[0].function.arguments) == {
        "path": "hello.py",
        "content": 'print("안녕하세요")\n',
        "options": {"overwrite": True, "mode": "0644"},
    }


def test_types_follow_schema():
    text = _call(
        "typed",
        ("s", "007"),
        ("i", "42"),
        ("f", "1.5"),
        ("b", "false"),
        ("arr", '["a", "b\\"c"]'),
        ("obj", '{"k": [1, null]}'),
        ("maybe", "null"),
        ("untyped", "123"),
    )
    _, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, _request())
    assert json.loads(calls[0].function.arguments) == {
        "s": "007",
        "i": 42,
        "f": 1.5,
        "b": False,
        "arr": ["a", 'b"c'],
        "obj": {"k": [1, None]},
        "maybe": None,
        "untyped": 123,
    }


def test_string_or_null_keeps_text_and_unknown_params():
    text = _call(
        "typed",
        ("maybe", "plain text"),
        ("unknown", "not json"),
        ("unknown2", "[1, 2]"),
    )
    _, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, _request())
    assert json.loads(calls[0].function.arguments) == {
        "maybe": "plain text",
        "unknown": "not json",
        "unknown2": [1, 2],
    }


def test_string_values_are_verbatim_including_tags_and_escapes():
    tricky = 'a "quote" \\ back\tslash\n<tool_call>x</tool_call> <|parameter|> 한글\n\n'
    text = _call("write_file", ("path", " spaced.py "), ("content", tricky))
    _, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, _request())
    args = json.loads(calls[0].function.arguments)
    assert args == {"path": " spaced.py ", "content": tricky}


def test_content_before_and_multiple_calls():
    text = (
        "파일을 저장할게요.\n"
        + _call("write_file", ("path", "a.py"), ("content", "x"))
        + "\n"
        + _call("ping")
    )
    content, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, _request())
    assert content == "파일을 저장할게요.\n\n"
    assert [c.function.name for c in calls] == ["write_file", "ping"]
    assert json.loads(calls[1].function.arguments) == {}


def test_unclosed_call_is_not_a_tool_call(parser):
    text = "<tool_call>write_file<|parameter|>path\na.py<|/parameter|>"
    content, calls = run_tool_extraction(parser, text, _request())
    assert calls == [] and content is None


def test_missing_parameter_end_does_not_complete_a_call(parser):
    text = "<tool_call>write_file<|parameter|>path\na.py</tool_call>"
    content, calls = run_tool_extraction(parser, text, _request())
    assert calls == [] and content is None


def test_bool_fallback_for_non_json_literal(parser):
    text = _call("typed", ("b", "True"), ("i", " 7 "))
    _, calls = run_tool_extraction(parser, text, _request())
    assert json.loads(calls[0].function.arguments) == {"b": True, "i": 7}


def test_string_argument_streams_incrementally():
    long = "line\n" * 200
    text = _call("write_file", ("path", "big.txt"), ("content", long))
    p = AXK3ToolParser(MagicMock())
    req = _request()
    prev, arg_deltas = "", 0
    for ch in text:
        cur = prev + ch
        d = p.extract_tool_calls_streaming(prev, cur, ch, [], [], [], req)
        if (
            d
            and d.tool_calls
            and d.tool_calls[0].function
            and d.tool_calls[0].function.arguments
            and "</tool_call>" not in cur
        ):
            arg_deltas += 1
        prev = cur
    assert arg_deltas > 100  # content was streamed as it was generated


def test_partial_start_tag_is_held_back():
    p = AXK3ToolParser(MagicMock())
    req = _request()
    d = p.extract_tool_calls_streaming("", "Hi <tool_", "Hi <tool_", [], [], [], req)
    assert d.content == "Hi "
    d = p.extract_tool_calls_streaming(
        "Hi <tool_", "Hi <tool_call>pi", "call>pi", [], [], [], req
    )
    assert d is None  # name not finished yet
    d = p.extract_tool_calls_streaming(
        "Hi <tool_call>pi",
        "Hi <tool_call>ping</tool_call>",
        "ng</tool_call>",
        [],
        [],
        [],
        req,
    )
    assert (
        d.tool_calls[0].function.name == "ping"
        and d.tool_calls[0].function.arguments == "{}"
    )


def test_adjust_request_keeps_special_markers():
    tok = MagicMock()
    tok.all_special_tokens = [
        "<|im_end|>",
        "<tool_call>",
        "</tool_call>",
        "<|parameter|>",
        "<|/parameter|>",
    ]
    p = AXK3ToolParser(tok)
    req = p.adjust_request(_request())
    assert req.skip_special_tokens is False
    req = p.adjust_request(_request(tool_choice="none"))
    assert req.skip_special_tokens is True


def test_adjust_request_leaves_plain_text_markers_alone():
    tok = MagicMock()
    tok.all_special_tokens = ["<|im_end|>"]
    p = AXK3ToolParser(tok)
    assert p.adjust_request(_request()).skip_special_tokens is True


def _grammar(parser, request):
    adjusted = parser.adjust_request(request)
    return Grammar.from_structural_tag(adjusted.structured_outputs.structural_tag)


def test_required_forces_a_native_call(parser):
    grammar = _grammar(parser, _request("required"))
    assert _is_grammar_accept_string(grammar, _call("ping"))
    assert _is_grammar_accept_string(grammar, _call("ping") * 2)
    assert not _is_grammar_accept_string(grammar, "No tool needed.")
    assert not _is_grammar_accept_string(grammar, "")


def test_named_choice_allows_only_the_selected_function(parser):
    req = _request({"type": "function", "function": {"name": "ping"}})
    grammar = _grammar(parser, req)
    assert _is_grammar_accept_string(grammar, _call("ping"))
    assert not _is_grammar_accept_string(grammar, _call("typed"))
    assert not _is_grammar_accept_string(grammar, _call("ping") * 2)


def test_required_respects_parallel_tool_calls_false(parser):
    req = _request("required")
    req.parallel_tool_calls = False
    grammar = _grammar(parser, req)
    assert _is_grammar_accept_string(grammar, _call("ping"))
    assert not _is_grammar_accept_string(grammar, _call("ping") * 2)


def test_none_prevents_native_tool_tags_in_content(parser):
    grammar = _grammar(parser, _request("none"))
    assert _is_grammar_accept_string(grammar, "No tool needed.")
    assert not _is_grammar_accept_string(grammar, _call("ping"))
    assert not _is_grammar_accept_string(grammar, "Answer " + _call("ping"))
    assert not _is_grammar_accept_string(grammar, "Answer </think> repeated")


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_tool_choice_enforces_parameter_keys_and_nested_json(parser, tool_choice):
    grammar = _grammar(parser, _request(tool_choice))
    params = (("path", "a.txt"), ("content", 'a "quote"\nnext line'))
    valid = _call("write_file", *params, ("options", '{"overwrite": true}'))
    assert _is_grammar_accept_string(grammar, valid)
    assert not _is_grammar_accept_string(
        grammar, _call("write_file", *params, ("overwrite", "true"))
    )
    assert not _is_grammar_accept_string(
        grammar, _call("write_file", *params, ("options", "not json"))
    )
    assert not _is_grammar_accept_string(
        grammar, _call("write_file", ("path", "a.txt"))
    )


def test_auto_allows_plain_content_and_parallel_calls(parser):
    grammar = _grammar(parser, _request())
    assert _is_grammar_accept_string(grammar, "No tool needed.")
    assert _is_grammar_accept_string(grammar, "Checking. " + _call("ping") * 2)


def test_none_preserves_an_explicit_response_format(parser):
    data = _request("none").model_dump()
    data["response_format"] = {"type": "json_object"}
    req = ChatCompletionRequest.model_validate(data)
    parser.adjust_request(req)
    assert req.response_format.type == "json_object"
    assert req.structured_outputs is None


def test_content_between_and_after_calls_is_preserved():
    text = "먼저 " + _call("ping") + " 중간 " + _call("ping") + " 끝."
    content, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, _request())
    assert content == "먼저  중간  끝."
    assert len({c.id for c in calls}) == 2


def test_auto_response_format_allows_json_or_native_tool_calls(parser):
    data = _request().model_dump()
    data["response_format"] = {"type": "json_object"}
    req = ChatCompletionRequest.model_validate(data)
    grammar = _grammar(parser, req)
    assert _is_grammar_accept_string(grammar, '{"answer": "hello"}')
    assert _is_grammar_accept_string(grammar, '\n\n{"answer": "hello"}\n')
    assert _is_grammar_accept_string(grammar, _call("ping"))
    assert not _is_grammar_accept_string(grammar, "unstructured answer")


@pytest.mark.parametrize(
    "schema, valid, invalid",
    [
        ({"type": "string", "enum": ["yes", "no"]}, "yes", "maybe"),
        ({"type": "string", "const": "007"}, "007", "7"),
        ({"type": "string", "pattern": "^[A-Z]{2}$"}, "AB", "abc"),
        ({"type": "string", "minLength": 2, "maxLength": 3}, "abc", "abcd"),
        ({"type": ["string", "null"], "enum": ["yes", None]}, "null", "maybe"),
        (
            {"anyOf": [{"type": "string", "enum": ["yes"]}, {"type": "integer"}]},
            "7",
            "maybe",
        ),
    ],
)
def test_scalar_value_constraints_use_the_native_encoding(
    parser, schema, valid, invalid
):
    data = _request("required").model_dump()
    data["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "typed",
                "parameters": {
                    "type": "object",
                    "properties": {"x": schema},
                    "required": ["x"],
                },
            },
        }
    ]
    req = ChatCompletionRequest.model_validate(data)
    grammar = _grammar(parser, req)
    assert _is_grammar_accept_string(grammar, _call("typed", ("x", valid)))
    assert not _is_grammar_accept_string(grammar, _call("typed", ("x", invalid)))


def test_local_refs_preserve_raw_strings_and_validate_nested_objects(parser):
    data = _request("required").model_dump()
    data["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "typed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "s": {"$ref": "#/$defs/Code"},
                        "obj": {"$ref": "#/$defs/Options"},
                    },
                    "required": ["s", "obj"],
                    "$defs": {
                        "Code": {"type": "string", "enum": ["007"]},
                        "Options": {
                            "type": "object",
                            "properties": {
                                "cc": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["cc"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        }
    ]
    req = ChatCompletionRequest.model_validate(data)
    text = _call("typed", ("s", "007"), ("obj", '{"cc": ["a", "b"]}'))
    grammar = _grammar(parser, req)
    assert _is_grammar_accept_string(grammar, text)
    assert not _is_grammar_accept_string(
        grammar, _call("typed", ("s", "007"), ("obj", '{"cc": true}'))
    )
    _, calls = _both(lambda: AXK3ToolParser(MagicMock()), text, req)
    assert json.loads(calls[0].function.arguments) == {
        "s": "007",
        "obj": {"cc": ["a", "b"]},
    }


def test_pattern_and_length_cannot_be_silently_ignored(parser):
    req = _request()
    req.tools[0].function.parameters["properties"]["path"] = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "maxLength": 3,
    }
    with pytest.raises(VLLMValidationError, match="pattern and length"):
        parser.adjust_request(req)
