# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parser for A.X-K3.

Format emitted by the model (one ``<tool_call>`` block per call, back to back)::

    <tool_call>NAME<|parameter|>KEY
    VALUE<|/parameter|><|parameter|>KEY2
    VALUE2<|/parameter|></tool_call>

* The key runs up to the first newline after ``<|parameter|>``; the value is
  everything after that newline up to ``<|/parameter|>``, taken verbatim.
* String values are written raw (no quotes, no escaping). Every other value
  (number, boolean, null, array, object) is written as JSON.

So the value of a parameter is decoded with the tool's JSON schema: a
parameter typed ``string`` keeps the raw text exactly; anything else is parsed
as JSON. Parameters the schema does not describe are parsed as JSON when that
yields a non-string, and kept as raw text otherwise.

Arguments are always re-serialised as ``json.dumps(args, ensure_ascii=False)``
-- in streaming, the concatenation of the argument deltas is exactly that
string, so the serving layer's end-of-stream flush
(``get_remaining_unstreamed_args``) stays consistent. String-typed parameters
are streamed character by character as they are generated (useful for long
``content`` arguments); other parameters are emitted once their value is
complete.

``<think>...</think>`` is not handled here: pair this with a reasoning parser.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
    partial_tag_overlap,
)

logger = init_logger(__name__)

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
PARAM_START = "<|parameter|>"
PARAM_END = "<|/parameter|>"
_MARKERS = (TOOL_CALL_START, TOOL_CALL_END, PARAM_START, PARAM_END)


# --------------------------------------------------------------------------
# Parsing (pure functions; shared by the streaming and non-streaming paths)
# --------------------------------------------------------------------------
@dataclass
class _Param:
    key: str | None  # None while the key line is still being generated
    raw: str = ""  # verbatim value text (streaming: held-back tag overlap removed)
    closed: bool = False  # True once <|/parameter|> has been seen


@dataclass
class _Call:
    name: str | None = None  # None until the name is terminated by a tag
    params: list[_Param] = field(default_factory=list)
    closed: bool = False  # True once </tool_call> has been seen


def _parse_call(text: str, start: int, final: bool) -> tuple[_Call, int]:
    """Parse one call whose body starts at ``text[start]`` (just after
    ``<tool_call>``). Returns the call and the index just past it."""
    call = _Call()
    n = len(text)

    ends = [
        i
        for i in (text.find(PARAM_START, start), text.find(TOOL_CALL_END, start))
        if i != -1
    ]
    if not ends:
        return call, n  # name still being generated
    name_end = min(ends)
    call.name = text[start:name_end].strip() or None
    pos = name_end

    while True:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            return call, n

        if text.startswith(TOOL_CALL_END, pos):
            call.closed = True
            return call, pos + len(TOOL_CALL_END)

        if text.startswith(PARAM_START, pos):
            key_start = pos + len(PARAM_START)
            newline = text.find("\n", key_start)
            if newline == -1:
                call.params.append(_Param(key=None))
                return call, n
            key = text[key_start:newline].strip()
            value_start = newline + 1
            value_end = text.find(PARAM_END, value_start)
            if value_end != -1:
                call.params.append(_Param(key, text[value_start:value_end], True))
                pos = value_end + len(PARAM_END)
                continue
            if final:
                # Tolerate a missing <|/parameter|>: the value then runs to the
                # next </tool_call>.
                call_end = text.find(TOOL_CALL_END, value_start)
                if call_end != -1:
                    call.params.append(_Param(key, text[value_start:call_end], True))
                    call.closed = True
                    return call, call_end + len(TOOL_CALL_END)
                call.params.append(_Param(key, text[value_start:], False))
                return call, n
            raw = text[value_start:]
            overlap = partial_tag_overlap(raw, PARAM_END)
            call.params.append(_Param(key, raw[: len(raw) - overlap], False))
            return call, n

        tail = text[pos:]
        if not final and (
            TOOL_CALL_END.startswith(tail) or PARAM_START.startswith(tail)
        ):
            return call, n  # a tag is being generated
        # Stray text between tags: skip to the next tag.
        nxt = [
            i
            for i in (text.find(PARAM_START, pos), text.find(TOOL_CALL_END, pos))
            if i != -1
        ]
        if not nxt:
            return call, n
        pos = min(nxt)


def _scan(text: str, final: bool) -> tuple[str, list[_Call]]:
    """Split text into (content before the first call, calls). Parsing stops at
    the first call that is not closed."""
    first = text.find(TOOL_CALL_START)
    if first == -1:
        return text, []
    calls: list[_Call] = []
    pos = first
    while pos != -1:
        call, end = _parse_call(text, pos + len(TOOL_CALL_START), final)
        calls.append(call)
        if not call.closed:
            break
        pos = text.find(TOOL_CALL_START, end)
    return text[:first], calls


# --------------------------------------------------------------------------
# Value decoding and argument rendering
# --------------------------------------------------------------------------
def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _is_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite(v) for v in value)
    if isinstance(value, dict):
        return all(_is_finite(v) for v in value.values())
    return True


def _schema_types(schema: Any) -> set[str] | None:
    """JSON types a property schema allows, or None when it does not say."""
    if not isinstance(schema, dict) or not any(
        k in schema for k in ("type", "enum", "anyOf", "oneOf", "allOf")
    ):
        return None
    return set(extract_types_from_schema(schema))


def _json_type_ok(value: Any, types: set[str]) -> bool:
    if value is None:
        return "null" in types
    if isinstance(value, bool):
        return "boolean" in types
    if isinstance(value, int):
        return bool(types & {"integer", "number"})
    if isinstance(value, float):
        return "number" in types or ("integer" in types and value.is_integer())
    if isinstance(value, str):
        return "string" in types
    if isinstance(value, list):
        return "array" in types
    if isinstance(value, dict):
        return "object" in types
    return False


def _decode_value(raw: str, schema: Any, known: bool) -> Any:
    types = _schema_types(schema) if known else None
    if types == {"string"}:
        return raw  # strings are written verbatim
    text = raw.strip()
    try:
        parsed = json.loads(text)
        parsed_ok = _is_finite(parsed)
    except ValueError:
        parsed, parsed_ok = None, False

    if types is None:  # not described by the schema
        return parsed if parsed_ok and not isinstance(parsed, str) else raw
    if parsed_ok and not isinstance(parsed, str) and _json_type_ok(parsed, types):
        return parsed
    if "string" in types:
        return raw
    return coerce_to_schema_type(text, sorted(types))  # best effort, e.g. True


def _is_streamable_string(schema: Any, known: bool) -> bool:
    return known and _schema_types(schema) == {"string"}


def _render_arguments(call: _Call, properties: dict[str, Any]) -> str:
    """JSON text of the arguments parsed so far.

    For a complete call this equals ``json.dumps(args, ensure_ascii=False)``.
    For a partial call it is always a prefix of that final string: closed
    parameters are rendered whole, an open string-typed parameter is rendered
    up to its last generated character, and anything else waits.
    """
    pieces: list[str] = []
    seen: set[str] = set()
    complete = True
    for param in call.params:
        if param.key is None:
            complete = False
            break
        if param.key in seen:  # duplicate key: the first occurrence wins
            if param.closed:
                continue
            complete = False
            break
        known = param.key in properties
        schema = properties.get(param.key)
        if param.closed:
            seen.add(param.key)
            value = _decode_value(param.raw, schema, known)
            pieces.append(f"{_dumps(param.key)}: {_dumps(value)}")
            continue
        complete = False
        if _is_streamable_string(schema, known):
            # json.dumps escapes a str character by character, so the escaped
            # partial value is a prefix of the escaped final value.
            pieces.append(f'{_dumps(param.key)}: "{_dumps(param.raw)[1:-1]}')
        break
    text = "{" + ", ".join(pieces)
    if complete and call.closed:
        text += "}"
    return text


# --------------------------------------------------------------------------
# vLLM tool parser
# --------------------------------------------------------------------------
class AXK3ToolParser(ToolParser):
    # The output is not JSON, so tool_choice="required"/named cannot use the
    # JSON-schema path; the serving layer parses those like "auto" instead.
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._sent_content_idx = 0
        self._tool_ids: list[str] = []
        self._markers_are_special = self._detect_special_markers()

    def _detect_special_markers(self) -> bool:
        """True when the markers would be dropped by skip_special_tokens."""
        tok = self.model_tokenizer
        try:
            special = set(getattr(tok, "all_special_tokens", None) or [])
            added = getattr(tok, "added_tokens_decoder", None) or {}
            for token in added.values():
                if getattr(token, "special", False):
                    special.add(str(getattr(token, "content", token)))
            if any(m in special for m in _MARKERS):
                return True
            if special:
                return False
            # Could not inspect special tokens: be safe if the markers are
            # single tokens in the vocabulary.
            return any(m in self.vocab for m in _MARKERS)
        except Exception:
            return True

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # super().adjust_request() is deliberately not called: for
        # tool_choice="required"/named it installs a JSON schema as a
        # structured-output constraint, which would force output this format
        # cannot be.
        if not request.tools or request.tool_choice == "none":
            return request
        if self._markers_are_special and hasattr(request, "skip_special_tokens"):
            request.skip_special_tokens = False
        return request

    def _properties(self, name: str, request: Any) -> dict[str, Any]:
        tools = getattr(request, "tools", None) or self.tools
        return find_tool_properties(tools, name) or {}

    # ---- non-streaming ------------------------------------------------------
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        if TOOL_CALL_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        try:
            content, calls = _scan(model_output, final=True)
            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=call.name,
                        arguments=_render_arguments(
                            call, self._properties(call.name, request)
                        ),
                    ),
                )
                for call in calls
                if call.closed and call.name
            ]
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content.strip() else None,
            )
        except Exception:
            logger.exception("Error extracting A.X-K3 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ---- streaming ----------------------------------------------------------
    def _content_delta(self, current_text: str) -> str | None:
        idx = current_text.find(TOOL_CALL_START)
        if idx == -1:
            idx = len(current_text) - partial_tag_overlap(current_text, TOOL_CALL_START)
        if idx > self._sent_content_idx:
            delta = current_text[self._sent_content_idx : idx]
            self._sent_content_idx = idx
            return delta
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        try:
            content = self._content_delta(current_text)
            _, calls = _scan(current_text, final=False)
            deltas: list[DeltaToolCall] = []

            for i, call in enumerate(calls):
                is_new = i >= len(self.prev_tool_call_arr)
                if is_new:
                    if call.name is None:
                        break  # the name is still being generated
                    self.prev_tool_call_arr.append({"name": call.name, "arguments": ""})
                    self.streamed_args_for_tool.append("")
                    self._tool_ids.append(make_tool_call_id())
                    self.current_tool_id = i
                    self.current_tool_name_sent = True

                args = _render_arguments(
                    call, self._properties(call.name or "", request)
                )
                sent = self.streamed_args_for_tool[i]
                diff = ""
                if args.startswith(sent):
                    diff = args[len(sent) :]
                else:  # cannot happen by construction; never corrupt the stream
                    logger.warning("A.X-K3 tool call %d: arguments diverged", i)
                if diff:
                    self.streamed_args_for_tool[i] = args
                    self.prev_tool_call_arr[i]["arguments"] = args

                if is_new:
                    deltas.append(
                        DeltaToolCall(
                            index=i,
                            id=self._tool_ids[i],
                            type="function",
                            function=DeltaFunctionCall(
                                name=call.name, arguments=diff or None
                            ).model_dump(exclude_none=True),
                        )
                    )
                elif diff:
                    deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            if content or deltas:
                return DeltaMessage(content=content, tool_calls=deltas)
            return None
        except Exception:
            logger.exception("Error streaming A.X-K3 tool calls.")
            return None
