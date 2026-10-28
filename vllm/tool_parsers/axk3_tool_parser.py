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
* The reserved closing marker ``<|/parameter|>`` cannot occur in a raw value.

So the value of a parameter is decoded with the tool's JSON schema: a
parameter typed ``string`` keeps the raw text exactly; anything else is parsed
as JSON. Parameters the schema does not describe are parsed as JSON when that
yields a non-string, and kept as raw text otherwise.

Complete arguments are re-serialised as ``json.dumps(args, ensure_ascii=False)``
-- in streaming, the concatenation of the argument deltas is exactly that
string, so the serving layer's end-of-stream flush
(``get_remaining_unstreamed_args``) stays consistent. String-typed parameters
are streamed character by character as they are generated (useful for long
``content`` arguments); other parameters are emitted once their value is
complete.
Content before, between, and after calls is preserved. Truncated calls are
omitted from non-streaming tool calls; their already streamed arguments stay
incomplete rather than being repaired into executable JSON.

Generation uses the native tag grammar for auto/required/named calls, with
parameters in schema declaration order and JSON constraints on non-string
values. Raw strings support enum, const, pattern, and length constraints;
combining pattern and length is rejected unless a finite enum/const resolves
both. For tool_choice="none", tool markers are excluded from content.
An explicit response format allows either that format or a native call under
auto, and is preserved as-is under none.

``<think>...</think>`` is not handled here: pair this with a reasoning parser.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    get_function_tools,
    partial_tag_overlap,
)

logger = init_logger(__name__)

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
PARAM_START = "<|parameter|>"
PARAM_END = "<|/parameter|>"
_MARKERS = (TOOL_CALL_START, TOOL_CALL_END, PARAM_START, PARAM_END)
_REASONING_MARKERS = ("<think>", "</think>")


def _resolve_property(schema: Any, root: dict[str, Any]) -> Any:
    seen: set[str] = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            raise VLLMValidationError(
                "AXK3 requires resolvable local schema references"
            )
        seen.add(ref)
        target: Any = root
        try:
            for key in ref[2:].split("/"):
                target = target[key.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError) as exc:
            raise VLLMValidationError(
                f"Unresolved AXK3 schema reference: {ref}"
            ) from exc
        schema = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
    if isinstance(schema, dict):
        schema = dict(schema)
        for key in ("$defs", "definitions"):
            if key in root:
                schema.setdefault(key, root[key])
    return schema


def _value_format(schema: Any) -> Any:
    if isinstance(schema, dict) and "$ref" in schema:
        schema = _resolve_property(schema, schema)
    types = _schema_types(schema)
    if isinstance(schema, dict) and ("enum" in schema or "const" in schema):
        values = schema.get("enum", [schema.get("const")])
        validator = Draft202012Validator(schema)
        encodings = [
            ConstStringFormat(value=v if isinstance(v, str) else _dumps(v))
            for v in values
            if validator.is_valid(v) and not (isinstance(v, str) and PARAM_END in v)
        ]
        if not encodings:
            raise VLLMValidationError("AXK3 parameter has no representable enum values")
        return encodings[0] if len(encodings) == 1 else OrFormat(elements=encodings)
    if types is None:
        return AnyTextFormat(excludes=[PARAM_END])
    if "string" in types and len(types) > 1:
        for union in ("anyOf", "oneOf"):
            if union in schema:
                base = {k: v for k, v in schema.items() if k != union}
                return OrFormat(
                    elements=[_value_format({**base, **s}) for s in schema[union]]
                )
        branches = []
        for kind in sorted(types):
            branch = {**schema, "type": kind}
            if "enum" in branch:
                branch["enum"] = [v for v in branch["enum"] if _json_type_ok(v, {kind})]
            branches.append(_value_format(branch))
        return OrFormat(elements=branches)
    if types == {"string"} and any(key in schema for key in ("minLength", "maxLength")):
        if "pattern" in schema:
            raise VLLMValidationError(
                "AXK3 raw strings cannot combine pattern and length constraints"
            )
        # xgrammar ignores root string lengths when excludes is set.
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength", "")
        schema = {
            **{k: v for k, v in schema.items() if k not in ("minLength", "maxLength")},
            "pattern": rf"^[\s\S]{{{minimum},{maximum}}}$",
        }
    # qwen_xml renders a scalar string verbatim, matching K3. Object and
    # array values must retain the ordinary JSON encoding.
    return JSONSchemaFormat(
        json_schema=schema,
        style="qwen_xml" if types == {"string"} else "json",
        excludes=[PARAM_END],
    )


def _tool_tag(name: str, parameters: dict[str, Any]) -> TagFormat:
    # Fix the parameter order to enforce required keys without allowing
    # duplicates. The extraction paths still accept any order.
    required = parameters.get("required", [])
    elements: list[Any] = []
    for key, schema in parameters.get("properties", {}).items():
        schema = _resolve_property(schema, parameters)
        value = _value_format(schema)
        param = TagFormat(begin=PARAM_START + key + "\n", content=value, end=PARAM_END)
        elements.append(param if key in required else OptionalFormat(content=param))
    body = (
        SequenceFormat(elements=elements) if elements else ConstStringFormat(value="")
    )
    return TagFormat(begin=TOOL_CALL_START + name, content=body, end=TOOL_CALL_END)


def _tool_choice_tag(
    request: ChatCompletionRequest | ResponsesRequest,
) -> StructuralTag:
    if request.tool_choice == "none":
        return StructuralTag(
            format=AnyTextFormat(excludes=list(_MARKERS + _REASONING_MARKERS))
        )
    choice = request.tool_choice
    function = getattr(choice, "function", None)
    name = getattr(function, "name", None) or getattr(choice, "name", None)
    tags = []
    for tool in get_function_tools(request.tools or []):
        tool_function = (
            tool.function if isinstance(tool, ChatCompletionToolsParam) else tool
        )
        if name is None or tool_function.name == name:
            tags.append(_tool_tag(tool_function.name, tool_function.parameters or {}))
    if not tags:
        raise ValueError("AXK3 tool_choice requires an available function tool")
    if choice in (None, "auto"):
        return StructuralTag(
            format=TriggeredTagsFormat(
                triggers=[TOOL_CALL_START],
                tags=tags,
                stop_after_first=request.parallel_tool_calls is False,
                excludes=list(_REASONING_MARKERS),
            )
        )
    return StructuralTag(
        format=TagsWithSeparatorFormat(
            tags=tags,
            separator="",
            at_least_one=True,
            stop_after_first=name is not None or request.parallel_tool_calls is False,
        )
    )


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
    """Collect content outside calls, stopping at the first incomplete call."""
    content: list[str] = []
    calls: list[_Call] = []
    end = 0
    while True:
        pos = text.find(TOOL_CALL_START, end)
        if pos == -1:
            tail = text[end:]
            overlap = 0 if final else partial_tag_overlap(tail, TOOL_CALL_START)
            content.append(tail[: len(tail) - overlap])
            break
        content.append(text[end:pos])
        call, end = _parse_call(text, pos + len(TOOL_CALL_START), final)
        calls.append(call)
        if not call.closed:
            break
    return "".join(content), calls


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
        k in schema for k in ("type", "enum", "const", "anyOf", "oneOf", "allOf")
    ):
        return None
    if "const" in schema:
        schema = {**schema, "enum": [schema["const"]]}
    if "type" in schema:
        kind = schema["type"]
        return {kind} if isinstance(kind, str) else set(kind)
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
    if (
        parsed_ok
        and not isinstance(parsed, str)
        and _json_type_ok(parsed, types)
        and ("string" not in types or Draft202012Validator(schema).is_valid(parsed))
    ):
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
        self._current_text = ""
        self._property_cache: dict[str, dict[str, Any]] = {}

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
        if not request.tools:
            return request
        outputs = request.extract_structured_outputs()
        if request.tool_choice in (None, "auto") and outputs is not None:
            from vllm.parser.abstract_parser import structured_outputs_to_format

            output_format = structured_outputs_to_format(outputs)
            if output_format is not None:
                forced = request.model_copy(update={"tool_choice": "required"})
                combined: Any = OrFormat(
                    elements=[output_format, _tool_choice_tag(forced).format]
                )
                if isinstance(output_format, JSONSchemaFormat):
                    combined = SequenceFormat(
                        elements=[
                            RegexFormat(pattern=r"[ \t\r\n]*"),
                            combined,
                            RegexFormat(pattern=r"[ \t\r\n]*"),
                        ]
                    )
                request.structured_outputs = StructuredOutputsParams(
                    structural_tag=StructuralTag(format=combined).model_dump_json()
                )
                if isinstance(request, ResponsesRequest):
                    request.text = None
                else:
                    request.response_format = None
        elif request.tool_choice != "none" or outputs is None:
            # Constrain native K3 tags; the generic JSON tool-call grammar
            # cannot be parsed by this parser.
            request.structured_outputs = StructuredOutputsParams(
                structural_tag=_tool_choice_tag(request).model_dump_json()
            )
            if isinstance(request, ResponsesRequest):
                request.text = None
            else:
                request.response_format = None
        if request.tool_choice == "none":
            return request
        if self._markers_are_special and hasattr(request, "skip_special_tokens"):
            request.skip_special_tokens = False
        return request

    def _properties(self, name: str, request: Any) -> dict[str, Any]:
        if name not in self._property_cache:
            tools = getattr(request, "tools", None) or self.tools
            for tool in get_function_tools(tools or []):
                function = (
                    tool.function
                    if isinstance(tool, ChatCompletionToolsParam)
                    else tool
                )
                if function.name == name:
                    root = function.parameters or {}
                    self._property_cache[name] = {
                        key: _resolve_property(schema, root)
                        for key, schema in root.get("properties", {}).items()
                    }
                    break
            else:
                self._property_cache[name] = {}
        return self._property_cache[name]

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
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content if content.strip() else None,
            )
        except Exception:
            logger.exception("Error extracting A.X-K3 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ---- streaming ----------------------------------------------------------
    def _content_delta(self, content: str) -> str | None:
        if len(content) > self._sent_content_idx:
            delta = content[self._sent_content_idx :]
            self._sent_content_idx = len(content)
            return delta
        return None

    def finish_streaming(self) -> DeltaMessage | None:
        content, _ = _scan(self._current_text, final=True)
        delta = self._content_delta(content)
        return DeltaMessage(content=delta) if delta else None

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
            self._current_text = current_text
            visible_content, calls = _scan(current_text, final=False)
            content = self._content_delta(visible_content)
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
