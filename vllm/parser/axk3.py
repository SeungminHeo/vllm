# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    GrammarFormat,
    JSONSchemaFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    TagFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.generate.base.protocol import DeltaMessage, FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import (
    DelegatingParser,
    StreamState,
)
from vllm.reasoning.axk3_reasoning_parser import AXK3ReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.axk3_tool_parser import AXK3ToolParser


def structured_outputs_to_format(params: StructuredOutputsParams) -> Format | None:
    """Map StructuredOutputsParams in a XGrammar Format."""
    if params.json_object:
        return JSONSchemaFormat(json_schema={"type": "object"})
    if params.json is not None:
        schema = params.json
        if isinstance(schema, str):
            schema = json.loads(schema)
        return JSONSchemaFormat(json_schema=schema)
    if params.regex is not None:
        return RegexFormat(pattern=params.regex)
    if params.choice is not None:
        return OrFormat(
            elements=[ConstStringFormat(value=choice) for choice in params.choice]
        )
    if params.grammar is not None:
        from vllm.v1.structured_output.utils import grammar_is_likely_lark

        grammar = params.grammar
        # IMPORTANT(arpera):
        # GrammarFormat only accepts EBNF.
        if grammar_is_likely_lark(grammar):
            try:
                grammar = str(Grammar.from_lark(grammar))
            except Exception as e:
                raise VLLMValidationError("Invalid grammar specification.") from e
        return GrammarFormat(grammar=grammar)
    if params.structural_tag is not None:
        s_tag = json.loads(params.structural_tag)
        if "structures" in s_tag:
            # LegacyStructuralTagResponseFormat
            return TriggeredTagsFormat(
                triggers=s_tag["triggers"],
                tags=[
                    TagFormat(
                        begin=structure["begin"],
                        content=JSONSchemaFormat(json_schema=structure["schema"]),
                        end=structure["end"],
                    )
                    for structure in s_tag["structures"]
                ],
            )
        # StructuralTagResponseFormat
        return StructuralTag.model_validate(s_tag).format
    return None


class AXK3Parser(DelegatingParser):
    """Compose the existing K3 parsers with consistent end-of-stream handling."""

    always_adjust_request = True

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        outputs = request.extract_structured_outputs()
        if (
            not request.include_reasoning
            and isinstance(self.reasoning_parser, AXK3ReasoningParser)
            and not getattr(request, "_grammar_from_parser", False)
            and outputs is not None
        ):
            output_format = structured_outputs_to_format(outputs)
            if output_format is not None:
                # Hidden reasoning activates the engine grammar immediately,
                # but K3's prompt still opens a thinking turn.
                request.structured_outputs = StructuredOutputsParams(
                    structural_tag=StructuralTag(
                        format=SequenceFormat(
                            elements=[
                                TagFormat(
                                    begin="",
                                    content=AnyTextFormat(excludes=["</think>"]),
                                    end="</think>",
                                ),
                                output_format,
                            ]
                        )
                    ).model_dump_json()
                )
                if isinstance(request, ResponsesRequest):
                    request.text = None
                else:
                    request.response_format = None
        return request

    def _extract_tool_calls(
        self,
        content: str | None,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        if (
            not isinstance(self.tool_parser, AXK3ToolParser)
            or not enable_auto_tools
            or request.tool_choice == "none"
        ):
            return super()._extract_tool_calls(content, request, enable_auto_tools)
        result = self.extract_tool_calls(content or "", request)
        return [
            FunctionCall(
                id=call.id, name=call.function.name, arguments=call.function.arguments
            )
            for call in result.tool_calls
        ], result.content

    def finalize_generation(
        self,
        delta_message: DeltaMessage | None,
        request: ChatCompletionRequest | ResponsesRequest,
        state: StreamState,
    ) -> DeltaMessage | None:
        delta_message = super().finalize_generation(delta_message, request, state)
        if (
            state.reasoning_ended
            and request.tool_choice != "none"
            and isinstance(self.tool_parser, AXK3ToolParser)
        ):
            flush = self.tool_parser.finish_streaming()
            if flush:
                if delta_message is None:
                    return flush
                delta_message.content = (delta_message.content or "") + (
                    flush.content or ""
                )
        return delta_message
