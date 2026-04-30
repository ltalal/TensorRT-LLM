# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)


def _get_param_type(param_name: str, func_name: str, tools: List[Tool]) -> Optional[str]:
    """Get the expected parameter type from the tool JSON schema."""
    for tool in tools:
        if tool.function.name != func_name:
            continue

        props = (tool.function.parameters or {}).get("properties", {})
        if param_name not in props:
            return None

        type_val = props[param_name].get("type")
        if isinstance(type_val, str):
            return type_val
        if isinstance(type_val, list):
            non_null_types = [t for t in type_val if t != "null"]
            return non_null_types[0] if non_null_types else "string"
    return None


def _strip_template_trailing_newline(value: str) -> str:
    """Remove the separator newline emitted by the chat template."""
    value = value.rstrip(" \t")
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def _parse_param_value(value: str, param_type: Optional[str]) -> Any:
    """Parse one argument value, preserving declared strings verbatim."""
    value = _strip_template_trailing_newline(value)

    if param_type == "string":
        return value

    stripped_value = value.strip()
    if stripped_value == "":
        return value

    try:
        return json.loads(stripped_value)
    except (json.JSONDecodeError, ValueError):
        pass

    if param_type in ("number", "integer"):
        try:
            if "." in stripped_value or "e" in stripped_value.lower():
                return float(stripped_value)
            return int(stripped_value)
        except (ValueError, TypeError):
            pass

    if param_type == "boolean":
        if stripped_value.lower() == "true":
            return True
        if stripped_value.lower() == "false":
            return False

    return value


class MultilineToolParser(BaseToolParser):
    r"""Tool parser for multiline separator-based tool calls.

    Expected format:
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
        tool_name
        <｜tool▁sep｜>arg_name
        multiline argument value
        <｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """

    def __init__(self):
        super().__init__()
        self.calls_begin = "<｜tool▁calls▁begin｜>"  # nosec B105
        self.calls_end = "<｜tool▁calls▁end｜>"  # nosec B105
        self.bot_token = "<｜tool▁call▁begin｜>"  # nosec B105
        self.eot_token = "<｜tool▁call▁end｜>"  # nosec B105
        self.tool_separator = "<｜tool▁sep｜>"
        self._emitted_call_count = 0
        self._prefix_sent = False

    def has_tool_call(self, text: str) -> bool:
        """Check whether the text contains the multiline tool call markers."""
        return self.calls_begin in text or self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: detect and parse all complete tool call blocks."""
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        pos = 0

        while True:
            block_start = text.find(self.calls_begin, pos)
            if block_start == -1:
                normal_parts.append(text[pos:])
                break

            normal_parts.append(text[pos:block_start])
            block_content_start = block_start + len(self.calls_begin)
            block_end = text.find(self.calls_end, block_content_start)
            if block_end == -1:
                logger.warning("Malformed multiline tool call block: missing end marker")
                normal_parts.append(text[block_start:])
                break

            block = text[block_content_start:block_end]
            parsed_calls = self._parse_tool_calls(block, len(calls), tools)
            calls.extend(parsed_calls)
            pos = block_end + len(self.calls_end)

        normal_text = " ".join(part.strip() for part in normal_parts if part.strip())
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming parser for complete tool calls as they arrive."""
        self._buffer += new_text
        current_text = self._buffer

        if not self.has_tool_call(current_text):
            if self._ends_with_any_partial_token(current_text):
                return StreamingParseResult()

            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        calls: List[ToolCallItem] = []
        try:
            first_call_start = current_text.find(self.bot_token)
            prefix_text = ""
            if first_call_start > 0 and self._emitted_call_count == 0 and not self._prefix_sent:
                prefix_text = current_text[:first_call_start].replace(self.calls_begin, "").strip()
                self._prefix_sent = True

            complete_calls, last_end = self._extract_complete_call_bodies(current_text)
            for call_idx, body in enumerate(complete_calls):
                if call_idx < self._emitted_call_count:
                    continue

                func_name, arguments = self._parse_call_body(body, tools)
                calls.append(
                    ToolCallItem(
                        tool_index=call_idx,
                        name=func_name,
                        parameters="",
                    )
                )
                calls.append(
                    ToolCallItem(
                        tool_index=call_idx,
                        name=None,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )

            self._emitted_call_count = len(complete_calls)

            if self.calls_end in current_text and last_end != -1:
                calls_end_idx = current_text.find(self.calls_end, last_end)
                if calls_end_idx != -1:
                    self._buffer = current_text[calls_end_idx + len(self.calls_end) :]
                    self._emitted_call_count = 0
                    self._prefix_sent = False

            return StreamingParseResult(normal_text=prefix_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in MultilineToolParser parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def supports_structural_tag(self) -> bool:
        """Return whether this parser supports structural tag guided decoding."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Return structure info for guided decoding (not supported)."""
        raise NotImplementedError()

    def _extract_complete_call_bodies(self, text: str) -> Tuple[List[str], int]:
        """Extract complete call bodies from the current text."""
        bodies: List[str] = []
        pos = 0
        last_end = -1

        while True:
            call_start = text.find(self.bot_token, pos)
            if call_start == -1:
                break

            body_start = call_start + len(self.bot_token)
            call_end = text.find(self.eot_token, body_start)
            if call_end == -1:
                break

            bodies.append(text[body_start:call_end])
            last_end = call_end + len(self.eot_token)
            pos = last_end

        return bodies, last_end

    def _parse_tool_calls(
        self, block: str, tool_index_start: int, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """Parse all complete tool calls from one outer block."""
        calls: List[ToolCallItem] = []
        bodies, _ = self._extract_complete_call_bodies(block)
        for offset, body in enumerate(bodies):
            func_name, arguments = self._parse_call_body(body, tools)
            calls.append(
                ToolCallItem(
                    tool_index=tool_index_start + offset,
                    name=func_name,
                    parameters=json.dumps(arguments, ensure_ascii=False),
                )
            )
        return calls

    def _parse_call_body(self, body: str, tools: List[Tool]) -> Tuple[str, Dict[str, Any]]:
        """Parse a single call body into a function name and arguments."""
        body = body.lstrip()
        first_sep = body.find(self.tool_separator)
        if first_sep == -1:
            return body.strip(), {}

        func_name = body[:first_sep].strip()
        args_text = body[first_sep:]
        arguments: Dict[str, Any] = {}
        pos = 0

        while pos < len(args_text):
            sep_idx = args_text.find(self.tool_separator, pos)
            if sep_idx == -1:
                break

            key_start = sep_idx + len(self.tool_separator)
            key_end = args_text.find("\n", key_start)
            if key_end == -1:
                break

            param_name = args_text[key_start:key_end].strip()
            value_start = key_end + 1
            next_sep = args_text.find(self.tool_separator, value_start)

            if next_sep == -1:
                param_value = args_text[value_start:]
                pos = len(args_text)
            else:
                param_value = args_text[value_start:next_sep]
                pos = next_sep

            if not param_name:
                continue

            param_type = _get_param_type(param_name, func_name, tools)
            arguments[param_name] = _parse_param_value(param_value, param_type)

        return func_name, arguments

    def _ends_with_any_partial_token(self, text: str) -> bool:
        """Check if text may be the beginning of a tool marker."""
        return any(
            self._ends_with_partial_token(text, token) > 0
            for token in (self.calls_begin, self.bot_token)
        )
