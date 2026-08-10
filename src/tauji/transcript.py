"""Durable transcript types used by Tauji and CLIProxyAPI."""

from __future__ import annotations

from time import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _timestamp_ms() -> int:
    return int(time() * 1000)


class TranscriptModel(BaseModel):
    """Accept older Tauji fields while writing only the current schema."""

    model_config = ConfigDict(
        extra="ignore",
        validate_by_name=True,
        validate_by_alias=True,
        alias_generator=_to_camel,
    )


class TextContent(TranscriptModel):
    type: Literal["text"] = "text"
    text: str


class ToolCall(TranscriptModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)


type AssistantContent = Annotated[TextContent | ToolCall, Field(discriminator="type")]


class UserMessage(TranscriptModel):
    role: Literal["user"] = "user"
    content: str
    timestamp: int = Field(default_factory=_timestamp_ms)

    @property
    def text(self) -> str:
        return self.content


StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


class AssistantMessage(TranscriptModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=_timestamp_ms)

    @model_validator(mode="before")
    @classmethod
    def _normalize_content(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [{"type": "text", "text": content}] if content else []
        return data

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.content if isinstance(block, ToolCall))


class ToolResultMessage(TranscriptModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextContent] = Field(default_factory=list)
    is_error: bool = False
    timestamp: int = Field(default_factory=_timestamp_ms)

    @model_validator(mode="before")
    @classmethod
    def _normalize_content(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [{"type": "text", "text": content}] if content else []
        return data

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content)


type AgentMessage = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]
