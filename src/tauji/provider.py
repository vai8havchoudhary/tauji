from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from tauji.engine import AgentTool
from tauji.transcript import (
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


class CLIProxyProvider:
    """The only model boundary in Tauji: OpenAI-compatible Chat Completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client

    async def response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AssistantMessage:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *map(_to_openai_message, messages)],
            "stream": False,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"

        try:
            data = await self._post(payload)
            choice, raw_message = _response_message(data)

            content: list[AssistantContent] = []
            text = raw_message.get("content")
            if text:
                content.append(TextContent(text=text))
            raw_calls = raw_message.get("tool_calls") or []
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    raise ValueError("tool call must be an object")
                call_id = raw_call.get("id")
                fn = raw_call.get("function")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("tool call id must be a non-empty string")
                if not isinstance(fn, dict):
                    raise ValueError("tool call function must be an object")
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("tool call function name must be a non-empty string")
                arguments = fn.get("arguments") or "{}"
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool call arguments must decode to an object")
                content.append(ToolCall(id=call_id, name=name, arguments=arguments))

            finish_reason = choice.get("finish_reason")
            stop_reason: Literal["toolUse", "length", "stop"] = (
                "toolUse"
                if any(isinstance(block, ToolCall) for block in content)
                else "length"
                if finish_reason == "length"
                else "stop"
            )
            return AssistantMessage(
                content=content,
                stop_reason=stop_reason,
            )
        except Exception as exc:
            return _error_message(str(exc))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self._client is not None:
            response = await self._client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("CLIProxyAPI returned a non-object response")
        return data


def _response_message(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("CLIProxyAPI response must contain at least one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("CLIProxyAPI choice must be an object")
    raw_message = choice.get("message")
    if not isinstance(raw_message, dict):
        raise ValueError("CLIProxyAPI choice must contain a message object")

    text = raw_message.get("content")
    if text is not None and not isinstance(text, str):
        raise ValueError("CLIProxyAPI message content must be a string or null")
    raw_calls = raw_message.get("tool_calls")
    if raw_calls is not None and not isinstance(raw_calls, list):
        raise ValueError("CLIProxyAPI message tool_calls must be a list or null")
    if not text and not raw_calls:
        raise ValueError("CLIProxyAPI message contains neither content nor tool calls")

    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length", "tool_calls", "function_call", "content_filter"}:
        raise ValueError(f"unsupported CLIProxyAPI finish_reason: {finish_reason!r}")
    return choice, raw_message


def _error_message(message: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        stop_reason="error",
        error_message=message,
    )


def _to_openai_message(message: AgentMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        out: dict[str, Any] = {"role": "assistant", "content": message.text or None}
        if message.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return out
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.text,
        }
    raise TypeError(f"unsupported message: {type(message).__name__}")
