from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tau_agent.provider import CancellationToken
from tau_agent.provider_events import AssistantDoneEvent, AssistantErrorEvent, AssistantMessageEvent
from tau_agent.tools import AgentTool


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

    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        del session_id
        if signal and signal.is_cancelled():
            yield AssistantErrorEvent(
                reason="aborted",
                error=_error_message(model, "aborted", "request cancelled"),
            )
            return

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
            choice = data["choices"][0]
            raw_message = choice["message"]

            content = []
            text = raw_message.get("content") or ""
            if text:
                content.append(TextContent(text=text))
            for raw_call in raw_message.get("tool_calls") or ():
                fn = raw_call["function"]
                arguments = fn.get("arguments") or "{}"
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool call arguments must decode to an object")
                content.append(
                    ToolCall(id=raw_call["id"], name=fn["name"], arguments=arguments)
                )

            finish_reason = choice.get("finish_reason")
            stop_reason = (
                "toolUse"
                if any(isinstance(block, ToolCall) for block in content)
                else "length"
                if finish_reason == "length"
                else "stop"
            )
            usage = data.get("usage") or {}
            message = AssistantMessage(
                content=content,
                api="openai-chat",
                provider="cliproxy",
                model=model,
                response_id=data.get("id"),
                usage=Usage(
                    input=int(usage.get("prompt_tokens") or 0),
                    output=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                ),
                stop_reason=stop_reason,
            )
            yield AssistantDoneEvent(reason=stop_reason, message=message)
        except Exception as exc:
            yield AssistantErrorEvent(
                reason="error",
                error=_error_message(model, "error", str(exc)),
            )

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


def _error_message(model: str, reason: str, message: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-chat",
        provider="cliproxy",
        model=model,
        stop_reason=reason,
        error_message=message,
    )


def _to_openai_message(message: AgentMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": _user_content(message.content)}
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
    if isinstance(message, BashExecutionMessage):
        return {"role": "user", "content": f"[bash]\n{message.command}\n{message.output}"}
    if isinstance(message, CustomMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, BranchSummaryMessage):
        return {"role": "user", "content": f"[branch summary]\n{message.summary}"}
    if isinstance(message, CompactionSummaryMessage):
        return {"role": "user", "content": f"[compaction summary]\n{message.summary}"}
    raise TypeError(f"unsupported message: {type(message).__name__}")


def _user_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    return blocks
