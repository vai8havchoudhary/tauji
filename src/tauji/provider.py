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
    """Single OpenAI-compatible Chat Completions boundary for every CLIProxyAPI model."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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
        if signal and signal.is_cancelled():
            yield AssistantErrorEvent(
                reason="aborted",
                error=AssistantMessage(model=model, provider="cliproxy", api="openai-chat", stop_reason="aborted"),
            )
            return
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *[_message_to_openai(m) for m in messages]],
            "stream": False,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": dict(t.parameters),
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            choice = data["choices"][0]
            msg = choice["message"]
            blocks = []
            text = msg.get("content") or ""
            if text:
                blocks.append(TextContent(text=text))
            for raw in msg.get("tool_calls") or []:
                fn = raw["function"]
                args = fn.get("arguments") or "{}"
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed = {"_raw": args}
                blocks.append(ToolCall(id=raw["id"], name=fn["name"], arguments=parsed))
            finish = choice.get("finish_reason")
            stop_reason = "toolUse" if blocks and any(isinstance(b, ToolCall) for b in blocks) else (
                "length" if finish == "length" else "stop"
            )
            usage_raw = data.get("usage") or {}
            message = AssistantMessage(
                content=blocks,
                api="openai-chat",
                provider="cliproxy",
                model=model,
                response_id=data.get("id"),
                usage=Usage(
                    input=int(usage_raw.get("prompt_tokens") or 0),
                    output=int(usage_raw.get("completion_tokens") or 0),
                    total_tokens=int(usage_raw.get("total_tokens") or 0),
                ),
                stop_reason=stop_reason,
            )
            yield AssistantDoneEvent(reason=stop_reason, message=message)
        except Exception as exc:
            error = AssistantMessage(
                content=[],
                api="openai-chat",
                provider="cliproxy",
                model=model,
                stop_reason="error",
                error_message=str(exc),
            )
            yield AssistantErrorEvent(reason="error", error=error)


def _message_to_openai(message: AgentMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": _user_content(message.content)}
    if isinstance(message, AssistantMessage):
        out: dict[str, Any] = {"role": "assistant", "content": message.text or None}
        calls = []
        for call in message.tool_calls:
            calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
            )
        if calls:
            out["tool_calls"] = calls
        return out
    if isinstance(message, ToolResultMessage):
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text}
    if isinstance(message, BashExecutionMessage):
        return {"role": "user", "content": f"[bash]\n{message.command}\n{message.output}"}
    if isinstance(message, CustomMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, BranchSummaryMessage):
        return {"role": "user", "content": f"[branch summary]\n{message.summary}"}
    if isinstance(message, CompactionSummaryMessage):
        return {"role": "user", "content": f"[compaction summary]\n{message.summary}"}
    raise TypeError(type(message).__name__)


def _user_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    result = []
    for block in content:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            result.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    return result
