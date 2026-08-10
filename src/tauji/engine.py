"""Tauji's minimal provider/tool execution loop."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from tauji.transcript import (
    AgentMessage,
    AssistantMessage,
    JSONValue,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

Checkpoint = Callable[[], None]
ToolExecutor = Callable[[Mapping[str, JSONValue]], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class AgentTool:
    name: str
    description: str
    parameters: Mapping[str, JSONValue]
    execute: ToolExecutor


class ModelProvider(Protocol):
    async def response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AssistantMessage: ...

    async def aclose(self) -> None: ...


class AgentHarness:
    """One Tauji agent's transcript, steering queue, and execution state."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        system: str,
        tools: list[AgentTool],
        messages: Sequence[AgentMessage] = (),
    ) -> None:
        self._provider = provider
        self._model = model
        self._system = system
        self._tools = tools
        self._messages = list(messages)
        self._steering: deque[UserMessage] = deque()
        self._running = False

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def steering_messages(self) -> tuple[UserMessage, ...]:
        return tuple(self._steering)

    @property
    def is_running(self) -> bool:
        return self._running

    def steer(self, content: str) -> None:
        self._steering.append(UserMessage(content=content))

    def prepare_prompt(self, content: str, checkpoint: Checkpoint) -> None:
        """Admit a prompt synchronously before background execution starts."""
        if self._running:
            raise RuntimeError("agent is already running; use agent_send to steer it")
        original_length = len(self._messages)
        try:
            self._append_interrupted_tool_results()
            self._running = True
            self._messages.append(UserMessage(content=content))
            checkpoint()
        except BaseException:
            del self._messages[original_length:]
            self._running = False
            raise

    async def run(self, checkpoint: Checkpoint) -> AssistantMessage:
        if not self._running:
            raise RuntimeError("agent has no admitted prompt")
        try:
            return await self._run(checkpoint)
        finally:
            self.interrupt()

    def interrupt(self) -> None:
        self._append_interrupted_tool_results()
        self._running = False

    async def _run(self, checkpoint: Checkpoint) -> AssistantMessage:
        tools = {tool.name: tool for tool in self._tools}
        while True:
            assistant = await self._provider.response(
                model=self._model,
                system=self._system,
                messages=_provider_context(self._messages),
                tools=self._tools,
            )
            self._messages.append(assistant)
            checkpoint()
            if assistant.stop_reason in {"error", "aborted"}:
                return assistant

            for call in assistant.tool_calls:
                result = await _execute_tool(call, tools)
                self._messages.append(result)
                checkpoint()

            steering = tuple(self._steering)
            self._steering.clear()
            if steering:
                self._messages.extend(steering)
                checkpoint()
            if not assistant.tool_calls and not steering:
                return assistant

    def _append_interrupted_tool_results(self) -> None:
        returned_ids = {
            message.tool_call_id
            for message in self._messages
            if isinstance(message, ToolResultMessage)
        }
        for message in tuple(self._messages):
            if not isinstance(message, AssistantMessage):
                continue
            for call in message.tool_calls:
                if call.id in returned_ids:
                    continue
                returned_ids.add(call.id)
                self._messages.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=[TextContent(text="Tool call interrupted by user")],
                        is_error=True,
                    )
                )


async def _execute_tool(
    call: ToolCall,
    tools: Mapping[str, AgentTool],
) -> ToolResultMessage:
    tool = tools.get(call.name)
    if tool is None:
        return _tool_error(call, f"Tool {call.name} not found")
    try:
        text = await tool.execute(call.arguments)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - tools are an isolation boundary
        return _tool_error(call, str(exc))
    return ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[TextContent(text=text)],
    )


def _tool_error(call: ToolCall, message: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[TextContent(text=message)],
        is_error=True,
    )


def _provider_context(messages: list[AgentMessage]) -> list[AgentMessage]:
    return [
        message
        for message in messages
        if not (
            isinstance(message, AssistantMessage)
            and message.stop_reason in {"error", "aborted"}
            and not message.content
        )
    ]
