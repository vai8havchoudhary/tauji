import asyncio
from typing import Any

import pytest

from tauji.engine import AgentHarness, AgentTool
from tauji.transcript import AssistantMessage, TextContent, ToolCall, ToolResultMessage


class ToolProvider:
    def __init__(self) -> None:
        self.contexts: list[list[Any]] = []

    async def response(self, **kwargs: Any) -> AssistantMessage:
        self.contexts.append(kwargs["messages"])
        if len(self.contexts) == 1:
            return AssistantMessage(
                content=[ToolCall(id="call_1", name="echo", arguments={"value": "hello"})],
                stop_reason="toolUse",
            )
        return AssistantMessage(content=[TextContent(text="DONE")])

    async def aclose(self) -> None:
        pass


async def test_tool_call_is_persisted_before_execution_and_replayed() -> None:
    provider = ToolProvider()
    executed_after_checkpoints: list[int] = []
    checkpoints = 0

    async def execute(_arguments: Any) -> str:
        executed_after_checkpoints.append(checkpoints)
        return "echoed"

    harness = AgentHarness(
        provider=provider,
        model="test",
        system="test",
        tools=[AgentTool("echo", "Echo", {}, execute)],
    )

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1

    harness.prepare_prompt("go", checkpoint)
    terminal = await harness.run(checkpoint)

    assert terminal.text == "DONE"
    assert executed_after_checkpoints == [2]
    assert any(isinstance(message, ToolResultMessage) for message in provider.contexts[1])


async def test_steering_causes_another_provider_turn() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SteeringProvider:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def response(self, **kwargs: Any) -> AssistantMessage:
            self.calls.append([message.text for message in kwargs["messages"]])
            if len(self.calls) == 1:
                entered.set()
                await release.wait()
                return AssistantMessage(content=[TextContent(text="FIRST")])
            return AssistantMessage(content=[TextContent(text="SECOND")])

        async def aclose(self) -> None:
            pass

    provider = SteeringProvider()
    harness = AgentHarness(provider=provider, model="test", system="test", tools=[])
    harness.prepare_prompt("initial", lambda: None)
    task = asyncio.create_task(harness.run(lambda: None))
    await entered.wait()
    harness.steer("redirect")
    release.set()

    terminal = await task

    assert terminal.text == "SECOND"
    assert provider.calls == [["initial"], ["initial", "FIRST", "redirect"]]


async def test_cancelled_tool_call_gets_interrupted_result() -> None:
    entered = asyncio.Event()

    async def execute(_arguments: Any) -> str:
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    class Provider:
        async def response(self, **_kwargs: Any) -> AssistantMessage:
            return AssistantMessage(
                content=[ToolCall(id="call_1", name="wait", arguments={})],
                stop_reason="toolUse",
            )

        async def aclose(self) -> None:
            pass

    harness = AgentHarness(
        provider=Provider(),
        model="test",
        system="test",
        tools=[AgentTool("wait", "Wait", {}, execute)],
    )
    harness.prepare_prompt("go", lambda: None)
    task = asyncio.create_task(harness.run(lambda: None))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    result = harness.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == "call_1"
    assert result.is_error is True
    assert result.text == "Tool call interrupted by user"
