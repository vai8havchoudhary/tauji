import asyncio
from pathlib import Path
from typing import Any

import pytest

from tauji.config import Settings
from tauji.runtime import AgentRegistry
from tauji.store import Store
from tauji.transcript import AssistantMessage, TextContent, UserMessage


class GateProvider:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def response(self, **_kwargs: Any) -> AssistantMessage:
        self.entered.set()
        await self.release.wait()
        return AssistantMessage(
            content=[TextContent(text="DONE")],
            stop_reason="stop",
        )

    async def aclose(self) -> None:
        pass


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def response(self, **kwargs: Any) -> AssistantMessage:
        self.calls.append([message.text for message in kwargs["messages"]])
        return AssistantMessage(
            content=[TextContent(text="DONE")],
            stop_reason="stop",
        )

    async def aclose(self) -> None:
        pass


def settings(tmp_path: Path) -> Settings:
    root = tmp_path / "workspace"
    root.mkdir()
    return Settings(
        base_url="http://proxy/v1",
        api_key="key",
        default_model="test",
        data_dir=tmp_path / "data",
        workspace_roots=(root,),
        host="127.0.0.1",
        port=8765,
        max_depth=1,
        request_timeout=30,
    )


async def test_prompt_is_durable_while_provider_is_running(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = GateProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    run = await registry.run_agent(agent["id"], "durable prompt")
    await provider.entered.wait()
    messages = registry.store.load_messages(agent["id"])
    assert any(
        isinstance(message, UserMessage) and message.text == "durable prompt"
        for message in messages
    )
    await registry.aclose()
    reopened = Store(config.data_dir / "tauji.db")
    assert reopened.get_run(run["id"])["status"] == "interrupted"
    reopened.close()


async def test_steering_is_durable_before_provider_finishes(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = GateProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    run = await registry.run_agent(agent["id"], "initial")
    await provider.entered.wait()
    steering = await registry.send(agent["id"], "durable steering")
    assert steering["run_id"] == run["id"]
    texts = [
        message.text
        for message in registry.store.load_messages(agent["id"])
        if isinstance(message, UserMessage)
    ]
    assert texts == ["initial", "durable steering"]
    await registry.aclose()


async def test_idle_send_prompts_provider_without_unsolicited_turn(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = RecordingProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    sent = await registry.send(agent["id"], "follow up now")
    runtime = await registry.get_runtime(agent["id"])
    assert runtime._task is not None
    await runtime._task
    assert sent["run_id"] is not None
    assert provider.calls == [["follow up now"]]
    await registry.aclose()


async def test_send_waits_for_finishing_runtime_before_persisting(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = RecordingProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    runtime = await registry.get_runtime(agent["id"])
    release = asyncio.Event()

    async def finishing_notification() -> None:
        await release.wait()

    runtime._task = asyncio.create_task(finishing_notification())
    send_task = asyncio.create_task(registry.send(agent["id"], "after notification"))
    await asyncio.sleep(0.05)
    assert not send_task.done()
    assert registry.list_runs(agent["id"]) == []

    release.set()
    sent = await send_task
    assert sent["run_id"] is not None
    assert runtime._task is not None
    await runtime._task
    assert len(registry.list_runs(agent["id"])) == 1
    assert provider.calls == [["after notification"]]
    await registry.aclose()


async def test_cancelled_send_does_not_resume_after_finishing_runtime(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = RecordingProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    runtime = await registry.get_runtime(agent["id"])
    release = asyncio.Event()

    async def finishing_notification() -> None:
        await release.wait()

    runtime._task = asyncio.create_task(finishing_notification())
    send_task = asyncio.create_task(registry.send(agent["id"], "must not run"))
    await asyncio.sleep(0.05)
    send_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await send_task
    release.set()
    assert runtime._task is not None
    await runtime._task
    assert registry.list_runs(agent["id"]) == []
    assert provider.calls == []
    await registry.aclose()


async def test_late_cancel_cannot_overwrite_completion(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = GateProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    run = await registry.run_agent(agent["id"], "go")
    runtime = await registry.get_runtime(agent["id"])
    await provider.entered.wait()

    await registry._lock.acquire()
    cancel_task = asyncio.create_task(registry.cancel_run(run["id"]))
    await asyncio.sleep(0.05)
    provider.release.set()
    await runtime._task
    registry._lock.release()

    assert (await cancel_task)["status"] == "completed"
    await registry.aclose()


async def test_immediate_cancel_finishes_run_before_task_starts(tmp_path: Path) -> None:
    config = settings(tmp_path)
    registry = AgentRegistry(config, provider=GateProvider())
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    run = await registry.run_agent(agent["id"], "cancel immediately")

    cancelled = await registry.cancel_run(run["id"])

    assert cancelled["status"] == "cancelled"
    assert registry.agent_info(agent["id"])["status"] == "idle"
    assert [message.text for message in registry.store.load_messages(agent["id"])] == [
        "cancel immediately"
    ]
    await registry.aclose()


async def test_concurrent_run_admission_creates_one_run(tmp_path: Path) -> None:
    config = settings(tmp_path)
    provider = GateProvider()
    registry = AgentRegistry(config, provider=provider)
    agent = await registry.create_agent(workspace=str(config.workspace_roots[0]))
    outcomes = await asyncio.gather(
        *(registry.run_agent(agent["id"], f"prompt {index}") for index in range(12)),
        return_exceptions=True,
    )
    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    assert len(registry.list_runs(agent["id"], limit=20)) == 1
    await registry.aclose()
