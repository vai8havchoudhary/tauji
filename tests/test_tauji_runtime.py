import asyncio
from pathlib import Path
from typing import Any

from tau_agent.messages import AssistantMessage, TextContent, Usage, UserMessage
from tau_agent.provider_events import AssistantDoneEvent
from tauji.config import Settings
from tauji.runtime import AgentRegistry
from tauji.store import Store


class GateProvider:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_response(self, **_kwargs: Any) -> Any:
        self.entered.set()
        await self.release.wait()
        message = AssistantMessage(
            content=[TextContent(text="DONE")],
            api="test",
            provider="test",
            model="test",
            usage=Usage(input=0, output=0, total_tokens=0),
            stop_reason="stop",
        )
        yield AssistantDoneEvent(reason="stop", message=message)

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
