from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from tau_agent.events import AgentEndEvent
from tau_agent.harness import AgentHarness, AgentHarnessConfig
from tau_agent.messages import AssistantMessage

from tauji.config import Settings
from tauji.provider import CLIProxyProvider
from tauji.store import Store
from tauji.tools import coding_tools


SYSTEM_PROMPT = """You are Tauji, a remote coding agent running beside the repository on a VPS.
Use tools to inspect and modify the workspace. Work programmatically and verify changes.
Use fork for independent context-heavy subtasks. A fork returns child/run handles immediately;
do not assume its result is returned synchronously. Keep final answers concise and operational.
"""


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: "AgentRegistry",
        agent_id: str,
        name: str,
        workspace: Path,
        model: str,
        depth: int,
        parent_id: str | None,
        messages: list[Any],
    ) -> None:
        self.registry = registry
        self.agent_id = agent_id
        self.name = name
        self.workspace = workspace
        self.model = model
        self.depth = depth
        self.parent_id = parent_id
        self._current_run: str | None = None
        self._task: asyncio.Task[None] | None = None
        config = AgentHarnessConfig(
            provider=registry.provider,
            model=model,
            system=SYSTEM_PROMPT + f"\nWorkspace: {workspace}\nAgent: {name}\nDepth: {depth}",
            tools=[],
            queue_mode="all",
        )
        self.harness = AgentHarness(config, messages=messages)
        self.harness.config.tools[:] = coding_tools(self, workspace)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def create_agent(
        self, *, workspace: str, model: str | None, name: str | None, parent_id: str | None
    ) -> dict[str, Any]:
        return await self.registry.create_agent(
            workspace=workspace, model=model, name=name, parent_id=parent_id
        )

    async def start(self, run_id: str, prompt: str) -> None:
        if self.is_running:
            raise RuntimeError(f"agent {self.agent_id} already has a running task")
        self._current_run = run_id
        self._task = asyncio.create_task(self._execute(run_id, prompt), name=f"tauji:{run_id}")

    async def send(self, message: str) -> None:
        if self.harness.is_running:
            self.harness.steer(message)
        else:
            self.harness.follow_up(message)
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            self.registry.store.create_run(run_id, self.agent_id, f"[follow-up] {message}")
            await self.start_continue(run_id)

    async def start_continue(self, run_id: str) -> None:
        if self.is_running:
            return
        self._current_run = run_id
        self._task = asyncio.create_task(self._continue(run_id), name=f"tauji:{run_id}")

    async def cancel(self) -> None:
        self.harness.cancel()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _execute(self, run_id: str, prompt: str) -> None:
        await self._consume(run_id, self.harness.prompt(prompt))

    async def _continue(self, run_id: str) -> None:
        await self._consume(run_id, self.harness.continue_())

    async def _consume(self, run_id: str, stream: Any) -> None:
        self.registry._set_agent_status(self.agent_id, "running")
        result = ""
        try:
            async for event in stream:
                if isinstance(event, AgentEndEvent):
                    result = _final_text(event.messages)
            self.registry.store.save_messages(self.agent_id, self.harness.messages)
            self.registry.store.finish_run(run_id, "completed", result=result)
            self.registry._set_agent_status(self.agent_id, "idle")
            if self.parent_id:
                await self.registry.deliver_to_parent(
                    self.parent_id,
                    f"<child_result agent_id={self.agent_id!r} run_id={run_id!r} name={self.name!r}>\n"
                    f"{result}\n</child_result>",
                )
        except asyncio.CancelledError:
            self.registry.store.save_messages(self.agent_id, self.harness.messages)
            self.registry.store.finish_run(run_id, "cancelled", error="cancelled")
            self.registry._set_agent_status(self.agent_id, "idle")
            raise
        except Exception as exc:
            self.registry.store.save_messages(self.agent_id, self.harness.messages)
            self.registry.store.finish_run(run_id, "failed", error=str(exc))
            self.registry._set_agent_status(self.agent_id, "idle")


class AgentRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.data_dir / "tauji.db")
        self.provider = CLIProxyProvider(settings.base_url, settings.api_key)
        self._agents: dict[str, AgentRuntime] = {}
        self._lock = asyncio.Lock()

    async def create_agent(
        self,
        *,
        workspace: str,
        model: str | None = None,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        path = self.settings.validate_workspace(workspace)
        depth = 0
        if parent_id:
            parent = await self.get_runtime(parent_id)
            depth = parent.depth + 1
            if depth > self.settings.max_depth:
                raise RuntimeError(
                    f"recursive depth {depth} exceeds TAUJI_MAX_DEPTH={self.settings.max_depth}"
                )
            if path != parent.workspace:
                raise ValueError("child agents must use the parent workspace")
        agent_id = f"agt_{uuid.uuid4().hex[:12]}"
        agent_name = name or f"agent-{agent_id[-6:]}"
        row = {
            "id": agent_id,
            "name": agent_name,
            "parent_id": parent_id,
            "workspace": str(path),
            "model": model or self.settings.default_model,
            "depth": depth,
            "status": "idle",
        }
        self.store.upsert_agent(row)
        async with self._lock:
            self._agents[agent_id] = self._build(row)
        return self.agent_info(agent_id)

    async def get_runtime(self, agent_id: str) -> AgentRuntime:
        async with self._lock:
            runtime = self._agents.get(agent_id)
            if runtime:
                return runtime
            row = self.store.get_agent(agent_id)
            if not row:
                raise KeyError(f"unknown agent: {agent_id}")
            runtime = self._build(row)
            self._agents[agent_id] = runtime
            return runtime

    def _build(self, row: dict[str, Any]) -> AgentRuntime:
        return AgentRuntime(
            registry=self,
            agent_id=row["id"],
            name=row["name"],
            workspace=Path(row["workspace"]),
            model=row["model"],
            depth=int(row["depth"]),
            parent_id=row["parent_id"],
            messages=self.store.load_messages(row["id"]),
        )

    async def run_agent(self, agent_id: str, prompt: str) -> dict[str, Any]:
        runtime = await self.get_runtime(agent_id)
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.store.create_run(run_id, agent_id, prompt)
        await runtime.start(run_id, prompt)
        return {"run_id": run_id, "agent_id": agent_id, "status": "running"}

    async def send(self, agent_id: str, message: str) -> dict[str, Any]:
        runtime = await self.get_runtime(agent_id)
        await runtime.send(message)
        return {"agent_id": agent_id, "accepted": True}

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        if not row:
            raise KeyError(f"unknown run: {run_id}")
        runtime = await self.get_runtime(row["agent_id"])
        if runtime._current_run == run_id and runtime.is_running:
            await runtime.cancel()
        else:
            self.store.finish_run(run_id, "cancelled", error="cancelled before/after active runtime")
        return self.run_info(run_id)

    async def delete_agent(self, agent_id: str) -> dict[str, Any]:
        runtime = await self.get_runtime(agent_id)
        await runtime.cancel()
        info = self.agent_info(agent_id)
        self.store.delete_agent(agent_id)
        async with self._lock:
            self._agents.pop(agent_id, None)
        return info

    async def deliver_to_parent(self, parent_id: str, message: str) -> None:
        parent = await self.get_runtime(parent_id)
        await parent.send(message)

    def agent_info(self, agent_id: str) -> dict[str, Any]:
        row = self.store.get_agent(agent_id)
        if not row:
            raise KeyError(f"unknown agent: {agent_id}")
        row["children"] = [r["id"] for r in self.store.list_children(agent_id)]
        return row

    def run_info(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        if not row:
            raise KeyError(f"unknown run: {run_id}")
        return row

    def children(self, agent_id: str) -> list[dict[str, Any]]:
        return self.store.list_children(agent_id)

    def _set_agent_status(self, agent_id: str, status: str) -> None:
        row = self.store.get_agent(agent_id)
        if not row:
            return
        row["status"] = status
        self.store.upsert_agent(
            {k: row[k] for k in ("id", "name", "parent_id", "workspace", "model", "depth", "status")}
        )


def _final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage) and message.text:
            return message.text
    return ""
