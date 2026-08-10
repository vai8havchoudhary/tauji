from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from tau_agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from tau_agent.harness import AgentHarness, AgentHarnessConfig
from tau_agent.messages import AssistantMessage, UserMessage
from tauji.config import Settings
from tauji.provider import CLIProxyProvider
from tauji.store import RunStatus, Store
from tauji.tools import coding_tools

SYSTEM_PROMPT = """You are Tauji, a coding agent running beside the repository on a VPS.
Inspect and modify the workspace with tools. Verify substantive changes.
Use fork only for independent context-heavy subtasks. Fork is asynchronous and returns handles,
not the child's answer. Keep final answers concise and operational.
"""


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        row: dict[str, Any],
        messages: list[Any],
    ) -> None:
        self.registry = registry
        self.agent_id = str(row["id"])
        self.name = str(row["name"])
        self.workspace = Path(row["workspace"])
        self.model = str(row["model"])
        self.depth = int(row["depth"])
        self.parent_id = row["parent_id"]
        self.current_run_id: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._control_lock = asyncio.Lock()
        self._cancel_status: Literal["cancelled", "interrupted"] = "cancelled"

        config = AgentHarnessConfig(
            provider=registry.provider,
            model=self.model,
            system=(
                SYSTEM_PROMPT
                + f"\nWorkspace: {self.workspace}\nAgent: {self.name}\nDepth: {self.depth}\n"
            ),
            tools=[],
            queue_mode="all",
        )
        self.harness = AgentHarness(config, messages=messages)
        self.harness.config.tools[:] = coding_tools(self)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start_prompt(self, run_id: str, prompt: str) -> None:
        self._start(run_id, self.harness.prompt(prompt))

    async def send(self, message: str) -> str | None:
        if not message.strip():
            raise ValueError("message must be non-empty")
        async with self._control_lock:
            if self.harness.is_running:
                steering = UserMessage(content=message)
                self.harness.steer_message(steering)
                self._save_messages()
                return self.current_run_id

            follow_up = UserMessage(content=message)
            run_id = self.registry._new_run_id()
            self.registry.store.create_run(
                run_id,
                self.agent_id,
                f"[follow-up] {message}",
                messages=(*self.harness.messages, follow_up),
            )
            self._start(run_id, self.harness.prompt_message(follow_up))
            return run_id

    async def cancel(self, *, status: Literal["cancelled", "interrupted"] = "cancelled") -> None:
        self._cancel_status = status
        self.harness.cancel()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _start(self, run_id: str, stream: Any) -> None:
        if self.is_running:
            raise RuntimeError(f"agent {self.agent_id} already has an active run")
        self.current_run_id = run_id
        self._task = asyncio.create_task(
            self._consume(run_id, stream),
            name=f"tauji:{run_id}",
        )

    async def _consume(self, run_id: str, stream: Any) -> None:
        self.registry.set_agent_status(self.agent_id, "running")
        terminal: AssistantMessage | None = None
        try:
            async for event in stream:
                if isinstance(
                    event,
                    (AgentStartEvent, MessageEndEvent, ToolExecutionStartEvent, TurnEndEvent),
                ):
                    self._save_messages()
                if isinstance(event, AgentEndEvent):
                    terminal = _last_assistant(event.messages)

            self._save_messages()
            if terminal is not None and terminal.stop_reason in {"error", "aborted"}:
                status: RunStatus = "cancelled" if terminal.stop_reason == "aborted" else "failed"
                error = terminal.error_message or terminal.stop_reason
                changed = self.registry.store.finish_run(run_id, status, error=error)
                if changed:
                    await self._notify_parent(run_id, error=error)
            else:
                result = terminal.text if terminal is not None else ""
                changed = self.registry.store.finish_run(run_id, "completed", result=result)
                if changed:
                    await self._notify_parent(run_id, result=result)
            self.registry.set_agent_status(self.agent_id, "idle")
        except asyncio.CancelledError:
            self._save_messages()
            error = "cancelled" if self._cancel_status == "cancelled" else "tauji stopped"
            changed = self.registry.store.finish_run(run_id, self._cancel_status, error=error)
            self.registry.set_agent_status(self.agent_id, "idle")
            if changed and self._cancel_status == "cancelled":
                await self._notify_parent(run_id, error=error)
            raise
        except Exception as exc:
            error = str(exc)
            self._save_messages()
            changed = self.registry.store.finish_run(run_id, "failed", error=error)
            self.registry.set_agent_status(self.agent_id, "idle")
            if changed:
                await self._notify_parent(run_id, error=error)
        finally:
            if self.current_run_id == run_id:
                self.current_run_id = None
            self._cancel_status = "cancelled"

    def _save_messages(self) -> None:
        queued = self.harness.queued_messages
        self.registry.store.save_messages(
            self.agent_id,
            (*self.harness.messages, *queued.steering, *queued.follow_up),
        )

    async def _notify_parent(
        self,
        run_id: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        if self.parent_id is None:
            return
        try:
            await self.registry.deliver_child_outcome(
                parent_id=str(self.parent_id),
                child_id=self.agent_id,
                child_name=self.name,
                run_id=run_id,
                result=result,
                error=error,
            )
        except Exception:
            return


class AgentRegistry:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: CLIProxyProvider | None = None,
        store: Store | None = None,
    ) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or Store(settings.data_dir / "tauji.db")
        self.provider = provider or CLIProxyProvider(
            settings.base_url,
            settings.api_key,
            timeout=settings.request_timeout,
        )
        self._agents: dict[str, AgentRuntime] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        runtimes = list(self._agents.values())
        await asyncio.gather(
            *(runtime.cancel(status="interrupted") for runtime in runtimes),
            return_exceptions=True,
        )
        await self.provider.aclose()
        self.store.close()

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
        if parent_id is not None:
            parent = await self.get_runtime(parent_id)
            depth = parent.depth + 1
            if depth > self.settings.max_depth:
                raise RuntimeError(
                    f"recursive depth {depth} exceeds TAUJI_MAX_DEPTH={self.settings.max_depth}"
                )
            if path != parent.workspace:
                raise ValueError("child agents must use the parent workspace")

        agent_id = self._new_agent_id()
        row = {
            "id": agent_id,
            "name": _normalize_name(name) if name else f"agent-{agent_id[-6:]}",
            "parent_id": parent_id,
            "workspace": str(path),
            "model": (model or self.settings.default_model).strip(),
            "depth": depth,
            "status": "idle",
        }
        if not row["model"]:
            raise ValueError("model must be non-empty")

        self.store.upsert_agent(row)
        async with self._lock:
            self._agents[agent_id] = self._build(row)
        return self.agent_info(agent_id)

    async def get_runtime(self, agent_id: str) -> AgentRuntime:
        async with self._lock:
            runtime = self._agents.get(agent_id)
            if runtime is not None:
                return runtime
            row = self.store.get_agent(agent_id)
            if row is None:
                raise KeyError(f"unknown agent: {agent_id}")
            runtime = self._build(row)
            self._agents[agent_id] = runtime
            return runtime

    async def run_agent(self, agent_id: str, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        runtime = await self.get_runtime(agent_id)
        async with runtime._control_lock:
            if runtime.is_running:
                raise RuntimeError(f"agent {agent_id} already has an active run")
            run_id = self._new_run_id()
            durable_messages = (*runtime.harness.messages, UserMessage(content=prompt))
            self.store.create_run(run_id, agent_id, prompt, messages=durable_messages)
            try:
                await runtime.start_prompt(run_id, prompt)
            except Exception:
                self.store.finish_run(run_id, "failed", error="run admission failed")
                raise
        return self.run_info(run_id)

    async def send(self, agent_id: str, message: str) -> dict[str, Any]:
        runtime = await self.get_runtime(agent_id)
        run_id = await runtime.send(message)
        return {"agent_id": agent_id, "accepted": True, "run_id": run_id}

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        if row["status"] != "running":
            return row

        runtime = await self.get_runtime(str(row["agent_id"]))
        async with runtime._control_lock:
            current = self.store.get_run(run_id)
            if current is None or current["status"] != "running":
                return self.run_info(run_id)
            if runtime.current_run_id == run_id and runtime.is_running:
                await runtime.cancel()
            else:
                self.store.finish_run(run_id, "cancelled", error="no active runtime")
        return self.run_info(run_id)

    async def delete_agent(self, agent_id: str) -> dict[str, Any]:
        runtime = await self.get_runtime(agent_id)
        for child in self.store.list_agents(agent_id):
            await self.delete_agent(str(child["id"]))
        async with runtime._control_lock:
            await runtime.cancel()
        info = self.agent_info(agent_id)
        self.store.delete_agent(agent_id)
        async with self._lock:
            self._agents.pop(agent_id, None)
        return info

    async def deliver_child_outcome(
        self,
        *,
        parent_id: str,
        child_id: str,
        child_name: str,
        run_id: str,
        result: str | None,
        error: str | None,
    ) -> None:
        tag = "child_error" if error is not None else "child_result"
        body = error if error is not None else (result or "")
        parent = await self.get_runtime(parent_id)
        await parent.send(
            f"<{tag} agent_id={child_id!r} run_id={run_id!r} name={child_name!r}>\n{body}\n</{tag}>"
        )

    def agent_info(self, agent_id: str) -> dict[str, Any]:
        row = self.store.get_agent(agent_id)
        if row is None:
            raise KeyError(f"unknown agent: {agent_id}")
        row["children"] = [child["id"] for child in self.store.list_agents(agent_id)]
        return row

    def list_agents(self, parent_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_agents(parent_id)

    def run_info(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return row

    def list_runs(self, agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if self.store.get_agent(agent_id) is None:
            raise KeyError(f"unknown agent: {agent_id}")
        return self.store.list_runs(agent_id, limit)

    def set_agent_status(self, agent_id: str, status: str) -> None:
        row = self.store.get_agent(agent_id)
        if row is None:
            return
        row["status"] = status
        self.store.upsert_agent(row)

    def _build(self, row: dict[str, Any]) -> AgentRuntime:
        return AgentRuntime(
            registry=self,
            row=row,
            messages=self.store.load_messages(str(row["id"])),
        )

    @staticmethod
    def _new_agent_id() -> str:
        return f"agt_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _new_run_id() -> str:
        return f"run_{uuid.uuid4().hex[:12]}"


def _normalize_name(name: str) -> str:
    normalized = "-".join(name.strip().split())
    if not normalized:
        raise ValueError("name must be non-empty")
    if len(normalized) > 64:
        raise ValueError("name must be at most 64 characters")
    return normalized


def _last_assistant(messages: list[Any] | tuple[Any, ...]) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None
