from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue

if TYPE_CHECKING:
    from tauji.runtime import AgentRuntime

_TOOL = Callable[[Mapping[str, JSONValue]], str | Awaitable[str]]


def coding_tools(runtime: "AgentRuntime") -> list[AgentTool]:
    root = runtime.workspace
    return [
        _tool(
            "read",
            "Read file",
            "Read a UTF-8 text file inside the workspace, optionally by 1-based line range.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            lambda args: _read(root, args),
        ),
        _tool(
            "write",
            "Write file",
            "Create or replace a UTF-8 text file inside the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            lambda args: _write(root, args),
        ),
        _tool(
            "edit",
            "Edit file",
            "Replace exactly one occurrence of literal text in a workspace file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
            lambda args: _edit(root, args),
        ),
        _tool(
            "bash",
            "Run command",
            "Run a shell command with the workspace as cwd. Output is capped at 30k characters.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number", "minimum": 0.1, "maximum": 3600},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            lambda args: _bash(root, args),
        ),
        _tool(
            "fork",
            "Fork agent",
            "Spawn an isolated child agent in this workspace. Returns agent/run handles immediately.",
            {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "name": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            lambda args: _fork(runtime, args),
        ),
    ]


def _tool(
    name: str,
    label: str,
    description: str,
    schema: Mapping[str, JSONValue],
    fn: _TOOL,
) -> AgentTool:
    async def execute(
        _tool_call_id: str,
        args: Mapping[str, JSONValue],
        _signal: Any = None,
        _on_update: Any = None,
    ) -> AgentToolResult:
        value = fn(args)
        if asyncio.iscoroutine(value):
            value = await value
        return AgentToolResult(content=[TextContent(text=str(value))])

    return AgentTool(
        name=name,
        label=label,
        description=description,
        parameters=schema,
        execute_fn=execute,
        execution_mode="sequential",
    )


def _path(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("path must be a non-empty string")
    candidate = Path(raw)
    path = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if path != root and not path.is_relative_to(root):
        raise ValueError("path escapes workspace")
    return path


def _read(root: Path, args: Mapping[str, JSONValue]) -> str:
    path = _path(root, args["path"])
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = int(args.get("start") or 1)
    end = int(args.get("end") or len(lines))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    end = min(end, len(lines))
    return "\n".join(f"{index}: {lines[index - 1]}" for index in range(start, end + 1))


def _write(root: Path, args: Mapping[str, JSONValue]) -> str:
    path = _path(root, args["path"])
    content = args["content"]
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(root)} ({len(content)} chars)"


def _edit(root: Path, args: Mapping[str, JSONValue]) -> str:
    path = _path(root, args["path"])
    old, new = args["old"], args["new"]
    if not isinstance(old, str) or not isinstance(new, str):
        raise ValueError("old and new must be strings")
    if not old:
        raise ValueError("old must not be empty")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"edited {path.relative_to(root)}"


async def _bash(root: Path, args: Mapping[str, JSONValue]) -> str:
    command = args["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    timeout = float(args.get("timeout") or 120)
    if not 0.1 <= timeout <= 3600:
        raise ValueError("timeout must be between 0.1 and 3600 seconds")

    process = await asyncio.create_subprocess_shell(
        command,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RuntimeError(f"command timed out after {timeout:g}s") from exc

    output = stdout.decode(errors="replace")
    if len(output) > 30_000:
        output = "[output truncated to final 30000 chars]\n" + output[-30_000:]
    return f"exit={process.returncode}\n{output}"


async def _fork(runtime: "AgentRuntime", args: Mapping[str, JSONValue]) -> str:
    task = args["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be non-empty")
    name = args.get("name") if isinstance(args.get("name"), str) else None
    model = args.get("model") if isinstance(args.get("model"), str) else None

    child = await runtime.registry.create_agent(
        workspace=str(runtime.workspace),
        model=model,
        name=name,
        parent_id=runtime.agent_id,
    )
    run = await runtime.registry.run_agent(child["id"], task)
    return (
        f"spawned child agent_id={child['id']} run_id={run['id']} "
        f"name={child['name']}"
    )
