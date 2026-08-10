from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue

if TYPE_CHECKING:
    from tauji.runtime import AgentRuntime


def coding_tools(runtime: "AgentRuntime", workspace: Path) -> list[AgentTool]:
    return [
        _tool("read", "Read file", "Read a UTF-8 file inside the workspace.", {
            "type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["path"]
        }, lambda a: _read(workspace, a)),
        _tool("write", "Write file", "Replace or create a UTF-8 file inside the workspace.", {
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]
        }, lambda a: _write(workspace, a)),
        _tool("edit", "Edit file", "Replace one exact text occurrence in a file.", {
            "type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]
        }, lambda a: _edit(workspace, a)),
        _tool("bash", "Run command", "Run a shell command in the workspace.", {
            "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"]
        }, lambda a: _bash(workspace, a)),
        _tool("fork", "Fork agent", "Spawn an isolated child Tau agent. Returns admission handles immediately; inspect with the external agent/run handles later.", {
            "type": "object", "properties": {"task": {"type": "string"}, "name": {"type": "string"}, "model": {"type": "string"}}, "required": ["task"]
        }, lambda a: _fork(runtime, a)),
    ]


def _tool(name: str, label: str, description: str, schema: Mapping[str, JSONValue], fn: Any) -> AgentTool:
    async def execute(_id: str, args: Mapping[str, JSONValue], _signal: Any = None, _update: Any = None) -> AgentToolResult:
        value = fn(args)
        if asyncio.iscoroutine(value):
            value = await value
        return AgentToolResult(content=[TextContent(text=str(value))], details={})
    return AgentTool(name=name, label=label, description=description, parameters=schema, execute_fn=execute, execution_mode="sequential")


def _path(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("path must be a non-empty string")
    p = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if p != root and not p.is_relative_to(root):
        raise ValueError("path escapes workspace")
    return p


def _read(root: Path, a: Mapping[str, JSONValue]) -> str:
    p = _path(root, a["path"])
    lines = p.read_text(errors="replace").splitlines()
    start = max(1, int(a.get("start") or 1))
    end = min(len(lines), int(a.get("end") or len(lines)))
    return "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))


def _write(root: Path, a: Mapping[str, JSONValue]) -> str:
    p = _path(root, a["path"])
    content = a["content"]
    if not isinstance(content, str):
        raise ValueError("content must be string")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {p.relative_to(root)} ({len(content)} chars)"


def _edit(root: Path, a: Mapping[str, JSONValue]) -> str:
    p = _path(root, a["path"])
    old, new = a["old"], a["new"]
    if not isinstance(old, str) or not isinstance(new, str):
        raise ValueError("old/new must be strings")
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}")
    p.write_text(text.replace(old, new, 1))
    return f"edited {p.relative_to(root)}"


async def _bash(root: Path, a: Mapping[str, JSONValue]) -> str:
    command = a["command"]
    if not isinstance(command, str):
        raise ValueError("command must be string")
    timeout = float(a.get("timeout") or 120)
    proc = await asyncio.create_subprocess_shell(
        command, cwd=root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"command timed out after {timeout}s")
    text = out.decode(errors="replace")
    if len(text) > 30000:
        text = text[-30000:]
    return f"exit={proc.returncode}\n{text}"


async def _fork(runtime: "AgentRuntime", a: Mapping[str, JSONValue]) -> str:
    task = a["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be non-empty")
    name = a.get("name") if isinstance(a.get("name"), str) else None
    model = a.get("model") if isinstance(a.get("model"), str) else None
    child = await runtime.create_agent(
        workspace=str(runtime.workspace), model=model, name=name, parent_id=runtime.agent_id
    )
    run = await runtime.registry.run_agent(child["agent_id"], task)
    return f"spawned child agent_id={child['agent_id']} run_id={run['run_id']} name={child['name']}"
