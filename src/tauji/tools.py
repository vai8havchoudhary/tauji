from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from tau_agent.messages import TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue

if TYPE_CHECKING:
    from tauji.runtime import AgentRuntime

_TOOL = Callable[[Mapping[str, JSONValue]], str | Awaitable[str]]


def coding_tools(runtime: AgentRuntime) -> list[AgentTool]:
    root = runtime.workspace
    settings = runtime.registry.settings
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
            "Run a sandboxed shell command in the workspace. Output is capped at 30k characters.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number", "minimum": 0.1, "maximum": 3600},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            lambda args: _bash(
                root,
                args,
                bwrap_path=settings.bwrap_path,
                allow_network=settings.bash_network,
            ),
        ),
        _tool(
            "fork",
            "Fork agent",
            (
                "Spawn an isolated child agent in this workspace. "
                "Returns agent/run handles immediately."
            ),
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
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        value = fn(arguments)
        text = value if isinstance(value, str) else await value
        return AgentToolResult(content=[TextContent(text=text)])

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
    start = _integer_arg(args.get("start"), default=1, name="start")
    end = _integer_arg(args.get("end"), default=len(lines), name="end")
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


async def _bash(
    root: Path,
    args: Mapping[str, JSONValue],
    *,
    bwrap_path: str = "bwrap",
    allow_network: bool = False,
) -> str:
    command = args["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    timeout = _float_arg(args.get("timeout"), default=120, name="timeout")
    if not 0.1 <= timeout <= 3600:
        raise ValueError("timeout must be between 0.1 and 3600 seconds")

    sandbox = _bwrap_command(root, command, bwrap_path=bwrap_path, allow_network=allow_network)
    process = await asyncio.create_subprocess_exec(
        *sandbox,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            "HOME": "/home/sandbox",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "TMPDIR": "/tmp",
        },
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        await asyncio.shield(_terminate_process_group(process))
        raise RuntimeError(f"command timed out after {timeout:g}s") from exc
    except asyncio.CancelledError:
        await asyncio.shield(_terminate_process_group(process))
        raise

    output = stdout.decode(errors="replace")
    if len(output) > 30_000:
        output = "[output truncated to final 30000 chars]\n" + output[-30_000:]
    return f"exit={process.returncode}\n{output}"


def _bwrap_command(
    root: Path,
    command: str,
    *,
    bwrap_path: str,
    allow_network: bool,
) -> list[str]:
    executable = shutil.which(bwrap_path)
    if executable is None:
        raise RuntimeError(
            f"bash sandbox unavailable: {bwrap_path!r} was not found; command was not run"
        )

    root = root.resolve()
    argv = [executable, "--unshare-all"]
    if allow_network:
        argv.append("--share-net")
    argv.extend(
        [
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/sbin",
            "/sbin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--dir",
            "/etc",
        ]
    )

    for config_path in (
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/etc/ssl/certs",
        "/etc/ca-certificates",
        "/etc/ld.so.cache",
        "/etc/localtime",
        "/etc/os-release",
    ):
        if Path(config_path).exists():
            argv.extend(["--ro-bind", config_path, config_path])

    created = {Path("/"), Path("/etc"), Path("/usr")}
    for parent in reversed(root.parents):
        if parent in created or parent.is_relative_to("/usr"):
            continue
        argv.extend(["--dir", str(parent)])
        created.add(parent)
    sandbox_home = Path("/home/sandbox")
    for path in (sandbox_home.parent, sandbox_home):
        if path not in created:
            argv.extend(["--dir", str(path)])
            created.add(path)

    argv.extend(
        [
            "--bind",
            str(root),
            str(root),
            "--chdir",
            str(root),
            "--clearenv",
            "--setenv",
            "HOME",
            str(sandbox_home),
            "--setenv",
            "PATH",
            "/usr/sbin:/usr/bin:/sbin:/bin",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "/bin/bash",
            "-lc",
            command,
        ]
    )
    return argv


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()


def _integer_arg(value: JSONValue | None, *, default: int, name: str) -> int:
    if value is None:
        return default
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return int(value)


def _float_arg(value: JSONValue | None, *, default: float, name: str) -> float:
    if value is None:
        return default
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


async def _fork(runtime: AgentRuntime, args: Mapping[str, JSONValue]) -> str:
    task = args["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be non-empty")
    raw_name = args.get("name")
    raw_model = args.get("model")
    name = raw_name if isinstance(raw_name, str) else None
    model = raw_model if isinstance(raw_model, str) else None

    child = await runtime.registry.create_agent(
        workspace=str(runtime.workspace),
        model=model,
        name=name,
        parent_id=runtime.agent_id,
    )
    run = await runtime.registry.run_agent(child["id"], task)
    return f"spawned child agent_id={child['id']} run_id={run['id']} name={child['name']}"
