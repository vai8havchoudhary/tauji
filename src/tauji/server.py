from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from tauji.config import Settings
from tauji.runtime import AgentRegistry

settings = Settings.from_env()
registry = AgentRegistry(settings)

mcp = MCPServer(
    "tauji",
    description="Durable remote coding-agent harness for VPS workspaces.",
    instructions=(
        "Tauji agents run next to code on the remote machine. Create an agent for a workspace, "
        "start non-blocking runs, poll run_get, steer retained agents, and inspect child agents. "
        "All agent model inference is routed through CLIProxyAPI."
    ),
)


@mcp.tool()
async def agent_create(
    workspace: str,
    model: str | None = None,
    name: str | None = None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    """Create a durable agent context. Child agents inherit the parent's workspace."""
    return await registry.create_agent(
        workspace=workspace, model=model, name=name, parent_id=parent_id
    )


@mcp.tool()
async def agent_run(agent_id: str, prompt: str) -> dict[str, Any]:
    """Start work and return a run handle immediately; poll run_get for completion."""
    return await registry.run_agent(agent_id, prompt)


@mcp.tool()
async def run_get(run_id: str) -> dict[str, Any]:
    """Read durable run status/result: running, completed, failed, cancelled, interrupted."""
    return registry.run_info(run_id)


@mcp.tool()
async def agent_get(agent_id: str) -> dict[str, Any]:
    """Read durable agent metadata and direct child IDs."""
    return registry.agent_info(agent_id)


@mcp.tool()
async def agent_send(agent_id: str, message: str) -> dict[str, Any]:
    """Steer a running agent or queue a follow-up on an idle retained agent."""
    return await registry.send(agent_id, message)


@mcp.tool()
async def run_cancel(run_id: str) -> dict[str, Any]:
    """Cancel a currently active run."""
    return await registry.cancel_run(run_id)


@mcp.tool()
async def agent_children(agent_id: str) -> list[dict[str, Any]]:
    """List direct child agents created by recursive fork."""
    return registry.children(agent_id)


@mcp.tool()
async def agent_delete(agent_id: str) -> dict[str, Any]:
    """Cancel and delete an agent's active registry/transcript state."""
    return await registry.delete_agent(agent_id)


def main() -> None:
    mcp.run(
        "streamable-http",
        host=settings.host,
        port=settings.port,
        json_response=True,
    )


if __name__ == "__main__":
    main()
