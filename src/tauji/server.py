from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from tauji.config import Settings
from tauji.runtime import AgentRegistry


def create_server(registry: AgentRegistry) -> MCPServer:
    mcp = MCPServer(
        "tauji",
        description="Durable remote coding-agent harness for VPS workspaces.",
        instructions=(
            "Create durable agents for remote workspaces. agent_run returns an explicit run handle "
            "immediately; use run_get/run_list to inspect completion. agent_send steers or resumes a "
            "retained agent. Child agents created by fork use the same durable registry."
        ),
    )

    @mcp.tool()
    async def agent_create(
        workspace: str,
        model: str | None = None,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a durable agent context."""
        return await registry.create_agent(
            workspace=workspace,
            model=model,
            name=name,
            parent_id=parent_id,
        )

    @mcp.tool()
    async def agent_run(agent_id: str, prompt: str) -> dict[str, Any]:
        """Start work and return an explicit run handle immediately."""
        return await registry.run_agent(agent_id, prompt)

    @mcp.tool()
    async def run_get(run_id: str) -> dict[str, Any]:
        """Get durable run state/result."""
        return registry.run_info(run_id)

    @mcp.tool()
    async def run_list(agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """List recent runs for an agent, newest first."""
        return registry.list_runs(agent_id, limit)

    @mcp.tool()
    async def agent_get(agent_id: str) -> dict[str, Any]:
        """Get durable agent metadata and direct child IDs."""
        return registry.agent_info(agent_id)

    @mcp.tool()
    async def agent_list(parent_id: str | None = None) -> list[dict[str, Any]]:
        """List top-level agents or direct children of parent_id."""
        return registry.list_agents(parent_id)

    @mcp.tool()
    async def agent_send(agent_id: str, message: str) -> dict[str, Any]:
        """Steer a running agent or resume an idle retained agent."""
        return await registry.send(agent_id, message)

    @mcp.tool()
    async def run_cancel(run_id: str) -> dict[str, Any]:
        """Cancel an active run."""
        return await registry.cancel_run(run_id)

    @mcp.tool()
    async def agent_children(agent_id: str) -> list[dict[str, Any]]:
        """List direct child agents."""
        return registry.list_agents(agent_id)

    @mcp.tool()
    async def agent_delete(agent_id: str) -> dict[str, Any]:
        """Recursively cancel and delete an agent and its children."""
        return await registry.delete_agent(agent_id)

    return mcp


def main() -> None:
    settings = Settings.from_env()
    registry = AgentRegistry(settings)
    mcp = create_server(registry)
    mcp.run(
        "streamable-http",
        host=settings.host,
        port=settings.port,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
