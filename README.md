# Tauji

Tauji is a remote MCP coding-agent harness for code that already lives on a VPS.

It is derived from Hugging Face Tau, but the product boundary is different:

```text
local Claude Code / Codex
          |
          | MCP 2026-07-28 over Tailscale
          v
        Tauji
          |
     AgentRegistry
       /      \
 Tau harness  child Tau harnesses
          |
     CLIProxyAPI only
          |
  any model CLIProxyAPI exposes
```

## Design

- `tau_agent` remains the small provider/tool agent loop.
- `tauji` is the VPS runtime and MCP server.
- MCP uses the official Python SDK v2 and Streamable HTTP.
- MCP 2026-07-28 is stateless at the protocol layer; Tauji state is explicit through `agent_id` and `run_id`.
- Runs are non-blocking: `agent_run` returns immediately and `run_get` retrieves the eventual result.
- Agents are retained contexts. `agent_send` steers a running agent or continues an idle one.
- Internal `fork` creates isolated child Tau agents through the same `AgentRegistry`.
- SQLite persists agents, transcripts, and runs across MCP/client reconnects.
- Runs in progress during a Tauji process restart are marked `interrupted`.
- All inference uses one OpenAI-compatible CLIProxyAPI boundary: `/v1/chat/completions`.
- Workspaces are existing VPS directories; no upload/sync abstraction exists.

The 2026-07-28 Tasks extension (SEP-2663) is intentionally not emulated: the official Python SDK v2 does not implement it yet. The explicit `run_id` API is the compatibility seam for mapping to Tasks once the Python extension implementation is ready.

## Install

```bash
git clone -b mcp-agent-harness https://github.com/vai8havchoudhary/tauji
cd tauji
uv sync
```

## Configuration

```bash
export CLIPROXY_BASE_URL=http://127.0.0.1:8317/v1
export CLIPROXY_API_KEY=cliproxy
export TAUJI_MODEL=gpt-5.6

# Colon-separated VPS roots agents may work under.
export TAUJI_WORKSPACE_ROOTS=/startup:/srv/code:/home/vaibhav

# Bind directly to the VPS Tailscale address (recommended), not 0.0.0.0.
export TAUJI_HOST=100.x.y.z
export TAUJI_PORT=8765

export TAUJI_DATA_DIR=$HOME/.tauji
export TAUJI_MAX_DEPTH=1
```

## Run

```bash
uv run tauji
```

MCP endpoint:

```text
http://<tailscale-ip>:8765/mcp
```

Because this is on Tailscale, keep the port inaccessible from the public interface/firewall.

## MCP workflow

Create:

```text
agent_create(
  workspace="/startup/topology",
  model="gpt-5.6",
  name="topology"
)
-> agent_id=agt_...
```

Run:

```text
agent_run(
  agent_id="agt_...",
  prompt="Understand identity reconciliation and verify the implementation."
)
-> run_id=run_..., status=running
```

Poll:

```text
run_get(run_id="run_...")
```

Continue the retained context:

```text
agent_send(
  agent_id="agt_...",
  message="Now inspect the patch and verify the same invariants."
)
```

The agent itself can use `fork` for parallel isolated workers. Child completion is automatically injected back into the parent as a follow-up message.

## Local Claude Code / Codex

Point the local MCP client at:

```text
http://<tailscale-ip>:8765/mcp
```

No SSH process transport is required. SSH/tmux remain useful for operating the VPS service, starting CLIProxyAPI, and inspecting repositories.

## Development

```bash
uv run pytest
uv run ruff check src/tauji tests
uv run mypy src/tauji
```
