# Tauji

Tauji is a small durable MCP agent harness intended to run on the same VPS as your code.

Your local Claude Code or Codex client talks to Tauji over MCP/Streamable HTTP (typically over
Tailscale). Tauji keeps explicit durable `agent_id` and `run_id` handles, executes agents next to
the repository, and sends every model request through one OpenAI-compatible CLIProxyAPI endpoint.

## Architecture

```text
Claude Code / Codex
        |
        | MCP / Streamable HTTP
        | (protocol negotiated by the SDK)
        v
      Tauji
        |
   AgentRegistry ---- SQLite
     |      |
   agent   child agents
     \      /
     Tauji engine
          |
      CLIProxyAPI
          |
     any routed model
```

There is no TUI, provider catalog, OAuth layer, Jupyter kernel, reusable Tau compatibility package,
or hidden MCP session state. Recursive `fork` creates another instance of the same Tauji runtime.

## Install

```bash
git clone https://github.com/vai8havchoudhary/tauji.git
cd tauji
git switch mcp-agent-harness
uv sync
```

Tauji's shell tool requires Bubblewrap and fails closed when it is unavailable. On Ubuntu:

```bash
sudo apt-get install bubblewrap
```

## Configure

```bash
export CLIPROXY_BASE_URL=http://127.0.0.1:8317/v1
export CLIPROXY_API_KEY=cliproxy
export TAUJI_MODEL=gpt-5.6

# Colon-separated directories that an MCP caller may select as workspaces.
export TAUJI_WORKSPACE_ROOTS=/startup:/srv/code:/home/$USER/workspace

# Bind this to the VPS Tailscale address when calling Tauji directly over tailnet.
export TAUJI_HOST=127.0.0.1
export TAUJI_PORT=8765

export TAUJI_DATA_DIR=$HOME/.tauji
export TAUJI_MAX_DEPTH=1

# Shell commands run without network access by default. Enable only when the
# selected workspace is allowed to use the VPS network.
export TAUJI_BASH_NETWORK=false
export TAUJI_BWRAP=bwrap
```

Start it:

```bash
uv run tauji
```

The MCP endpoint is:

```text
http://<TAUJI_HOST>:8765/mcp
```

For a Tailscale deployment, bind `TAUJI_HOST` to the VPS tailnet address rather than
`0.0.0.0`.

## MCP surface

Tauji intentionally exposes a small handle-oriented API:

```text
agent_create(workspace, model?, name?, parent_id?) -> agent
agent_list(parent_id?)                             -> agents
agent_get(agent_id)                                -> agent

agent_run(agent_id, prompt)                        -> run
run_get(run_id)                                    -> run
run_list(agent_id, limit=50)                       -> runs
run_cancel(run_id)                                 -> run

agent_send(agent_id, message)                      -> accepted/run_id
agent_children(agent_id)                           -> agents
agent_delete(agent_id)                             -> deleted agent
```

`agent_run` is admission-only. It returns a durable `run_id` immediately. Poll `run_get` or
`run_list` for `running`, `completed`, `failed`, `cancelled`, or `interrupted`.

A server restart converts in-flight runs to `interrupted`; transcripts and handles remain durable.

## Recursive agents

Every agent receives five workspace tools:

```text
read
write
edit
bash
fork
```

`read`, `write`, and `edit` reject paths outside the selected workspace. `bash` runs inside a
mandatory Bubblewrap sandbox with a synthetic home, a scrubbed environment, read-only system
tools, and only the workspace mounted read/write. Host home directories, SSH credentials, runtime
sockets, and unrelated files are not mounted. Network is isolated unless `TAUJI_BASH_NETWORK` is
explicitly enabled. Timeout and cancellation terminate the complete sandbox process group.

`fork(task, name?, model?)` creates an isolated child context in the same workspace and returns
its `agent_id` and `run_id` immediately. On completion the child result is injected into its parent.
If the parent is idle, that delivery starts a continuation; if it is active, it becomes steering.

Root and child agents use the same `AgentRegistry`; there is no separate RLM implementation.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

`uv.lock` pins the tested dependency graph. Update it intentionally with `uv lock --upgrade`.

## License

MIT.
