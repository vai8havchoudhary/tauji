from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from tauji.transcript import AgentMessage

_MESSAGE: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)
_AGENT_FIELDS = ("id", "name", "parent_id", "workspace", "model", "depth", "status")
RunStatus = Literal["running", "completed", "failed", "cancelled", "interrupted"]


class Store:
    """Small durable ledger for explicit MCP agent/run handles."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_id TEXT REFERENCES agents(id) ON DELETE CASCADE,
                    workspace TEXT NOT NULL,
                    model TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS agents_parent_idx ON agents(parent_id);

                CREATE TABLE IF NOT EXISTS messages (
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(agent_id, seq)
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS runs_agent_idx ON runs(agent_id, created_at);
                """
            )
            self._db.execute(
                "UPDATE runs SET status='interrupted', "
                "error=COALESCE(error,'tauji restarted'), updated_at=CURRENT_TIMESTAMP "
                "WHERE status='running'"
            )
            self._db.execute("UPDATE agents SET status='idle' WHERE status='running'")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def upsert_agent(self, row: dict[str, Any]) -> None:
        values = {key: row[key] for key in _AGENT_FIELDS}
        with self._lock:
            self._db.execute(
                """
                INSERT INTO agents(id,name,parent_id,workspace,model,depth,status)
                VALUES(:id,:name,:parent_id,:workspace,:model,:depth,:status)
                ON CONFLICT(id) DO UPDATE SET
                  name=excluded.name,
                  parent_id=excluded.parent_id,
                  workspace=excluded.workspace,
                  model=excluded.model,
                  depth=excluded.depth,
                  status=excluded.status
                """,
                values,
            )
            self._db.commit()

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM agents WHERE id=?", (agent_id,))

    def list_agents(self, parent_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if parent_id is None:
                rows = self._db.execute(
                    "SELECT * FROM agents WHERE parent_id IS NULL ORDER BY created_at, id"
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM agents WHERE parent_id=? ORDER BY created_at, id",
                    (parent_id,),
                ).fetchall()
            return [dict(row) for row in rows]

    def save_messages(self, agent_id: str, messages: tuple[AgentMessage, ...]) -> None:
        with self._lock:
            self._replace_messages(agent_id, messages)
            self._db.commit()

    def load_messages(self, agent_id: str) -> list[AgentMessage]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM messages WHERE agent_id=? ORDER BY seq",
                (agent_id,),
            ).fetchall()
        return [_MESSAGE.validate_json(row["payload"]) for row in rows]

    def create_run(
        self,
        run_id: str,
        agent_id: str,
        prompt: str,
        *,
        messages: tuple[AgentMessage, ...] | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO runs(id,agent_id,prompt,status) VALUES(?,?,?,'running')",
                (run_id, agent_id, prompt),
            )
            if messages is not None:
                self._replace_messages(agent_id, messages)
            self._db.commit()

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._db.execute(
                "UPDATE runs SET status=?,result=?,error=?,updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status='running'",
                (status, result, error, run_id),
            )
            self._db.commit()
            return cursor.rowcount == 1

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM runs WHERE id=?", (run_id,))

    def list_runs(self, agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM runs WHERE agent_id=? ORDER BY rowid DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_agent(self, agent_id: str) -> None:
        """Delete one already-drained agent, compatible with pre-FK Tauji databases."""
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE agent_id=?", (agent_id,))
            self._db.execute("DELETE FROM runs WHERE agent_id=?", (agent_id,))
            self._db.execute("DELETE FROM agents WHERE id=?", (agent_id,))
            self._db.commit()

    def _one(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(sql, params).fetchone()
            return dict(row) if row else None

    def _replace_messages(self, agent_id: str, messages: tuple[AgentMessage, ...]) -> None:
        self._db.execute("DELETE FROM messages WHERE agent_id=?", (agent_id,))
        self._db.executemany(
            "INSERT INTO messages(agent_id,seq,payload) VALUES(?,?,?)",
            (
                (agent_id, index, message.model_dump_json(by_alias=False))
                for index, message in enumerate(messages)
            ),
        )
