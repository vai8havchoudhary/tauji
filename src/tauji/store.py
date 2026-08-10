from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from tau_agent.messages import AgentMessage

_MESSAGE = TypeAdapter(AgentMessage)


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_id TEXT,
                    workspace TEXT NOT NULL,
                    model TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    agent_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(agent_id, seq)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._db.execute(
                "UPDATE runs SET status='interrupted', error=COALESCE(error,'tauji restarted') WHERE status='running'"
            )
            self._db.execute("UPDATE agents SET status='idle' WHERE status='running'")
            self._db.commit()

    def upsert_agent(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO agents(id,name,parent_id,workspace,model,depth,status)
                VALUES(:id,:name,:parent_id,:workspace,:model,:depth,:status)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name,parent_id=excluded.parent_id,
                workspace=excluded.workspace,model=excluded.model,depth=excluded.depth,status=excluded.status""",
                row,
            )
            self._db.commit()

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM agents WHERE id=?", (agent_id,)).fetchone()
            return dict(row) if row else None

    def list_children(self, parent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM agents WHERE parent_id=? ORDER BY created_at", (parent_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def save_messages(self, agent_id: str, messages: tuple[AgentMessage, ...]) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE agent_id=?", (agent_id,))
            self._db.executemany(
                "INSERT INTO messages(agent_id,seq,payload) VALUES(?,?,?)",
                [(agent_id, i, m.model_dump_json(by_alias=False)) for i, m in enumerate(messages)],
            )
            self._db.commit()

    def load_messages(self, agent_id: str) -> list[AgentMessage]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM messages WHERE agent_id=? ORDER BY seq", (agent_id,)
            ).fetchall()
        return [_MESSAGE.validate_json(r["payload"]) for r in rows]

    def create_run(self, run_id: str, agent_id: str, prompt: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO runs(id,agent_id,prompt,status) VALUES(?,?,?,'running')",
                (run_id, agent_id, prompt),
            )
            self._db.commit()

    def finish_run(self, run_id: str, status: str, result: str | None = None, error: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE runs SET status=?,result=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (status, result, error, run_id),
            )
            self._db.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def delete_agent(self, agent_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE agent_id=?", (agent_id,))
            self._db.execute("DELETE FROM agents WHERE id=?", (agent_id,))
            self._db.commit()
