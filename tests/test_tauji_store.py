import json
import sqlite3
from pathlib import Path

from tauji.store import Store
from tauji.transcript import AssistantMessage


def _agent(agent_id: str = "agt_1", parent_id: str | None = None) -> dict[str, object]:
    return {
        "id": agent_id,
        "name": agent_id,
        "parent_id": parent_id,
        "workspace": "/tmp",
        "model": "test",
        "depth": 0 if parent_id is None else 1,
        "status": "idle",
    }


def test_restart_interrupts_running_work(tmp_path: Path) -> None:
    path = tmp_path / "tauji.db"
    store = Store(path)
    store.upsert_agent(_agent())
    store.create_run("run_1", "agt_1", "hello")
    store.close()

    reopened = Store(path)
    assert reopened.get_run("run_1")["status"] == "interrupted"
    assert reopened.get_agent("agt_1")["status"] == "idle"
    reopened.close()


def test_delete_parent_cascades_children_and_runs(tmp_path: Path) -> None:
    store = Store(tmp_path / "tauji.db")
    store.upsert_agent(_agent())
    store.upsert_agent(_agent("agt_2", "agt_1"))
    store.create_run("run_2", "agt_2", "child")
    store.delete_agent("agt_1")
    assert store.get_agent("agt_2") is None
    assert store.get_run("run_2") is None
    store.close()


def test_first_terminal_run_status_wins(tmp_path: Path) -> None:
    store = Store(tmp_path / "tauji.db")
    store.upsert_agent(_agent())
    store.create_run("run_1", "agt_1", "hello")
    assert store.finish_run("run_1", "completed", result="done") is True
    assert store.finish_run("run_1", "cancelled", error="late cancel") is False
    assert store.get_run("run_1")["status"] == "completed"
    assert store.get_run("run_1")["result"] == "done"
    store.close()


def test_current_verbose_transcript_remains_readable(tmp_path: Path) -> None:
    """Freeze compatibility with transcripts written before the Tau cleanup."""
    path = tmp_path / "tauji.db"
    store = Store(path)
    store.upsert_agent(_agent())
    store.close()

    payload = {
        "role": "assistant",
        "content": [{"type": "text", "text": "PERSIST_OK", "text_signature": None}],
        "api": "openai-chat",
        "provider": "cliproxy",
        "model": "gpt-5.6-terra",
        "response_model": None,
        "response_provider": None,
        "response_id": "response-old",
        "diagnostics": None,
        "usage": {
            "input": 617,
            "output": 7,
            "cache_read": 0,
            "cache_write": 0,
            "cache_write_1h": None,
            "reasoning": None,
            "total_tokens": 624,
            "cost": {"input": 0.0, "output": 0.0, "total": 0.0},
        },
        "stop_reason": "stop",
        "error_message": None,
        "timestamp": 1_786_000_000_000,
    }
    with sqlite3.connect(path) as database:
        database.execute(
            "INSERT INTO messages(agent_id,seq,payload) VALUES(?,?,?)",
            ("agt_1", 0, json.dumps(payload)),
        )

    reopened = Store(path)
    messages = reopened.load_messages("agt_1")
    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)
    assert messages[0].text == "PERSIST_OK"

    reopened.save_messages("agt_1", tuple(messages))
    with sqlite3.connect(path) as database:
        normalized = json.loads(
            database.execute("SELECT payload FROM messages WHERE agent_id='agt_1'").fetchone()[0]
        )
    assert "usage" not in normalized
    assert "provider" not in normalized
    reopened.close()
