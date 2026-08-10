from pathlib import Path

from tauji.store import Store


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
