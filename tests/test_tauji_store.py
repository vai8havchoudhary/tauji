from pathlib import Path

from tauji.store import Store


def test_run_restart_marks_running_interrupted(tmp_path: Path) -> None:
    db = tmp_path / "tauji.db"
    store = Store(db)
    store.upsert_agent(
        {
            "id": "agt_1",
            "name": "a",
            "parent_id": None,
            "workspace": str(tmp_path),
            "model": "m",
            "depth": 0,
            "status": "running",
        }
    )
    store.create_run("run_1", "agt_1", "x")

    restarted = Store(db)
    assert restarted.get_run("run_1")["status"] == "interrupted"
    assert restarted.get_agent("agt_1")["status"] == "idle"
