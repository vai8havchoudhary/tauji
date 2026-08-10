import asyncio
import shutil
from pathlib import Path

import pytest

from tauji.tools import _bash, _bwrap_command, _edit, _path, _read, _write


def test_workspace_file_tools(tmp_path: Path) -> None:
    assert _write(tmp_path, {"path": "a.txt", "content": "one\ntwo\n"}) == "wrote a.txt (8 chars)"
    assert _read(tmp_path, {"path": "a.txt", "start": 2}) == "2: two"
    assert _edit(tmp_path, {"path": "a.txt", "old": "two", "new": "three"}) == "edited a.txt"
    assert (tmp_path / "a.txt").read_text() == "one\nthree\n"


def test_path_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _path(tmp_path, "../outside")


def test_bash_fails_closed_without_bubblewrap(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        _bwrap_command(
            tmp_path,
            "touch must-not-exist",
            bwrap_path="definitely-not-installed-bwrap",
            allow_network=False,
        )
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
async def test_bash_sandbox_hides_host_and_scrubs_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAUJI_TEST_SECRET", "must-not-leak")
    result = await _bash(
        tmp_path,
        {
            "command": (
                "touch inside; "
                "test ! -e /etc/hostname; "
                "test ! -e /home/$USER/.ssh; "
                'test -z "${TAUJI_TEST_SECRET:-}"'
            )
        },
    )
    assert result.startswith("exit=0")
    assert (tmp_path / "inside").is_file()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
async def test_bash_cancellation_kills_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "after-cancel"
    task = asyncio.create_task(
        _bash(tmp_path, {"command": "sleep 1; touch after-cancel", "timeout": 10})
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.2)
    assert not marker.exists()
