from pathlib import Path

import pytest

from tauji.tools import _edit, _path, _read, _write


def test_workspace_file_tools(tmp_path: Path) -> None:
    assert _write(tmp_path, {"path": "a.txt", "content": "one\ntwo\n"}) == "wrote a.txt (8 chars)"
    assert _read(tmp_path, {"path": "a.txt", "start": 2}) == "2: two"
    assert _edit(tmp_path, {"path": "a.txt", "old": "two", "new": "three"}) == "edited a.txt"
    assert (tmp_path / "a.txt").read_text() == "one\nthree\n"


def test_path_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _path(tmp_path, "../outside")
