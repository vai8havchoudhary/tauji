from pathlib import Path

import pytest

from tauji.config import Settings


def test_validate_workspace_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    settings = Settings(
        base_url="http://localhost/v1",
        api_key="x",
        default_model="m",
        data_dir=tmp_path / "data",
        workspace_roots=(root.resolve(),),
        host="127.0.0.1",
        port=8765,
        max_depth=1,
        request_timeout=30,
    )
    assert settings.validate_workspace(str(root)) == root.resolve()
    with pytest.raises(ValueError, match="outside"):
        settings.validate_workspace(str(outside))
