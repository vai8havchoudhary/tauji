from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str
    api_key: str
    default_model: str
    data_dir: Path
    workspace_roots: tuple[Path, ...]
    host: str
    port: int
    max_depth: int

    @classmethod
    def from_env(cls) -> "Settings":
        roots_raw = os.environ.get("TAUJI_WORKSPACE_ROOTS", "/srv:/startup:/home")
        roots = tuple(Path(p).expanduser().resolve() for p in roots_raw.split(":") if p)
        return cls(
            base_url=os.environ.get("CLIPROXY_BASE_URL", "http://127.0.0.1:8317/v1").rstrip("/"),
            api_key=os.environ.get("CLIPROXY_API_KEY", "cliproxy"),
            default_model=os.environ.get("TAUJI_MODEL", "gpt-5.6"),
            data_dir=Path(os.environ.get("TAUJI_DATA_DIR", "~/.tauji")).expanduser().resolve(),
            workspace_roots=roots,
            host=os.environ.get("TAUJI_HOST", "127.0.0.1"),
            port=int(os.environ.get("TAUJI_PORT", "8765")),
            max_depth=int(os.environ.get("TAUJI_MAX_DEPTH", "1")),
        )

    def validate_workspace(self, workspace: str) -> Path:
        path = Path(workspace).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"workspace does not exist or is not a directory: {path}")
        if not any(path == root or path.is_relative_to(root) for root in self.workspace_roots):
            allowed = ", ".join(map(str, self.workspace_roots))
            raise ValueError(f"workspace {path} is outside TAUJI_WORKSPACE_ROOTS ({allowed})")
        return path
