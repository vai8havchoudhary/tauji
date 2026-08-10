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
    request_timeout: float
    bwrap_path: str = "bwrap"
    bash_network: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        roots = tuple(
            Path(raw).expanduser().resolve()
            for raw in os.environ.get("TAUJI_WORKSPACE_ROOTS", "/srv:/startup:/home").split(":")
            if raw.strip()
        )
        if not roots:
            raise ValueError("TAUJI_WORKSPACE_ROOTS must contain at least one path")

        port = int(os.environ.get("TAUJI_PORT", "8765"))
        max_depth = int(os.environ.get("TAUJI_MAX_DEPTH", "1"))
        timeout = float(os.environ.get("TAUJI_REQUEST_TIMEOUT", "300"))
        if not 1 <= port <= 65535:
            raise ValueError("TAUJI_PORT must be between 1 and 65535")
        if max_depth < 0:
            raise ValueError("TAUJI_MAX_DEPTH must be >= 0")
        if timeout <= 0:
            raise ValueError("TAUJI_REQUEST_TIMEOUT must be > 0")

        return cls(
            base_url=os.environ.get("CLIPROXY_BASE_URL", "http://127.0.0.1:8317/v1").rstrip("/"),
            api_key=os.environ.get("CLIPROXY_API_KEY", "cliproxy"),
            default_model=os.environ.get("TAUJI_MODEL", "gpt-5.6"),
            data_dir=Path(os.environ.get("TAUJI_DATA_DIR", "~/.tauji")).expanduser().resolve(),
            workspace_roots=roots,
            host=os.environ.get("TAUJI_HOST", "127.0.0.1"),
            port=port,
            max_depth=max_depth,
            request_timeout=timeout,
            bwrap_path=os.environ.get("TAUJI_BWRAP", "bwrap"),
            bash_network=_env_bool("TAUJI_BASH_NETWORK", default=False),
        )

    def validate_workspace(self, workspace: str) -> Path:
        path = Path(workspace).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"workspace does not exist or is not a directory: {path}")
        if not any(path == root or path.is_relative_to(root) for root in self.workspace_roots):
            allowed = ", ".join(str(root) for root in self.workspace_roots)
            raise ValueError(f"workspace {path} is outside TAUJI_WORKSPACE_ROOTS ({allowed})")
        return path


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")
