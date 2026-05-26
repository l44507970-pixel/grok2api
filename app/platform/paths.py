"""Shared runtime paths derived from environment variables."""

import os
from pathlib import Path

from app.platform.runtime.environment import is_serverless_runtime


_ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default).strip() or default
    path = Path(raw)
    if path.is_absolute():
        return path
    if is_serverless_runtime():
        path = Path("/tmp/grok2api") / path
    else:
        path = _ROOT_DIR / path
    return path


def data_dir() -> Path:
    default = "/tmp/grok2api/data" if is_serverless_runtime() else "data"
    return _resolve_env_path("DATA_DIR", default)


def log_dir() -> Path:
    default = "/tmp/grok2api/logs" if is_serverless_runtime() else "logs"
    return _resolve_env_path("LOG_DIR", default)


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)


def log_path(*parts: str) -> Path:
    return log_dir().joinpath(*parts)


__all__ = ["data_dir", "log_dir", "data_path", "log_path"]
