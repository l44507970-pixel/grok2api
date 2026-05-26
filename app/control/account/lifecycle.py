"""Lazy account-control runtime initialisation."""

import asyncio
from typing import Any

from app.control.account.runtime import set_refresh_service

_init_lock = asyncio.Lock()
_repository = None
_refresh_service = None


def _state_value(app: Any | None, name: str):
    if app is None:
        return None
    return getattr(app.state, name, None)


def _set_state_value(app: Any | None, name: str, value) -> None:
    if app is not None:
        setattr(app.state, name, value)


async def get_runtime_repository(app: Any | None = None):
    """Return the process repository, initialising it on first use."""
    global _repository, _refresh_service

    repo = _state_value(app, "repository") or _repository
    if repo is not None:
        return repo

    async with _init_lock:
        repo = _state_value(app, "repository") or _repository
        if repo is not None:
            return repo

        from app.control.account.backends.factory import create_repository
        from app.control.account.refresh import AccountRefreshService

        repo = create_repository()
        await repo.initialize()
        _repository = repo
        _set_state_value(app, "repository", repo)

        _refresh_service = AccountRefreshService(repo)
        set_refresh_service(_refresh_service)
        _set_state_value(app, "refresh_service", _refresh_service)
        return repo


async def get_runtime_directory(app: Any | None = None):
    """Return the account directory, bootstrapping it lazily if needed."""
    directory = _state_value(app, "directory")
    if directory is not None:
        return directory

    repo = await get_runtime_repository(app)
    from app.dataplane.account import get_account_directory

    directory = await get_account_directory(repo)
    _set_state_value(app, "directory", directory)
    return directory


async def get_runtime_refresh_service(app: Any | None = None):
    """Return the refresh service, initialising account runtime if needed."""
    svc = _state_value(app, "refresh_service") or _refresh_service
    if svc is not None:
        return svc
    await get_runtime_repository(app)
    return _state_value(app, "refresh_service") or _refresh_service


async def close_runtime_repository(app: Any | None = None) -> None:
    """Close the lazily created repository, if this process owns one."""
    global _repository, _refresh_service

    repo = _repository
    _repository = None
    _refresh_service = None
    set_refresh_service(None)
    if app is not None:
        for name in ("repository", "directory", "refresh_service"):
            if hasattr(app.state, name):
                delattr(app.state, name)
    from app.dataplane.account import reset_account_directory

    reset_account_directory()
    if repo is not None:
        await repo.close()


__all__ = [
    "close_runtime_repository",
    "get_runtime_directory",
    "get_runtime_refresh_service",
    "get_runtime_repository",
]
