"""Shared account selection helpers for products-layer request handlers."""

import asyncio

from app.control.model.enums import ModeId
from app.control.model.spec import ModelSpec
from app.control.account.runtime import get_refresh_service
from app.dataplane.account.selector import current_strategy
from app.platform.config.snapshot import get_config
from app.platform.logging.logger import logger
from app.platform.runtime.environment import is_serverless_runtime

# Random strategy has no config key for retry count; it is pinned here so that
# every retry-driven call site (chat / images / video / anthropic) sees the same
# value without introducing scattered magic numbers.
_RANDOM_MAX_RETRIES = 5
_refresh_task: asyncio.Task | None = None


def _log_refresh_task(task: "asyncio.Task") -> None:
    global _refresh_task
    if task is _refresh_task:
        _refresh_task = None
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("account on-demand refresh failed: error={}", exc)


def selection_max_retries() -> int:
    """Retry count for account-swap loops, aware of the active selection strategy.

    - ``random`` strategy: fixed at :data:`_RANDOM_MAX_RETRIES` (=5).
    - ``quota`` strategy:  reads ``retry.max_retries`` (default 1), preserving
      the historical behaviour.
    """
    if current_strategy() == "random":
        return _RANDOM_MAX_RETRIES
    return int(get_config("retry.max_retries", 1))


def mode_candidates(spec: ModelSpec) -> tuple[int, ...]:
    """Return mode IDs to try for *spec* in priority order.

    Chat models using ``AUTO`` can optionally fall back to ``FAST`` and then
    ``EXPERT`` when the upstream ``auto`` quota window is exhausted but the
    account still has usable quota in the other chat windows.
    """
    primary = int(spec.mode_id)
    if (
        spec.is_chat()
        and spec.mode_id == ModeId.AUTO
        and get_config("features.auto_chat_mode_fallback", True)
    ):
        return (primary, int(ModeId.FAST), int(ModeId.EXPERT))
    return (primary,)


async def reserve_account(
    directory,
    spec: ModelSpec,
    *,
    exclude_tokens: list[str] | None = None,
    now_s_override: int | None = None,
):
    """Reserve an account and return ``(lease, selected_mode_id)``.

    Returns ``(None, original_mode_id)`` when no account is available. Under the
    random strategy no on-demand refresh fallback is attempted — upstream quota
    data is never probed.
    """
    original_mode_id = int(spec.mode_id)

    async def _try_reserve():
        for candidate_mode_id in mode_candidates(spec):
            lease = await directory.reserve(
                pool_candidates=spec.pool_candidates(),
                mode_id=candidate_mode_id,
                now_s_override=now_s_override,
                exclude_tokens=exclude_tokens,
            )
            if lease is not None:
                return lease, candidate_mode_id
        return None, original_mode_id

    lease, selected_mode_id = await _try_reserve()
    if lease is not None:
        return lease, selected_mode_id

    if current_strategy() == "random":
        return None, original_mode_id

    refresh_svc = get_refresh_service()
    if refresh_svc is not None:
        timeout_s = max(0.0, float(get_config("account.refresh.on_demand_timeout_sec", 3)))
        if is_serverless_runtime() or timeout_s <= 0:
            global _refresh_task
            if _refresh_task is None or _refresh_task.done():
                _refresh_task = asyncio.create_task(
                    refresh_svc.refresh_on_demand(),
                    name="account-refresh-on-demand",
                )
                _refresh_task.add_done_callback(_log_refresh_task)
        else:
            try:
                await asyncio.wait_for(refresh_svc.refresh_on_demand(), timeout=timeout_s)
                lease, selected_mode_id = await _try_reserve()
                if lease is not None:
                    return lease, selected_mode_id
            except asyncio.TimeoutError:
                logger.warning(
                    "account on-demand refresh timed out: timeout_s={}",
                    timeout_s,
                )

    return None, original_mode_id
