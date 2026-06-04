"""Anthropic Messages adapter for console.x.ai models."""

import asyncio
from typing import Any, AsyncGenerator

import orjson

from app.control.account.enums import FeedbackKind
from app.control.account.invalid_credentials import feedback_kind_for_error
from app.control.account.runtime import get_refresh_service
from app.control.model.registry import resolve as resolve_model
from app.dataplane.account.selector import current_strategy
from app.dataplane.reverse.protocol.xai_console_chat import (
    ConsoleStreamAdapter,
    build_console_payload,
    raise_empty_console_response,
    stream_console_chat,
)
from app.platform.config.snapshot import get_config
from app.platform.errors import RateLimitError, UpstreamError
from app.platform.logging.logger import logger
from app.platform.runtime.clock import now_s
from app.platform.tokens import estimate_prompt_tokens, estimate_tokens
from app.products._account_selection import reserve_account, selection_max_retries
from app.products.openai.chat import _configured_retry_codes, _should_retry_upstream


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"


def _log_task_exception(task: "asyncio.Task") -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.warning("background task failed: task={} error={}", task.get_name(), exc)


async def _quota_sync(token: str, mode_id: int) -> None:
    try:
        svc = get_refresh_service()
        if svc:
            await svc.record_success_async(token, mode_id)
            if current_strategy() == "quota":
                asyncio.create_task(
                    svc.sync_call_quota_async(token, mode_id)
                ).add_done_callback(_log_task_exception)
    except Exception as exc:
        logger.warning(
            "console messages success stats sync failed: token={}... mode_id={} error={}",
            token[:10],
            mode_id,
            exc,
        )


async def _fail_sync(token: str, mode_id: int, exc: BaseException | None = None) -> None:
    try:
        svc = get_refresh_service()
        if svc:
            await svc.record_failure_async(token, mode_id, exc)
    except Exception as e:
        logger.warning(
            "console messages fail sync failed: token={}... mode_id={} error={}",
            token[:10],
            mode_id,
            e,
        )


def _effective_effort(reasoning_effort: str | None, emit_think: bool | None) -> str:
    if reasoning_effort:
        return reasoning_effort
    if emit_think is False:
        return "none"
    return "low"


def _usage_int(data: dict[str, Any] | None, keys: tuple[str, ...], fallback: int) -> int:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
    return fallback


async def create(
    *,
    model: str,
    messages: list[dict],
    stream: bool,
    emit_think: bool,
    temperature: float,
    top_p: float,
    msg_id: str,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
) -> dict | AsyncGenerator[str, None]:
    """Route an Anthropic Messages request through console.x.ai/v1/responses."""
    cfg = get_config()
    spec = resolve_model(model)
    effort = _effective_effort(reasoning_effort, emit_think)
    timeout_s = cfg.get_float("chat.timeout", 120.0)
    max_retries = selection_max_retries()
    retry_codes = _configured_retry_codes(cfg)

    if tools:
        logger.info(
            "console messages tools requested: model={} tool_count={} choice={}",
            model,
            len(tools),
            tool_choice,
        )

    from app.control.account.lifecycle import get_runtime_directory

    directory = await get_runtime_directory()

    if stream:

        async def _run_stream() -> AsyncGenerator[str, None]:
            excluded: list[str] = []
            for attempt in range(max_retries + 1):
                acct, selected_mode_id = await reserve_account(
                    directory,
                    spec,
                    now_s_override=now_s(),
                    exclude_tokens=excluded or None,
                )
                if acct is None:
                    raise RateLimitError("No available accounts for this model tier")

                token = acct.token
                success = False
                retry = False
                fail_exc: BaseException | None = None
                adapter = ConsoleStreamAdapter()

                try:
                    payload = build_console_payload(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_effort=effort,
                        stream=True,
                        tools=tools,
                        tool_choice=tool_choice,
                    )

                    try:
                        yield _sse(
                            "message_start",
                            {
                                "type": "message_start",
                                "message": {
                                    "id": msg_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "model": model,
                                    "content": [],
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": {
                                        "input_tokens": estimate_prompt_tokens(messages),
                                        "output_tokens": 0,
                                    },
                                },
                            },
                        )
                        yield _sse("ping", {"type": "ping"})
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )

                        async for event_type, data in stream_console_chat(
                            token,
                            payload,
                            timeout_s=timeout_s,
                        ):
                            for token_text in adapter.feed(event_type, data):
                                yield _sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": token_text,
                                        },
                                    },
                                )

                        full_text = adapter.full_text
                        if not full_text.strip():
                            raise_empty_console_response(model)
                        output_tokens = _usage_int(
                            adapter.usage,
                            ("output_tokens", "completion_tokens"),
                            estimate_tokens(full_text),
                        )
                        yield _sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": 0},
                        )
                        yield _sse(
                            "message_delta",
                            {
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": "end_turn",
                                    "stop_sequence": None,
                                },
                                "usage": {"output_tokens": output_tokens},
                            },
                        )
                        yield _sse("message_stop", {"type": "message_stop"})
                        yield "data: [DONE]\n\n"
                        success = True
                        logger.info(
                            "console messages stream completed: attempt={}/{} model={} text_len={}",
                            attempt + 1,
                            max_retries + 1,
                            model,
                            len(full_text),
                        )
                    except UpstreamError as exc:
                        fail_exc = exc
                        if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                            retry = True
                            logger.warning(
                                "console messages stream retry scheduled: attempt={}/{} status={} token={}...",
                                attempt + 1,
                                max_retries,
                                exc.status,
                                token[:8],
                            )
                        else:
                            raise

                finally:
                    await directory.release(acct)
                    kind = (
                        FeedbackKind.SUCCESS
                        if success
                        else feedback_kind_for_error(fail_exc)
                        if fail_exc
                        else FeedbackKind.SERVER_ERROR
                    )
                    await directory.feedback(token, kind, selected_mode_id, now_s_val=now_s())
                    if success:
                        await _quota_sync(token, selected_mode_id)
                    else:
                        await _fail_sync(token, selected_mode_id, fail_exc)

                if success or not retry:
                    return
                excluded.append(token)

        return _run_stream()

    excluded: list[str] = []
    for attempt in range(max_retries + 1):
        acct, selected_mode_id = await reserve_account(
            directory,
            spec,
            now_s_override=now_s(),
            exclude_tokens=excluded or None,
        )
        if acct is None:
            raise RateLimitError("No available accounts for this model tier")

        token = acct.token
        success = False
        fail_exc: BaseException | None = None
        adapter = ConsoleStreamAdapter()

        try:
            payload = build_console_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=effort,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
            )

            try:
                async for event_type, data in stream_console_chat(
                    token,
                    payload,
                    timeout_s=timeout_s,
                ):
                    adapter.feed(event_type, data)

                if not adapter.full_text.strip():
                    raise_empty_console_response(model)

                input_tokens = _usage_int(
                    adapter.usage,
                    ("input_tokens", "prompt_tokens"),
                    estimate_prompt_tokens(messages),
                )
                output_tokens = _usage_int(
                    adapter.usage,
                    ("output_tokens", "completion_tokens"),
                    estimate_tokens(adapter.full_text),
                )
                success = True
                logger.info(
                    "console messages completed: attempt={}/{} model={} text_len={}",
                    attempt + 1,
                    max_retries + 1,
                    model,
                    len(adapter.full_text),
                )
                return {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": adapter.full_text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                }

            except UpstreamError as exc:
                fail_exc = exc
                if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                    excluded.append(token)
                    logger.warning(
                        "console messages retry scheduled: attempt={}/{} status={} token={}...",
                        attempt + 1,
                        max_retries,
                        exc.status,
                        token[:8],
                    )
                    continue
                raise

        finally:
            await directory.release(acct)
            kind = (
                FeedbackKind.SUCCESS
                if success
                else feedback_kind_for_error(fail_exc)
                if fail_exc
                else FeedbackKind.SERVER_ERROR
            )
            await directory.feedback(token, kind, selected_mode_id, now_s_val=now_s())
            if success:
                await _quota_sync(token, selected_mode_id)
            else:
                await _fail_sync(token, selected_mode_id, fail_exc)

    raise RateLimitError("No available accounts after retries")


__all__ = ["create"]
