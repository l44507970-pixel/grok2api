"""OpenAI Chat Completions adapter for console.x.ai models."""

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

from ._format import build_usage, make_chat_response, make_response_id, make_stream_chunk, SSE_HEARTBEAT


def _log_task_exception(task: "asyncio.Task") -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.warning("background task failed: task={} error={}", task.get_name(), exc)


async def _quota_sync(token: str, mode_id: int) -> None:
    """持久化成功统计，并在配额模式下顺手刷新真实额度。"""
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
            "console chat success stats sync failed: token={}... mode_id={} error={}",
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
            "console chat fail sync failed: token={}... mode_id={} error={}",
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


def _build_chat_usage(
    usage_data: dict[str, Any] | None,
    *,
    messages: list[dict],
    text: str,
) -> dict:
    prompt_tokens = _usage_int(
        usage_data,
        ("input_tokens", "prompt_tokens"),
        estimate_prompt_tokens(messages),
    )
    completion_tokens = _usage_int(
        usage_data,
        ("output_tokens", "completion_tokens"),
        estimate_tokens(text),
    )
    return build_usage(prompt_tokens, completion_tokens)


async def completions(
    *,
    model: str,
    messages: list[dict],
    stream: bool = True,
    emit_think: bool | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
) -> dict | AsyncGenerator[str, None]:
    """Route a Chat Completions request through console.x.ai/v1/responses."""
    cfg = get_config()
    spec = resolve_model(model)
    effort = _effective_effort(reasoning_effort, emit_think)
    timeout_s = cfg.get_float("chat.timeout", 120.0)
    max_retries = selection_max_retries()
    retry_codes = _configured_retry_codes(cfg)
    response_id = make_response_id()

    if tools:
        logger.info(
            "console chat tools requested: model={} tool_count={} choice={}",
            model,
            len(tools),
            tool_choice,
        )

    logger.info(
        "console chat request accepted: model={} stream={} message_count={}",
        model,
        stream,
        len(messages),
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
                        response_format=response_format,
                    )

                    try:
                        # See chat.py: leading SSE comment heartbeat keeps the
                        # connection alive during the upstream "thinking" phase.
                        yield SSE_HEARTBEAT
                        async for event_type, data in stream_console_chat(
                            token,
                            payload,
                            timeout_s=timeout_s,
                        ):
                            for token_text in adapter.feed(event_type, data):
                                chunk = make_stream_chunk(
                                    response_id,
                                    model,
                                    token_text,
                                )
                                yield f"data: {orjson.dumps(chunk).decode()}\n\n"

                        if not adapter.full_text.strip():
                            raise_empty_console_response(model)

                        final = make_stream_chunk(
                            response_id,
                            model,
                            "",
                            is_final=True,
                            usage=_build_chat_usage(
                                adapter.usage,
                                messages=messages,
                                text=adapter.full_text,
                            ),
                        )
                        yield f"data: {orjson.dumps(final).decode()}\n\n"
                        yield "data: [DONE]\n\n"
                        success = True
                        logger.info(
                            "console chat stream completed: attempt={}/{} model={} text_len={}",
                            attempt + 1,
                            max_retries + 1,
                            model,
                            len(adapter.full_text),
                        )
                    except UpstreamError as exc:
                        fail_exc = exc
                        if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                            retry = True
                            logger.warning(
                                "console chat stream retry scheduled: attempt={}/{} status={} token={}...",
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
                response_format=response_format,
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

                usage = _build_chat_usage(
                    adapter.usage,
                    messages=messages,
                    text=adapter.full_text,
                )
                success = True
                logger.info(
                    "console chat completed: attempt={}/{} model={} text_len={}",
                    attempt + 1,
                    max_retries + 1,
                    model,
                    len(adapter.full_text),
                )
                return make_chat_response(
                    model,
                    adapter.full_text,
                    prompt_content=messages,
                    response_id=response_id,
                    usage=usage,
                )

            except UpstreamError as exc:
                fail_exc = exc
                if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                    excluded.append(token)
                    logger.warning(
                        "console chat retry scheduled: attempt={}/{} status={} token={}...",
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


__all__ = ["completions"]
