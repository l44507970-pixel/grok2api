"""OpenAI Responses adapter for console.x.ai models."""

import asyncio
from typing import Any, AsyncGenerator

from app.control.account.enums import FeedbackKind
from app.control.account.invalid_credentials import feedback_kind_for_error
from app.control.account.runtime import get_refresh_service
from app.control.model.registry import resolve as resolve_model
from app.dataplane.account.selector import current_strategy
from app.dataplane.reverse.protocol.xai_console_chat import (
    ConsoleStreamAdapter,
    build_console_payload,
    stream_console_chat,
)
from app.platform.config.snapshot import get_config
from app.platform.errors import RateLimitError, UpstreamError
from app.platform.logging.logger import logger
from app.platform.runtime.clock import now_s
from app.platform.tokens import estimate_prompt_tokens, estimate_tokens
from app.products._account_selection import reserve_account, selection_max_retries
from app.products.openai.chat import _configured_retry_codes, _should_retry_upstream

from ._format import build_resp_usage, format_sse, make_resp_object


def _log_task_exception(task: "asyncio.Task") -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.warning("background task failed: task={} error={}", task.get_name(), exc)


async def _quota_sync(token: str, mode_id: int) -> None:
    try:
        if current_strategy() != "quota":
            return
        svc = get_refresh_service()
        if svc:
            await svc.refresh_call_async(token, mode_id)
    except Exception as exc:
        logger.warning(
            "console responses quota sync failed: token={}... mode_id={} error={}",
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
            "console responses fail sync failed: token={}... mode_id={} error={}",
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


def _response_usage(
    usage_data: dict[str, Any] | None,
    *,
    messages: list[dict],
    text: str,
) -> dict:
    input_tokens = _usage_int(
        usage_data,
        ("input_tokens", "prompt_tokens"),
        estimate_prompt_tokens(messages),
    )
    output_tokens = _usage_int(
        usage_data,
        ("output_tokens", "completion_tokens"),
        estimate_tokens(text),
    )
    return build_resp_usage(input_tokens, output_tokens)


def _message_item(message_id: str, full_text: str) -> dict:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": full_text, "annotations": []}],
    }


async def create(
    *,
    model: str,
    messages: list[dict],
    stream: bool,
    emit_think: bool,
    temperature: float,
    top_p: float,
    response_id: str,
    reasoning_id: str,
    message_id: str,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
) -> dict | AsyncGenerator[str, None]:
    """Route a Responses request through console.x.ai/v1/responses."""
    cfg = get_config()
    spec = resolve_model(model)
    effort = _effective_effort(reasoning_effort, emit_think)
    timeout_s = cfg.get_float("chat.timeout", 120.0)
    max_retries = selection_max_retries()
    retry_codes = _configured_retry_codes(cfg)

    if tools:
        logger.info(
            "console responses function tools ignored: model={} tool_count={} choice={}",
            model,
            len(tools),
            tool_choice,
        )

    from app.dataplane.account import _directory as _acct_dir

    if _acct_dir is None:
        raise RateLimitError("Account directory not initialised")
    directory = _acct_dir

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
                text_buf: list[str] = []

                try:
                    payload = build_console_payload(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_effort=effort,
                        stream=True,
                    )

                    try:
                        yield format_sse(
                            "response.created",
                            {
                                "type": "response.created",
                                "response": make_resp_object(
                                    response_id, model, "in_progress", []
                                ),
                            },
                        )
                        yield format_sse(
                            "response.in_progress",
                            {
                                "type": "response.in_progress",
                                "response": make_resp_object(
                                    response_id, model, "in_progress", []
                                ),
                            },
                        )
                        yield format_sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            },
                        )
                        yield format_sse(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                },
                            },
                        )

                        async for event_type, data in stream_console_chat(
                            token,
                            payload,
                            timeout_s=timeout_s,
                        ):
                            for token_text in adapter.feed(event_type, data):
                                text_buf.append(token_text)
                                yield format_sse(
                                    "response.output_text.delta",
                                    {
                                        "type": "response.output_text.delta",
                                        "item_id": message_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "delta": token_text,
                                    },
                                )

                        full_text = "".join(text_buf)
                        msg_item = _message_item(message_id, full_text)
                        yield format_sse(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": full_text,
                            },
                        )
                        yield format_sse(
                            "response.content_part.done",
                            {
                                "type": "response.content_part.done",
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": msg_item["content"][0],
                            },
                        )
                        yield format_sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": 0,
                                "item": msg_item,
                            },
                        )
                        yield format_sse(
                            "response.completed",
                            {
                                "type": "response.completed",
                                "response": make_resp_object(
                                    response_id,
                                    model,
                                    "completed",
                                    [msg_item],
                                    _response_usage(
                                        adapter.usage,
                                        messages=messages,
                                        text=full_text,
                                    ),
                                ),
                            },
                        )
                        yield "data: [DONE]\n\n"
                        success = True
                        logger.info(
                            "console responses stream completed: attempt={}/{} model={} text_len={} reasoning_id={}",
                            attempt + 1,
                            max_retries + 1,
                            model,
                            len(full_text),
                            reasoning_id,
                        )
                    except UpstreamError as exc:
                        fail_exc = exc
                        if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                            retry = True
                            logger.warning(
                                "console responses stream retry scheduled: attempt={}/{} status={} token={}...",
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
                        asyncio.create_task(
                            _quota_sync(token, selected_mode_id)
                        ).add_done_callback(_log_task_exception)
                    else:
                        asyncio.create_task(
                            _fail_sync(token, selected_mode_id, fail_exc)
                        ).add_done_callback(_log_task_exception)

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
            )

            try:
                async for event_type, data in stream_console_chat(
                    token,
                    payload,
                    timeout_s=timeout_s,
                ):
                    adapter.feed(event_type, data)

                msg_item = _message_item(message_id, adapter.full_text)
                result = make_resp_object(
                    response_id,
                    model,
                    "completed",
                    [msg_item],
                    _response_usage(
                        adapter.usage,
                        messages=messages,
                        text=adapter.full_text,
                    ),
                )
                success = True
                logger.info(
                    "console responses completed: attempt={}/{} model={} text_len={}",
                    attempt + 1,
                    max_retries + 1,
                    model,
                    len(adapter.full_text),
                )
                return result

            except UpstreamError as exc:
                fail_exc = exc
                if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                    excluded.append(token)
                    logger.warning(
                        "console responses retry scheduled: attempt={}/{} status={} token={}...",
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
                asyncio.create_task(_quota_sync(token, selected_mode_id)).add_done_callback(
                    _log_task_exception
                )
            else:
                asyncio.create_task(
                    _fail_sync(token, selected_mode_id, fail_exc)
                ).add_done_callback(_log_task_exception)

    raise RateLimitError("No available accounts after retries")


__all__ = ["create"]
