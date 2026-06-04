"""console.x.ai Responses protocol adapter."""

from typing import Any, AsyncGenerator

import orjson

from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
from app.dataplane.reverse.runtime.endpoint_table import CONSOLE_RESPONSES
from app.dataplane.reverse.transport._proxy_feedback import upstream_feedback
from app.platform.errors import UpstreamError
from app.platform.logging.logger import logger


# 对外模型名到 console.x.ai 实际模型名的映射。
CONSOLE_MODELS: dict[str, str] = {
    "grok-4.3-console": "grok-4.3",
    "grok-4.3-low": "grok-4.3",
    "grok-4.3-medium": "grok-4.3",
    "grok-4.3-high": "grok-4.3",
    "grok-4.20-0309-reasoning-console": "grok-4.20-0309-reasoning",
    "grok-4.20-0309-console": "grok-4.20-0309",
    "grok-4.20-0309-non-reasoning-console": "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-low": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-medium": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-high": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-xhigh": "grok-4.20-multi-agent-0309",
    "grok-build-console": "grok-build-0.1",
}

_MODELS_WITH_REASONING_FIELD = frozenset(
    {
        "grok-4.3",
        "grok-4.20-multi-agent-0309",
    }
)

_MODEL_FIXED_EFFORT: dict[str, str] = {
    "grok-4.3-low": "low",
    "grok-4.3-medium": "medium",
    "grok-4.3-high": "high",
    "grok-4.20-multi-agent-low": "low",
    "grok-4.20-multi-agent-medium": "medium",
    "grok-4.20-multi-agent-high": "high",
    "grok-4.20-multi-agent-xhigh": "xhigh",
}

_MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "grok-4.20-multi-agent-0309": 2_000_000,
    "grok-build-0.1": 256_000,
}

_MODELS_WITH_SEARCH_TOOLS = frozenset(
    {
        "grok-4.3",
        "grok-4.20-multi-agent-0309",
        "grok-4.20-0309",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-build-0.1",
    }
)

_EFFORT_MAP: dict[str, str] = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

_WEB_SEARCH_ALIASES = frozenset({"web_search", "web_search_preview"})
_X_SEARCH_ALIASES = frozenset({"x_search", "x_keyword_search", "x_semantic_search"})


def _api_role(role: str) -> str:
    if role in {"system", "developer", "assistant"}:
        return role if role != "developer" else "system"
    return "user"


def _image_url_from_block(block: dict[str, Any]) -> str:
    src = block.get("image_url") or block.get("source") or block.get("url") or ""
    if isinstance(src, dict):
        return str(src.get("url") or "")
    return str(src or "")


def _content_blocks(msg: dict[str, Any]) -> list[dict[str, Any]]:
    content = msg.get("content")
    blocks: list[dict[str, Any]] = []

    if isinstance(content, str):
        if content:
            blocks.append({"type": "input_text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in {"text", "input_text", "output_text"}:
                text = block.get("text") or ""
                if text:
                    blocks.append({"type": "input_text", "text": text})
            elif btype in {"image_url", "input_image", "image"}:
                url = _image_url_from_block(block)
                if url:
                    blocks.append({"type": "input_image", "image_url": url})
            elif btype == "tool_result":
                text = block.get("content") or block.get("text") or ""
                if text:
                    blocks.append({"type": "input_text", "text": str(text)})
            else:
                text = block.get("text")
                if text is None:
                    text = str(block)
                blocks.append({"type": "input_text", "text": str(text)})
    elif content is not None:
        blocks.append({"type": "input_text", "text": str(content)})

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            tool_text = orjson.dumps(tool_calls).decode()
        except Exception:
            tool_text = str(tool_calls)
        blocks.append({"type": "input_text", "text": f"[tool_calls]\n{tool_text}"})

    return blocks


def _default_console_tools() -> list[dict[str, Any]]:
    return [
        {"type": "web_search", "enable_image_understanding": True},
        {"type": "x_search", "enable_video_understanding": True},
    ]


def _normalize_console_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type") or "").strip()
    if tool_type in _WEB_SEARCH_ALIASES:
        normalized = dict(tool)
        normalized["type"] = "web_search"
        normalized.setdefault("enable_image_understanding", True)
        return normalized
    if tool_type in _X_SEARCH_ALIASES:
        normalized = dict(tool)
        normalized["type"] = "x_search"
        normalized.setdefault("enable_video_understanding", True)
        return normalized
    return None


def _has_console_tool(tools: list[dict[str, Any]]) -> bool:
    return any(_normalize_console_tool(tool) is not None for tool in tools)


def _console_tools(
    *,
    console_model: str,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = None,
) -> list[dict[str, Any]]:
    if console_model not in _MODELS_WITH_SEARCH_TOOLS:
        return []
    if tool_choice == "none":
        return []
    if tools is None:
        return _default_console_tools()
    if not tools:
        return []
    enabled = [
        normalized
        for tool in tools
        if (normalized := _normalize_console_tool(tool)) is not None
    ]
    return enabled or _default_console_tools()


def _tool_choice_for_console(
    tool_choice: Any,
    enabled_tools: list[dict[str, Any]],
) -> Any:
    if not enabled_tools:
        return None
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in {"auto", "none", "required"} else "auto"
    if isinstance(tool_choice, dict):
        forced_type = str(tool_choice.get("type") or "").strip()
        if forced_type in _WEB_SEARCH_ALIASES:
            return {"type": "web_search"}
        if forced_type in _X_SEARCH_ALIASES:
            return {"type": "x_search"}
        forced_name = str((tool_choice.get("function") or {}).get("name") or "").strip()
        if forced_name in _WEB_SEARCH_ALIASES:
            return {"type": "web_search"}
        if forced_name in _X_SEARCH_ALIASES:
            return {"type": "x_search"}
    return "auto"


def _normalize_response_format(response_format: Any) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if isinstance(response_format, str):
        fmt_type = response_format.strip()
        return {"type": fmt_type} if fmt_type else None
    if not isinstance(response_format, dict):
        return None

    if "format" in response_format and isinstance(response_format.get("format"), dict):
        return _normalize_response_format(response_format.get("format"))

    fmt_type = str(response_format.get("type") or "").strip()
    if not fmt_type:
        return None

    if fmt_type == "json_schema":
        json_schema = response_format.get("json_schema")
        source = json_schema if isinstance(json_schema, dict) else response_format
        normalized: dict[str, Any] = {"type": "json_schema"}
        normalized["name"] = str(source.get("name") or "response")
        if source.get("description") is not None:
            normalized["description"] = source.get("description")
        normalized["schema"] = source.get("schema") or {}
        if source.get("strict") is not None:
            normalized["strict"] = bool(source.get("strict"))
        return normalized

    normalized = dict(response_format)
    normalized["type"] = fmt_type
    normalized.pop("json_schema", None)
    return normalized


def _console_text_config(
    *,
    response_format: Any = None,
    text: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if isinstance(text, dict):
        config = dict(text)
        normalized = _normalize_response_format(config.get("format"))
        if normalized:
            config["format"] = normalized
        if config:
            return config

    normalized = _normalize_response_format(response_format)
    if normalized:
        return {"format": normalized}
    return None


def build_console_payload(
    *,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    reasoning_effort: str | None = None,
    stream: bool = True,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
    text: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON payload for ``POST console.x.ai/v1/responses``."""
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        blocks = _content_blocks(msg)
        if not blocks:
            continue
        input_items.append(
            {
                "role": _api_role(str(msg.get("role") or "user")),
                "content": blocks,
            }
        )

    console_model = CONSOLE_MODELS.get(model, model)
    effort = _MODEL_FIXED_EFFORT.get(model) or _EFFORT_MAP.get(
        reasoning_effort or "medium", "medium"
    )

    payload: dict[str, Any] = {
        "model": console_model,
        "input": input_items,
        "max_output_tokens": _MODEL_MAX_OUTPUT_TOKENS.get(console_model, 1_000_000),
        "temperature": temperature,
        "top_p": top_p,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "stream": stream,
    }

    if console_model in _MODELS_WITH_REASONING_FIELD:
        payload["reasoning"] = {"effort": effort}

    enabled_tools = _console_tools(
        console_model=console_model,
        tools=tools,
        tool_choice=tool_choice,
    )
    if enabled_tools:
        payload["tools"] = enabled_tools
        effective_tool_choice = (
            None
            if tools is not None and tools and not _has_console_tool(tools)
            else tool_choice
        )
        normalized_tool_choice = _tool_choice_for_console(
            effective_tool_choice,
            enabled_tools,
        )
        if normalized_tool_choice is not None:
            payload["tool_choice"] = normalized_tool_choice

    text_config = _console_text_config(response_format=response_format, text=text)
    if text_config:
        payload["text"] = text_config

    logger.debug(
        "console payload built: model={} console_model={} input_items={} has_reasoning={} tool_count={} has_text_format={}",
        model,
        console_model,
        len(input_items),
        console_model in _MODELS_WITH_REASONING_FIELD,
        len(enabled_tools),
        bool(text_config),
    )
    return payload


class ConsoleStreamAdapter:
    """Parse console.x.ai Responses SSE events into text tokens."""

    __slots__ = ("text_buf", "usage", "_done")

    def __init__(self) -> None:
        self.text_buf: list[str] = []
        self.usage: dict[str, Any] | None = None
        self._done = False

    def _apply_final_text(self, text: Any) -> list[str]:
        if not isinstance(text, str) or not text:
            return []
        current = self.full_text
        if not current:
            self.text_buf = [text]
            return [text]
        if len(text) > len(current):
            emitted = [text[len(current):]] if text.startswith(current) else []
            self.text_buf = [text]
            return [part for part in emitted if part]
        return []

    @staticmethod
    def _content_text(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type not in {"output_text", "text"}:
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)

    @classmethod
    def _output_text(cls, output: Any) -> str:
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            direct = item.get("output_text")
            if isinstance(direct, str) and direct:
                parts.append(direct)
            content_text = cls._content_text(item.get("content"))
            if content_text:
                parts.append(content_text)
        return "".join(parts)

    def feed(self, event_type: str, data: str) -> list[str]:
        if self._done:
            return []
        try:
            obj = orjson.loads(data)
        except (orjson.JSONDecodeError, ValueError):
            return []

        if not event_type:
            event_type = str(obj.get("type") or "")

        if event_type == "response.output_text.delta":
            delta = obj.get("delta") or ""
            if delta:
                self.text_buf.append(delta)
                return [str(delta)]
        elif event_type == "response.output_text.done":
            return self._apply_final_text(obj.get("text"))
        elif event_type == "response.content_part.done":
            part = obj.get("part") if isinstance(obj.get("part"), dict) else {}
            return self._apply_final_text(part.get("text"))
        elif event_type == "response.output_item.done":
            item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
            return self._apply_final_text(self._content_text(item.get("content")))
        elif event_type == "response.completed":
            emitted: list[str] = []
            response = obj.get("response") or {}
            if isinstance(response, dict):
                self.usage = response.get("usage")
                emitted.extend(self._apply_final_text(response.get("output_text")))
                emitted.extend(self._apply_final_text(self._output_text(response.get("output"))))
            self._done = True
            return emitted
        elif event_type == "error":
            msg = obj.get("message") or obj.get("error") or str(obj)
            raise UpstreamError(f"Console API error: {msg}", status=502)

        return []

    @property
    def full_text(self) -> str:
        return "".join(self.text_buf)


def raise_empty_console_response(model: str) -> None:
    raise UpstreamError(
        "Console API completed without output text",
        status=503,
        body=f"reason=empty_output model={model}",
    )


def classify_console_line(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped:
        return "skip", ""
    if stripped.startswith("event:"):
        return "event", stripped[6:].strip()
    if stripped.startswith("data:"):
        data = stripped[5:].strip()
        if data == "[DONE]":
            return "done", ""
        return "data", data
    return "skip", ""


def _transport_feedback() -> ProxyFeedback:
    return ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR)


def _success_feedback() -> ProxyFeedback:
    return ProxyFeedback(kind=ProxyFeedbackKind.SUCCESS, status_code=200)


async def stream_console_chat(
    token: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 120.0,
) -> AsyncGenerator[tuple[str, str], None]:
    """POST to console.x.ai/v1/responses and yield ``(event_type, data)``."""
    from app.dataplane.proxy import get_proxy_runtime
    from app.dataplane.proxy.adapters.headers import build_console_headers
    from app.dataplane.proxy.adapters.session import ResettableSession, build_session_kwargs

    proxy = await get_proxy_runtime()
    # console.x.ai 与 grok.com 共用 SSO/CF 访问态。这里沿用默认 grok.com
    # clearance，与参考项目保持一致，避免单独按 console.x.ai 生成无效 clearance。
    lease = await proxy.acquire()
    headers = build_console_headers(token, lease=lease)
    session_kwargs = build_session_kwargs(lease=lease)

    async with ResettableSession(**session_kwargs) as session:
        try:
            response = await session.post(
                CONSOLE_RESPONSES,
                headers=headers,
                data=orjson.dumps(payload),
                timeout=timeout_s,
                stream=True,
            )
        except UpstreamError as exc:
            await proxy.feedback(lease, upstream_feedback(exc))
            raise
        except Exception as exc:
            await proxy.feedback(lease, _transport_feedback())
            raise UpstreamError(
                f"Console transport failed: {exc}",
                status=502,
                body=str(exc).replace("\n", "\\n")[:400],
            ) from exc

        if response.status_code != 200:
            try:
                body = response.content.decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
            err = UpstreamError(
                f"Console API returned {response.status_code}",
                status=response.status_code,
                body=body,
            )
            await proxy.feedback(lease, upstream_feedback(err))
            raise err

        # 上游已经接受请求，proxy/clearance 已完成它们该完成的部分。
        # high/xhigh 这类长 SSE 流后续可能因平台或客户端中断，不应误伤代理池。
        await proxy.feedback(lease, _success_feedback())

        current_event = ""
        try:
            async for raw_line in response.aiter_lines():
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", "replace")
                kind, value = classify_console_line(str(raw_line))
                if kind == "event":
                    current_event = value
                elif kind == "data":
                    yield current_event, value
                    current_event = ""
                elif kind == "done":
                    return
        except UpstreamError:
            raise
        except Exception as exc:
            raise UpstreamError(
                f"Console stream read failed: {exc}",
                status=502,
                body=str(exc).replace("\n", "\\n")[:400],
            ) from exc


__all__ = [
    "CONSOLE_MODELS",
    "build_console_payload",
    "ConsoleStreamAdapter",
    "raise_empty_console_response",
    "classify_console_line",
    "stream_console_chat",
]
