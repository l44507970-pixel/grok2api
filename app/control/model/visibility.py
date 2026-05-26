"""Model visibility policy for public API surfaces."""

from app.control.model.spec import ModelSpec
from app.platform.config.snapshot import get_config
from app.platform.runtime.environment import is_serverless_runtime


# Serverless deployments should avoid exposing the whole legacy grok.com model
# surface, but a few non-console models are still useful and known to work.
_DEFAULT_SERVERLESS_MODEL_ALLOWLIST = frozenset(
    {
        "grok-imagine-image-lite",
        "grok-4.3-console",
        "grok-4.3-low",
        "grok-4.3-medium",
        "grok-4.3-high",
        "grok-4.20-0309-console",
        "grok-4.20-0309-non-reasoning-console",
        "grok-4.20-multi-agent-console",
        "grok-4.20-multi-agent-low",
        "grok-4.20-multi-agent-medium",
        "grok-4.20-multi-agent-high",
        "grok-4.20-multi-agent-xhigh",
    }
)


def console_only_enabled() -> bool:
    """Return whether public endpoints should expose only console models."""
    if get_config("models.console_only", False):
        return True
    if not is_serverless_runtime():
        return False
    return bool(get_config("models.serverless_console_only", True))


def stable_console_only_enabled() -> bool:
    """Return whether public endpoints should hide experimental console aliases."""
    if not console_only_enabled():
        return False
    if not is_serverless_runtime():
        return bool(get_config("models.stable_console_only", False))
    return bool(get_config("models.serverless_stable_console_only", True))


def _serverless_model_allowlist() -> frozenset[str]:
    configured = get_config("models.serverless_model_allowlist", None)
    if not configured:
        configured = get_config("models.serverless_console_allowlist", None)
    if not configured:
        return _DEFAULT_SERVERLESS_MODEL_ALLOWLIST
    if isinstance(configured, list):
        values = configured
    else:
        values = str(configured).split(",")
    allowed = frozenset(str(item).strip() for item in values if str(item).strip())
    return allowed or _DEFAULT_SERVERLESS_MODEL_ALLOWLIST


def is_public(spec: ModelSpec) -> bool:
    """Return whether *spec* should be exposed and accepted by public APIs."""
    if not spec.enabled:
        return False
    if not console_only_enabled():
        return True
    allowlist = _serverless_model_allowlist()
    if not spec.is_console_chat():
        return spec.model_name in allowlist
    if stable_console_only_enabled():
        return spec.model_name in allowlist
    return True


def list_public(models: list[ModelSpec] | tuple[ModelSpec, ...]) -> list[ModelSpec]:
    """Filter registered models through the current visibility policy."""
    return [spec for spec in models if is_public(spec)]


__all__ = [
    "console_only_enabled",
    "is_public",
    "list_public",
    "stable_console_only_enabled",
]
