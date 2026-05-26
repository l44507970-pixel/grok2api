"""Runtime environment detection helpers."""

import os


def is_serverless_runtime() -> bool:
    """Return True for common serverless function runtimes."""
    return bool(
        os.getenv("VERCEL")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        or os.getenv("FUNCTIONS_WORKER_RUNTIME")
        or os.getenv("K_SERVICE")  # Google Cloud Run functions
    )


def is_vercel_runtime() -> bool:
    """Return True when running inside Vercel Functions."""
    return bool(os.getenv("VERCEL"))


__all__ = ["is_serverless_runtime", "is_vercel_runtime"]
