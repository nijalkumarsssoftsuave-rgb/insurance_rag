"""Structured JSON logging with PII-safe formatters."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

from app.config import settings

# Coarse patterns applied to every log record. This is a backstop, not the PII
# control - `app/security/pii.py` masks payloads before they reach a logger.
# The point is that a stray f-string cannot leak an identifier into the logs.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\d{10}\b"), "<phone>"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "<pan>"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "<aadhaar>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "<api-key>"),
)


def _redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def redact_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = _redact(value)
    return event_dict


def _console_renderer() -> structlog.dev.ConsoleRenderer:
    """Colour only when it will actually work.

    structlog raises rather than degrading if asked for colours on Windows
    without colorama, and it is not worth a hard dependency for log prettiness.
    """
    colors = sys.stdout.isatty()
    if colors and sys.platform == "win32":
        try:
            import colorama  # noqa: F401
        except ImportError:
            colors = False
    return structlog.dev.ConsoleRenderer(colors=colors)


def configure_logging(level: str | None = None, *, json_output: bool | None = None) -> None:
    """Idempotent. Safe to call from the API, a worker or a script."""
    log_level = (level or settings.app.log_level).upper()
    as_json = settings.app.is_production if json_output is None else json_output

    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, log_level, logging.INFO)
    )
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,
    ]
    processors.append(structlog.processors.JSONRenderer() if as_json else _console_renderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
