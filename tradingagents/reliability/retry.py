"""Retry/backoff wrapper for LLM calls."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar


log = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorClass(str, Enum):
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"
    RETRYABLE_TRANSIENT = "retryable_transient"
    FATAL = "fatal"


FATAL = ErrorClass.FATAL
RETRYABLE_RATE_LIMIT = ErrorClass.RETRYABLE_RATE_LIMIT
RETRYABLE_TRANSIENT = ErrorClass.RETRYABLE_TRANSIENT


@dataclass
class RetryPolicy:
    max_attempts_rate_limit: int = 5
    max_attempts_transient: int = 4
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    jitter_frac: float = 0.1  # +/- 10%


def classify_error(err: BaseException) -> ErrorClass:
    status = getattr(err, "status_code", None) or getattr(err, "http_status", None)
    if status == 429:
        return RETRYABLE_RATE_LIMIT
    if status is not None and 500 <= status < 600:
        return RETRYABLE_TRANSIENT
    if status in (400, 401, 403, 404, 422):
        return FATAL

    name = type(err).__name__.lower()
    if "ratelimit" in name:
        return RETRYABLE_RATE_LIMIT
    if any(tok in name for tok in ("timeout", "connection", "transient", "internalserver")):
        return RETRYABLE_TRANSIENT
    if any(tok in name for tok in ("authentication", "permission", "badrequest", "validation")):
        return FATAL
    return FATAL  # unknown is fatal by design


def _jitter(delay: float, frac: float = 0.1) -> float:
    if frac <= 0:
        return delay
    return delay * (1 + random.uniform(-frac, frac))


def _sleep_for(err: BaseException, attempt: int, policy: RetryPolicy) -> float:
    retry_after = getattr(err, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    raw = min(policy.base_delay_s * (2 ** (attempt - 1)), policy.max_delay_s)
    return _jitter(raw, policy.jitter_frac)


def with_retries(
    fn: Callable[[], T],
    *,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
) -> T:
    """Invoke fn with exponential-backoff retries on transient errors.

    `on_retry(err, attempt, sleep_s)` is called before sleeping, once per retry.
    Fatal errors are not retried and are re-raised immediately.
    """
    policy = policy or RetryPolicy()
    attempt = 0

    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as err:  # noqa: BLE001 — we re-raise below
            cls = classify_error(err)
            if cls is FATAL:
                raise
            max_attempts = (
                policy.max_attempts_rate_limit
                if cls is RETRYABLE_RATE_LIMIT
                else policy.max_attempts_transient
            )
            if attempt >= max_attempts:
                raise
            sleep_s = _sleep_for(err, attempt, policy)
            if on_retry is not None:
                on_retry(err, attempt, sleep_s)
            log.warning("retry attempt=%d cls=%s sleep=%.1fs err=%s", attempt, cls.value, sleep_s, err)
            time.sleep(sleep_s)
