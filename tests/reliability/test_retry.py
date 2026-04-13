import time

import pytest

from tradingagents.reliability.retry import (
    RetryPolicy,
    classify_error,
    with_retries,
    FATAL,
    RETRYABLE_RATE_LIMIT,
    RETRYABLE_TRANSIENT,
)
from tests.reliability._fakes import ScriptedLLM


class FakeRateLimit(Exception):
    def __init__(self, retry_after: float | None = None):
        super().__init__("429 too many requests")
        self.retry_after = retry_after
    status_code = 429


class FakeTimeout(Exception):
    pass


class FakeAuthError(Exception):
    status_code = 401


def test_classify_rate_limit():
    assert classify_error(FakeRateLimit()) == RETRYABLE_RATE_LIMIT


def test_classify_timeout():
    assert classify_error(FakeTimeout()) == RETRYABLE_TRANSIENT


def test_classify_auth_error_is_fatal():
    assert classify_error(FakeAuthError()) == FATAL


def test_retries_until_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    llm = ScriptedLLM([FakeTimeout(), FakeTimeout(), "ok"])
    policy = RetryPolicy(max_attempts_transient=4, base_delay_s=1.0, jitter_frac=0.0)

    out = with_retries(lambda: llm.invoke("hi"), policy=policy)
    assert out == "ok"
    assert len(llm.calls) == 3
    assert sleeps == [1.0, 2.0]  # backoff 2^0, 2^1


def test_honors_retry_after_header(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    # Stub out jitter so the sleep matches exactly
    import tradingagents.reliability.retry as retry_mod
    monkeypatch.setattr(retry_mod, "_jitter", lambda x, frac=0.1: x)

    llm = ScriptedLLM([FakeRateLimit(retry_after=17.0), "ok"])
    out = with_retries(lambda: llm.invoke("hi"))
    assert out == "ok"
    assert sleeps == [17.0]


def test_fatal_error_does_not_retry(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    llm = ScriptedLLM([FakeAuthError()])
    with pytest.raises(FakeAuthError):
        with_retries(lambda: llm.invoke("hi"))
    assert len(llm.calls) == 1


def test_exhaustion_raises_last_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    llm = ScriptedLLM([FakeTimeout()] * 10)
    policy = RetryPolicy(max_attempts_transient=3, base_delay_s=0.1)
    with pytest.raises(FakeTimeout):
        with_retries(lambda: llm.invoke("hi"), policy=policy)
    assert len(llm.calls) == 3
