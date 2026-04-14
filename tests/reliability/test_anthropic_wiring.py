"""Integration tests proving retry + budgeter are wired into NormalizedChatAnthropic.invoke."""
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.llm_clients.anthropic_client import NormalizedChatAnthropic
from tradingagents.reliability.retry import RetryPolicy
from tradingagents.reliability.budgeter import ContextBudgeter


class _Fake429(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("fake 429")
        self.retry_after = None


def test_normalized_anthropic_retries_on_429(monkeypatch):
    """A 429 raised by the underlying ChatAnthropic invoke is absorbed by with_retries."""
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)

    llm = NormalizedChatAnthropic(
        model="claude-haiku-4-5",
        api_key="dummy",
        retry_policy=RetryPolicy(max_attempts_rate_limit=3, base_delay_s=0.01),
    )

    call_count = {"n": 0}

    def fake_super_invoke(self, input, config=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise _Fake429()
        from langchain_core.messages import AIMessage
        return AIMessage(content="ok")

    # Patch the parent class invoke so our subclass can call super() through it.
    with patch("langchain_anthropic.ChatAnthropic.invoke", new=fake_super_invoke):
        result = llm.invoke([HumanMessage(content="hi")])

    assert call_count["n"] == 2  # one failure, one success
    assert "ok" in str(result)


def test_normalized_anthropic_compresses_history_when_over_ceiling(monkeypatch):
    """When messages exceed the ceiling, ContextBudgeter compresses before invoke."""
    summarizer = MagicMock()
    summarizer.invoke.return_value = MagicMock(content="compressed")
    budgeter = ContextBudgeter(ceiling=200, summarizer_client=summarizer, keep_last_n=1)

    llm = NormalizedChatAnthropic(
        model="claude-haiku-4-5",
        api_key="dummy",
        context_budgeter=budgeter,
    )

    seen_input = []

    def fake_super_invoke(self, input, config=None, **kwargs):
        seen_input.append(input)
        from langchain_core.messages import AIMessage
        return AIMessage(content="ok")

    with patch("langchain_anthropic.ChatAnthropic.invoke", new=fake_super_invoke):
        # Build dict-style messages that estimate well above the 200 ceiling.
        # Using dicts ensures compress_history's `m.get('role', ...)` path works,
        # so the summarizer is actually invoked on the middle turns.
        msgs = [{"role": "system", "content": "x" * 4000}]
        for i in range(6):
            msgs.append({"role": "user", "content": f"turn {i}: " + "y" * 2000})
        llm.invoke(msgs)

    assert summarizer.invoke.called, "summarizer was never called by the budgeter"
    # The input that reached super().invoke should have fewer messages than what we passed in.
    assert len(seen_input[0]) < len(msgs)


def test_normalized_anthropic_no_retry_no_budgeter_is_passthrough(monkeypatch):
    """Without retry_policy or budgeter, the invoke path is the same as before."""
    llm = NormalizedChatAnthropic(model="claude-haiku-4-5", api_key="dummy")

    seen = []

    def fake_super_invoke(self, input, config=None, **kwargs):
        seen.append(input)
        from langchain_core.messages import AIMessage
        return AIMessage(content="ok")

    with patch("langchain_anthropic.ChatAnthropic.invoke", new=fake_super_invoke):
        llm.invoke([HumanMessage(content="hi")])

    assert len(seen) == 1
