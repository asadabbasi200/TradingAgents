from unittest.mock import MagicMock

from tradingagents.reliability.budgeter import (
    ContextBudgeter,
    estimate_tokens,
    compress_history,
)


def test_estimate_tokens_rough_char_ratio():
    # ~4 chars per token is the rule of thumb we use.
    text = "a" * 4000
    assert 900 <= estimate_tokens(text) <= 1100


def test_budgeter_under_ceiling_passes_through():
    b = ContextBudgeter(ceiling=10_000, summarizer_client=None)
    msgs = [{"role": "system", "content": "short"},
            {"role": "user", "content": "hi"}]
    out = b.ensure_under_ceiling(msgs)
    assert out == msgs  # unchanged


def test_budgeter_over_ceiling_triggers_compression():
    # Build a history above the ceiling with a clear structure:
    # long system message + 6 debate turns of growing size
    msgs = [{"role": "system", "content": "x" * 40_000}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"turn {i}: " + "y" * 5000})

    stub_summary = "Prior debate compressed to 3 key points."
    summarizer = MagicMock()
    summarizer.invoke.return_value = MagicMock(content=stub_summary)

    b = ContextBudgeter(ceiling=15_000, summarizer_client=summarizer, keep_last_n=2)
    out = b.ensure_under_ceiling(msgs)

    # System message is preserved
    assert out[0]["content"].startswith("x" * 100)
    # A summary message replaces the middle turns
    assert any(stub_summary in (m["content"] if isinstance(m["content"], str) else "")
               for m in out)
    # The last 2 turns are kept verbatim
    assert out[-2]["content"].startswith("turn 4:")
    assert out[-1]["content"].startswith("turn 5:")
    summarizer.invoke.assert_called_once()


def test_summarization_failure_falls_back_to_truncation():
    msgs = [{"role": "system", "content": "x" * 40_000}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"turn {i}: " + "y" * 5000})

    summarizer = MagicMock()
    summarizer.invoke.side_effect = RuntimeError("summarizer down")
    b = ContextBudgeter(ceiling=15_000, summarizer_client=summarizer, keep_last_n=2)
    out = b.ensure_under_ceiling(msgs)
    # Fallback: hard truncation to system + last 2 turns, no summary inserted.
    assert len(out) == 3
    assert out[-1]["content"].startswith("turn 5:")
