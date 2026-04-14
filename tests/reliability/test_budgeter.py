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


def test_compress_history_handles_base_message_objects():
    """compress_history must accept LangChain BaseMessage objects, not just dicts."""
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage, AIMessage
    from tradingagents.reliability.budgeter import compress_history

    summarizer = MagicMock()
    summarizer.invoke.return_value = MagicMock(content="compressed summary")

    msgs = [
        HumanMessage(content="user turn 1"),
        AIMessage(content="assistant reply 1"),
        HumanMessage(content="user turn 2"),
    ]
    out = compress_history(summarizer, msgs)
    assert out == "compressed summary"
    summarizer.invoke.assert_called_once()
    # The joined prompt should contain content from each message
    prompt_arg = summarizer.invoke.call_args[0][0]
    assert "user turn 1" in prompt_arg
    assert "assistant reply 1" in prompt_arg


def test_budgeter_with_base_messages_actually_compresses():
    """End-to-end: ContextBudgeter with BaseMessage input + working summarizer compresses (not truncates)."""
    from unittest.mock import MagicMock
    from langchain_core.messages import SystemMessage, HumanMessage
    from tradingagents.reliability.budgeter import ContextBudgeter

    summarizer = MagicMock()
    summarizer.invoke.return_value = MagicMock(content="COMPRESSED")
    b = ContextBudgeter(ceiling=200, summarizer_client=summarizer, keep_last_n=1)

    msgs = [SystemMessage(content="x" * 4000)]
    for i in range(6):
        msgs.append(HumanMessage(content=f"turn {i}: " + "y" * 2000))

    out = b.ensure_under_ceiling(msgs)

    summarizer.invoke.assert_called_once()  # MUST be called, not fall back to truncation
    # The output should contain a summary message with COMPRESSED in it
    summary_messages = [m for m in out if isinstance(m, dict) and "COMPRESSED" in m.get("content", "")]
    assert len(summary_messages) == 1
