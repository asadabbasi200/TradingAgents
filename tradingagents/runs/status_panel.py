"""Live CLI status panel and renderer for runs."""
from __future__ import annotations

from typing import Optional

from tradingagents.reliability.cost_tracker import RunTotals


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def render_panel_text(
    *,
    run_id: str,
    ticker: str,
    tier: str,
    node: str,
    elapsed_s: float,
    totals: RunTotals,
    cache_hit_ratio: float,
    budget: Optional[float],
    last_error: Optional[str],
) -> str:
    lines = []
    lines.append(f"[bold]{ticker}[/bold] · tier={tier} · run={run_id}")
    lines.append(f"Node: {node}    ⏱ {_fmt_elapsed(elapsed_s)}")
    lines.append(
        f"LLM: {totals.calls} calls · {totals.retries} retries · "
        f"{totals.summarizations} summarize"
    )
    lines.append(
        f"Tokens: {_fmt_tokens(totals.tokens_in)}↑ {_fmt_tokens(totals.tokens_out)}↓   "
        f"Cached: {int(cache_hit_ratio * 100)}%"
    )
    budget_str = f" / budget ${budget:.2f}" if budget is not None else ""
    lines.append(f"Cost: ${totals.cost_usd:.2f}{budget_str}")
    lines.append(f"Last error: {last_error or 'none'}")
    return "\n".join(lines)
