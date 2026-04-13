from tradingagents.runs.status_panel import render_panel_text
from tradingagents.reliability.cost_tracker import RunTotals


def test_panel_renders_key_fields():
    totals = RunTotals(calls=37, tokens_in=837_800, tokens_out=88_700, tokens_cached=610_000, cost_usd=1.42)
    text = render_panel_text(
        run_id="2026-04-13_AAPL_a3f2", ticker="AAPL", tier="balanced",
        node="Conservative Analyst", elapsed_s=763,
        totals=totals, cache_hit_ratio=0.74, budget=3.0, last_error=None,
    )
    assert "AAPL" in text
    assert "tier=balanced" in text
    assert "Conservative Analyst" in text
    assert "$1.42" in text
    assert "$3.00" in text
    assert "74%" in text
