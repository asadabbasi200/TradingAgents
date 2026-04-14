import json
from pathlib import Path

from tradingagents.reliability.cost_tracker import CostTracker, AgentTotals, RunTotals
from tradingagents.runs.metrics import write_run_metrics


def test_write_run_metrics_includes_per_agent(tmp_path: Path):
    tracker = CostTracker(emitter=None)
    tracker.totals = RunTotals(
        calls=42, tokens_in=142_300, tokens_out=38_720, tokens_cached=98_400,
        cost_usd=1.83, retries=2, summarizations=1,
    )
    tracker.by_agent["market_analyst"] = AgentTotals(
        calls=3, tokens_in=12_400, cost_usd=0.08,
    )

    write_run_metrics(
        path=tmp_path / "run_metrics.json",
        run_id="r1", ticker="AAPL", tier="balanced",
        duration_s=384, final_decision="BUY",
        tracker=tracker, latencies_ms=[200, 500, 1200, 3000],
        errors=[],
    )

    data = json.loads((tmp_path / "run_metrics.json").read_text())
    assert data["llm"]["calls_total"] == 42
    assert data["llm"]["cost_usd"] == 1.83
    assert data["llm"]["cache_hit_ratio"] > 0
    assert data["by_agent"]["market_analyst"]["calls"] == 3
    assert data["latency"]["p50_ms"] > 0
    assert data["final_decision"] == "BUY"
