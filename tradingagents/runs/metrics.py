"""End-of-run metrics writer."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.reliability.cost_tracker import CostTracker


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def write_run_metrics(
    *,
    path: Path | str,
    run_id: str,
    ticker: str,
    tier: str,
    duration_s: float,
    final_decision: Optional[str],
    tracker: CostTracker,
    latencies_ms: List[float],
    errors: List[Dict[str, Any]],
) -> None:
    totals = tracker.totals
    cache_hit_ratio = (totals.tokens_cached / totals.tokens_in) if totals.tokens_in else 0.0

    doc = {
        "run_id": run_id,
        "ticker": ticker,
        "tier": tier,
        "duration_s": duration_s,
        "final_decision": final_decision,
        "llm": {
            "calls_total": totals.calls,
            "retries": totals.retries,
            "summarizations": totals.summarizations,
            "tokens_in": totals.tokens_in,
            "tokens_out": totals.tokens_out,
            "cached_in": totals.tokens_cached,
            "cache_hit_ratio": round(cache_hit_ratio, 4),
            "cost_usd": round(totals.cost_usd, 4),
        },
        "latency": {
            "p50_ms": round(_percentile(latencies_ms, 0.5)),
            "p95_ms": round(_percentile(latencies_ms, 0.95)),
            "max_ms": round(max(latencies_ms)) if latencies_ms else 0,
        },
        "by_agent": {
            name: asdict(at) for name, at in tracker.by_agent.items()
        },
        "errors": errors,
    }
    Path(path).write_text(json.dumps(doc, indent=2))
