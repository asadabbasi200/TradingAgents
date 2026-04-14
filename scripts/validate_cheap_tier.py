"""Real-API validation of cheap-tier sub-project #1 changes.

Usage:
    source .venv/bin/activate
    python scripts/validate_cheap_tier.py

Reads ANTHROPIC_API_KEY from .env, runs a single propagation against the
configured ticker/date in the cheap tier, and prints the resulting metrics
so we can verify each spec success criterion against real-world numbers.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load .env (the project lists python-dotenv as a transitive dep)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set in environment or .env")
    sys.exit(1)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runs.orchestrator import start_run, finish_run
from tradingagents.runs.manifest import RunStatus
from tradingagents.reliability.cost_tracker import CostTracker

TICKER = "AAPL"
TRADE_DATE = "2026-04-10"
RESULTS_ROOT = Path("results")

config = DEFAULT_CONFIG.copy()
config["tier"] = "cheap"

ctx = start_run(
    ticker=TICKER,
    trade_date=TRADE_DATE,
    tier="cheap",
    config_snapshot={k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, type(None)))},
    results_root=RESULTS_ROOT,
)
tracker = CostTracker(emitter=ctx.emitter)
print(f"Started run: {ctx.run_id}")
print(f"Run dir:     {ctx.run_dir}")
print(f"Ticker:      {TICKER}")
print(f"Date:        {TRADE_DATE}")
print(f"Tier:        cheap")
print()
print("Propagating (this hits the real Anthropic API)...")

start_time = time.time()
try:
    graph = TradingAgentsGraph(
        debug=False,
        config=config,
        callbacks=[tracker],
        run_id=ctx.run_id,
    )
    _, decision = graph.propagate(TICKER, TRADE_DATE)
    duration = time.time() - start_time
    print(f"\nPropagate completed in {duration:.1f}s")
    print(f"Decision summary: {str(decision)[:300]}")
    finish_run(
        ctx,
        status=RunStatus.COMPLETED,
        tracker=tracker,
        latencies_ms=[],
        final_decision=str(decision)[:80],
        duration_s=duration,
    )
except KeyboardInterrupt:
    duration = time.time() - start_time
    print(f"\nInterrupted after {duration:.1f}s")
    finish_run(ctx, status=RunStatus.INTERRUPTED, tracker=tracker, duration_s=duration)
    raise
except Exception as e:
    duration = time.time() - start_time
    print(f"\nFAILED after {duration:.1f}s: {type(e).__name__}: {str(e)[:400]}")
    finish_run(
        ctx,
        status=RunStatus.FAILED,
        tracker=tracker,
        error=f"{type(e).__name__}: {str(e)[:300]}",
        duration_s=duration,
    )
    raise

# Print metrics
metrics_path = ctx.run_dir / "run_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    print()
    print("=" * 60)
    print(f"Run metrics: {ctx.run_id}")
    print("=" * 60)
    print(f"Status:           {metrics.get('final_decision', 'n/a')[:80]}")
    print(f"Duration:         {metrics['duration_s']:.1f}s")
    print(f"Cost:             ${metrics['llm']['cost_usd']:.4f}")
    print(f"Calls:            {metrics['llm']['calls_total']}")
    print(f"Tokens in:        {metrics['llm']['tokens_in']:,}")
    print(f"Tokens out:       {metrics['llm']['tokens_out']:,}")
    print(f"Cached in:        {metrics['llm']['cached_in']:,}")
    print(f"Cache hit ratio:  {metrics['llm']['cache_hit_ratio']*100:.1f}%")
    print(f"Retries:          {metrics['llm']['retries']}")
    print(f"Summarizations:   {metrics['llm']['summarizations']}")
    if metrics.get("by_agent"):
        print()
        print("Per-agent breakdown:")
        for agent, totals in sorted(metrics["by_agent"].items(), key=lambda x: -x[1]["cost_usd"]):
            print(f"  {agent:30s}  ${totals['cost_usd']:.4f}  ({totals['calls']} calls, {totals['tokens_in']:,} in)")
    print()
    print(f"Artifacts in {ctx.run_dir}:")
    for p in sorted(ctx.run_dir.iterdir()):
        size = p.stat().st_size if p.is_file() else "(dir)"
        print(f"  {p.name}  {size}")
else:
    print("WARNING: run_metrics.json was not produced")
    sys.exit(2)
