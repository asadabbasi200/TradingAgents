"""End-to-end smoke test: dry-run propagation produces the full results layout.

This test exercises the entire stack in dry-run mode:

- ``start_run`` mints a run directory and manifest.
- ``TradingAgentsGraph`` is built with ``tier="cheap"`` and ``dry_run=True``;
  every LLM call is served by ``DryRunLLM`` (zero real network traffic).
- ``propagate`` runs the LangGraph end-to-end. The stub response contains
  ``FINAL TRANSACTION PROPOSAL: **HOLD**`` so the graph terminates cleanly.
- ``finish_run`` writes the final manifest state plus ``run_metrics.json``.

The test asserts the three artifact files exist and contain the expected
skeleton. It is allowed to be non-trivial (it spins up the real graph) but
MUST NOT make any real LLM or external data API calls.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_dry_run_propagation_produces_layout(tmp_path, monkeypatch):
    # Avoid env failures even though we never hit the API in dry-run.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.reliability.cost_tracker import CostTracker
    from tradingagents.runs.manifest import RunStatus
    from tradingagents.runs.orchestrator import finish_run, start_run

    config = DEFAULT_CONFIG.copy()
    config["tier"] = "cheap"
    config["dry_run"] = True
    config["results_dir"] = str(tmp_path / "results-legacy")

    ctx = start_run(
        ticker="AAPL",
        trade_date="2026-04-10",
        tier="cheap",
        config_snapshot={"tier": "cheap", "dry_run": True},
        results_root=tmp_path / "results",
    )
    tracker = CostTracker(emitter=ctx.emitter)

    decision_str: str | None = None
    propagate_error: str | None = None
    try:
        graph = TradingAgentsGraph(
            debug=False,
            config=config,
            callbacks=[tracker],
            run_id=ctx.run_id,
        )
        _, decision = graph.propagate("AAPL", "2026-04-10")
        decision_str = str(decision)[:120]
    except Exception as exc:  # noqa: BLE001 — still verify orchestrator artifacts
        propagate_error = f"{type(exc).__name__}: {exc}"
    finally:
        finish_run(
            ctx,
            status=(
                RunStatus.FAILED if propagate_error else RunStatus.COMPLETED
            ),
            tracker=tracker,
            latencies_ms=[100.0, 200.0, 300.0],
            final_decision=decision_str,
            error=propagate_error,
        )

    # The point of the smoke test: the three artifact files MUST exist.
    run_dir: Path = tmp_path / "results" / ctx.run_id
    manifest_path = run_dir / "run_manifest.json"
    events_path = run_dir / "events.jsonl"
    metrics_path = run_dir / "run_metrics.json"

    assert manifest_path.exists(), f"missing run_manifest.json in {run_dir}"
    assert events_path.exists(), f"missing events.jsonl in {run_dir}"
    assert metrics_path.exists(), f"missing run_metrics.json in {run_dir}"

    # Sanity-check: propagation itself must succeed in dry-run mode.
    assert propagate_error is None, (
        f"dry-run propagation should not raise, got: {propagate_error}"
    )
    assert decision_str is not None

    # The stub always emits a FINAL TRANSACTION PROPOSAL so graph termination
    # conditions trip cleanly. The decision surfaces BUY / HOLD / SELL.
    assert any(token in decision_str.upper() for token in ("HOLD", "BUY", "SELL")), (
        f"decision string missing BUY/HOLD/SELL marker: {decision_str!r}"
    )

    # Manifest reflects a completed run.
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == ctx.run_id
    assert manifest["ticker"] == "AAPL"
    assert manifest["tier"] == "cheap"
    assert manifest["status"] == RunStatus.COMPLETED.value

    # events.jsonl has at least a RUN_START and RUN_END record.
    event_lines = [
        json.loads(line) for line in events_path.read_text().splitlines() if line.strip()
    ]
    assert len(event_lines) >= 2
    event_types = {e.get("type") for e in event_lines}
    assert "run_start" in event_types
    assert "run_end" in event_types

    # run_metrics.json has the expected top-level shape.
    metrics = json.loads(metrics_path.read_text())
    assert metrics["run_id"] == ctx.run_id
    assert metrics["ticker"] == "AAPL"
    assert metrics["tier"] == "cheap"
