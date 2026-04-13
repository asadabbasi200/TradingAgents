"""Run lifecycle: start, resume, finish, list, interrupt handler."""
from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.runs.events import EventEmitter, EventType
from tradingagents.runs.manifest import (
    RunManifest,
    RunStatus,
    TokenTotals,
    mint_run_id,
    read_manifest,
    update_manifest,
    write_manifest,
)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    manifest: RunManifest
    emitter: EventEmitter


def _run_dir(results_root: Path | str, run_id: str) -> Path:
    return Path(results_root) / run_id


def start_run(
    ticker: str,
    trade_date: str,
    tier: str,
    config_snapshot: Dict[str, Any],
    results_root: Path | str = "results",
) -> RunContext:
    run_id = mint_run_id(ticker, trade_date)
    run_dir = _run_dir(results_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    manifest = RunManifest(
        run_id=run_id,
        ticker=ticker,
        trade_date=trade_date,
        tier=tier,
        started_at=now_iso,
        status=RunStatus.RUNNING,
        config_snapshot=config_snapshot,
        tokens=TokenTotals(),
    )
    write_manifest(manifest, run_dir / "run_manifest.json")
    emitter = EventEmitter(run_dir / "events.jsonl", run_id=run_id)
    emitter.emit(EventType.RUN_START, ticker=ticker, trade_date=trade_date, tier=tier)

    ctx = RunContext(run_id=run_id, run_dir=run_dir, manifest=manifest, emitter=emitter)
    _install_interrupt_handler(ctx)
    return ctx


def resume_run(run_id: str, results_root: Path | str = "results") -> RunContext:
    run_dir = _run_dir(results_root, run_id)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run manifest for {run_id} at {manifest_path}")
    manifest = read_manifest(manifest_path)
    if manifest.status is not RunStatus.INTERRUPTED:
        raise ValueError(f"Run {run_id} is not resumable (status={manifest.status.value})")
    manifest.status = RunStatus.RUNNING
    write_manifest(manifest, manifest_path)
    emitter = EventEmitter(run_dir / "events.jsonl", run_id=run_id)
    emitter.emit(EventType.RUN_START, resumed=True, from_node=manifest.last_completed_node)
    ctx = RunContext(run_id=run_id, run_dir=run_dir, manifest=manifest, emitter=emitter)
    _install_interrupt_handler(ctx)
    return ctx


def finish_run(
    ctx: RunContext,
    *,
    status: RunStatus,
    last_completed_node: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    manifest_path = ctx.run_dir / "run_manifest.json"
    update_manifest(
        manifest_path,
        status=status,
        last_completed_node=last_completed_node or ctx.manifest.last_completed_node,
        error=error,
    )
    ctx.emitter.emit(EventType.RUN_END, status=status.value)


@dataclass
class RunRow:
    run_id: str
    ticker: str
    tier: str
    status: str
    started_at: str
    cost_usd: float


def list_runs(results_root: Path | str = "results") -> List[RunRow]:
    root = Path(results_root)
    if not root.exists():
        return []
    rows: List[RunRow] = []
    for child in root.iterdir():
        mpath = child / "run_manifest.json"
        if not mpath.exists():
            continue
        try:
            m = read_manifest(mpath)
        except Exception:  # noqa: BLE001 — skip corrupt manifests
            continue
        rows.append(
            RunRow(
                run_id=m.run_id, ticker=m.ticker, tier=m.tier,
                status=m.status.value, started_at=m.started_at, cost_usd=m.cost_usd,
            )
        )
    rows.sort(key=lambda r: r.started_at, reverse=True)
    return rows


def _install_interrupt_handler(ctx: RunContext) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        try:
            update_manifest(
                ctx.run_dir / "run_manifest.json",
                status=RunStatus.INTERRUPTED,
            )
            ctx.emitter.emit(EventType.RUN_END, status="interrupted")
        finally:
            sys.exit(130)
    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        # signal only works on main thread; tests may not be on main thread
        pass
