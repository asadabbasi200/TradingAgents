from pathlib import Path

from tradingagents.runs.orchestrator import start_run, resume_run, finish_run, list_runs
from tradingagents.runs.manifest import RunStatus, read_manifest


def test_start_run_writes_manifest(tmp_path: Path):
    ctx = start_run(
        ticker="AAPL",
        trade_date="2026-04-10",
        tier="cheap",
        config_snapshot={"x": 1},
        results_root=tmp_path,
    )
    assert (tmp_path / ctx.run_id / "run_manifest.json").exists()
    m = read_manifest(tmp_path / ctx.run_id / "run_manifest.json")
    assert m.status is RunStatus.RUNNING
    assert m.ticker == "AAPL"
    assert m.tier == "cheap"


def test_finish_run_marks_completed(tmp_path: Path):
    ctx = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    finish_run(ctx, status=RunStatus.COMPLETED)
    m = read_manifest(tmp_path / ctx.run_id / "run_manifest.json")
    assert m.status is RunStatus.COMPLETED


def test_resume_run_requires_interrupted(tmp_path: Path):
    ctx = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    finish_run(ctx, status=RunStatus.COMPLETED)
    import pytest
    with pytest.raises(ValueError, match="not resumable"):
        resume_run(ctx.run_id, results_root=tmp_path)


def test_resume_run_from_interrupted(tmp_path: Path):
    ctx = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    finish_run(ctx, status=RunStatus.INTERRUPTED, last_completed_node="trader")

    resumed = resume_run(ctx.run_id, results_root=tmp_path)
    assert resumed.run_id == ctx.run_id
    assert resumed.manifest.status is RunStatus.RUNNING  # flipped back
    assert resumed.manifest.last_completed_node == "trader"


def test_list_runs_sorted_newest_first(tmp_path: Path):
    import time
    ctx1 = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    time.sleep(0.01)
    ctx2 = start_run("NVDA", "2026-04-10", "cheap", {}, results_root=tmp_path)
    rows = list_runs(tmp_path)
    assert rows[0].run_id == ctx2.run_id
    assert rows[1].run_id == ctx1.run_id
