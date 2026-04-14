"""End-to-end test that an interrupted run can be resumed via the orchestrator."""
from pathlib import Path

from tradingagents.runs.orchestrator import start_run, finish_run, resume_run
from tradingagents.runs.manifest import RunStatus, read_manifest


def test_interrupted_run_can_be_resumed_and_completed(tmp_path: Path):
    # 1. Start a run.
    ctx = start_run(
        ticker="AAPL",
        trade_date="2026-04-10",
        tier="cheap",
        config_snapshot={"x": 1},
        results_root=tmp_path,
    )
    run_dir = tmp_path / ctx.run_id

    # 2. Simulate interruption mid-run (set status to interrupted, remember the node we got to).
    finish_run(ctx, status=RunStatus.INTERRUPTED, last_completed_node="trader")
    m = read_manifest(run_dir / "run_manifest.json")
    assert m.status is RunStatus.INTERRUPTED
    assert m.last_completed_node == "trader"

    # 3. Resume the run.
    resumed = resume_run(ctx.run_id, results_root=tmp_path)
    assert resumed.run_id == ctx.run_id
    assert resumed.manifest.status is RunStatus.RUNNING  # flipped back to running
    assert resumed.manifest.last_completed_node == "trader"  # preserved across resume

    # 4. Finish the resumed run cleanly.
    finish_run(resumed, status=RunStatus.COMPLETED)
    final = read_manifest(run_dir / "run_manifest.json")
    assert final.status is RunStatus.COMPLETED
