import re
from datetime import datetime

from tradingagents.runs.manifest import mint_run_id, RunManifest, RunStatus


def test_mint_run_id_format():
    rid = mint_run_id(ticker="AAPL", trade_date="2026-04-10", now=datetime(2026, 4, 13, 9, 14))
    # Format: YYYY-MM-DD_TICKER_<4-char hex>
    assert re.match(r"^2026-04-13_AAPL_[0-9a-f]{4}$", rid)


def test_mint_run_id_uniqueness():
    ids = {mint_run_id(ticker="AAPL", trade_date="2026-04-10") for _ in range(100)}
    assert len(ids) == 100  # collisions are astronomically unlikely


def test_manifest_roundtrip(tmp_path):
    m = RunManifest(
        run_id="2026-04-13_AAPL_a3f2",
        ticker="AAPL",
        trade_date="2026-04-10",
        tier="balanced",
        started_at="2026-04-13T09:14:22+00:00",
    )
    from tradingagents.runs.manifest import write_manifest, read_manifest
    path = tmp_path / "run_manifest.json"
    write_manifest(m, path)

    loaded = read_manifest(path)
    assert loaded.run_id == m.run_id
    assert loaded.status is RunStatus.RUNNING
    assert loaded.tokens.input == 0


def test_manifest_status_transitions(tmp_path):
    from tradingagents.runs.manifest import write_manifest, update_manifest
    m = RunManifest(
        run_id="r1", ticker="AAPL", trade_date="2026-04-10",
        tier="cheap", started_at="2026-04-13T09:00:00+00:00",
    )
    path = tmp_path / "run_manifest.json"
    write_manifest(m, path)

    update_manifest(path, status=RunStatus.INTERRUPTED, last_completed_node="trader")
    from tradingagents.runs.manifest import read_manifest
    loaded = read_manifest(path)
    assert loaded.status is RunStatus.INTERRUPTED
    assert loaded.last_completed_node == "trader"
