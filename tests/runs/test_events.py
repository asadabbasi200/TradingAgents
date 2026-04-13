import json
from pathlib import Path

from tradingagents.runs.events import EventEmitter, EventType


def test_emit_writes_one_line_per_event(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    em = EventEmitter(path, run_id="r1")
    em.emit(EventType.NODE_START, node="market_analyst")
    em.emit(EventType.LLM_CALL, node="market_analyst", model="claude-haiku-4-5",
            tokens_in=120, tokens_out=40, cost_usd=0.001)
    em.emit(EventType.NODE_END, node="market_analyst", duration_ms=2840)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(l) for l in lines]
    assert records[0]["type"] == "node_start"
    assert records[0]["run_id"] == "r1"
    assert "ts" in records[0]
    assert records[1]["tokens_in"] == 120


def test_emit_appends_to_existing_file(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    EventEmitter(path, run_id="r1").emit(EventType.RUN_START)
    EventEmitter(path, run_id="r1").emit(EventType.RUN_END)
    assert len(path.read_text().strip().splitlines()) == 2
