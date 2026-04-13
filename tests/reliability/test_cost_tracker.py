from pathlib import Path
from unittest.mock import MagicMock

from tradingagents.reliability.cost_tracker import CostTracker, ModelPricing
from tradingagents.runs.events import EventEmitter


def _mk_llm_result(input_tokens=1000, output_tokens=200, cached_in=0):
    result = MagicMock()
    result.llm_output = {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached_in,
            "cache_creation_input_tokens": 0,
        },
        "model_name": "claude-haiku-4-5",
    }
    return result


def test_cost_tracker_accumulates_totals(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.current_node = "market_analyst"

    tracker.on_llm_end(_mk_llm_result(1000, 200))
    tracker.on_llm_end(_mk_llm_result(500, 100))

    assert tracker.totals.calls == 2
    assert tracker.totals.tokens_in == 1500
    assert tracker.totals.tokens_out == 300
    assert tracker.totals.cost_usd > 0


def test_cost_tracker_per_agent_breakdown(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)

    tracker.current_node = "market_analyst"
    tracker.on_llm_end(_mk_llm_result(1000, 200))
    tracker.current_node = "bull_researcher"
    tracker.on_llm_end(_mk_llm_result(800, 300))

    assert tracker.by_agent["market_analyst"].calls == 1
    assert tracker.by_agent["bull_researcher"].calls == 1
    assert tracker.by_agent["market_analyst"].tokens_in == 1000


def test_cost_tracker_applies_cached_discount(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.current_node = "trader"

    # All input cached reads
    tracker.on_llm_end(_mk_llm_result(input_tokens=10000, output_tokens=500, cached_in=10000))

    # Cached reads are 0.1x the normal input price, so cost must be much lower
    # than a call with 10000 uncached input tokens would be.
    cached_cost = tracker.totals.cost_usd
    tracker2 = CostTracker(emitter=em)
    tracker2.current_node = "trader"
    tracker2.on_llm_end(_mk_llm_result(input_tokens=10000, output_tokens=500, cached_in=0))
    uncached_cost = tracker2.totals.cost_usd
    assert cached_cost < uncached_cost * 0.5
