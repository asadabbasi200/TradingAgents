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
    """Anthropic billing: input_tokens excludes cache reads/writes (separate counts).

    A call where 10k tokens are served from cache (0 fresh input) must cost
    much less than a call with 10k fresh input tokens.
    """
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.current_node = "trader"

    # 0 fresh input + 10k cached reads
    tracker.on_llm_end(_mk_llm_result(input_tokens=0, output_tokens=500, cached_in=10000))
    cached_cost = tracker.totals.cost_usd

    # 10k fresh input, no cache
    tracker2 = CostTracker(emitter=em)
    tracker2.current_node = "trader"
    tracker2.on_llm_end(_mk_llm_result(input_tokens=10000, output_tokens=500, cached_in=0))
    uncached_cost = tracker2.totals.cost_usd

    # Cached reads are 0.1x normal input price for Haiku → input portion
    # of the all-cached call is ~10x cheaper. Output cost is identical so
    # the ratio is dominated by input.
    assert cached_cost < uncached_cost * 0.5


def test_cost_tracker_reads_node_from_langgraph_metadata(tmp_path: Path):
    """on_llm_start (and on_chat_model_start) must extract langgraph_node from metadata."""
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)

    assert tracker.current_node == "unknown"

    # LangGraph passes the current node name in metadata.langgraph_node
    tracker.on_chat_model_start(
        serialized={},
        prompts=[],
        metadata={"langgraph_node": "market_analyst", "langgraph_step": 1},
    )
    assert tracker.current_node == "market_analyst"

    # A subsequent llm_end attributes the call to that node
    tracker.on_llm_end(_mk_llm_result(input_tokens=1000, output_tokens=200))
    assert tracker.by_agent["market_analyst"].calls == 1
    assert "unknown" not in tracker.by_agent

    # Switching nodes works
    tracker.on_chat_model_start(
        serialized={}, prompts=[], metadata={"langgraph_node": "bull_researcher"}
    )
    tracker.on_llm_end(_mk_llm_result(input_tokens=500, output_tokens=100))
    assert tracker.by_agent["bull_researcher"].calls == 1


def test_cost_tracker_metadata_without_node_is_safe(tmp_path: Path):
    """Missing or None metadata should not crash; current_node stays unchanged."""
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.current_node = "preset"

    tracker.on_chat_model_start(serialized={}, prompts=[], metadata=None)
    assert tracker.current_node == "preset"

    tracker.on_chat_model_start(serialized={}, prompts=[], metadata={})
    assert tracker.current_node == "preset"


def test_extract_usage_handles_langchain_normalized_shape(tmp_path: Path):
    """When the response only has usage_metadata (no llm_output, no response_metadata.usage),
    the extractor must convert LangChain's inclusive input_tokens to native Anthropic semantics.
    """
    # Build a fake LLMResult with the LangChain-normalized shape (no llm_output)
    response = MagicMock()
    response.llm_output = None  # no legacy shape
    fake_gen = MagicMock()
    fake_msg = MagicMock()
    fake_msg.usage_metadata = {
        "input_tokens": 16510,        # INCLUDES cache reads — LangChain's inclusive shape
        "output_tokens": 4,
        "input_token_details": {
            "cache_read": 16502,
            "cache_creation": 0,
        },
    }
    fake_msg.response_metadata = {"model_name": "claude-haiku-4-5"}
    # Important: response_metadata has NO "usage" key, forcing the normalized fallback
    fake_gen.message = fake_msg
    response.generations = [[fake_gen]]

    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.on_llm_end(response)

    # Fresh input should be 16510 - 16502 = 8
    assert tracker.totals.tokens_in == 8
    assert tracker.totals.tokens_cached == 16502
    # Cost: 8 fresh × $1/M + 16502 cached × $0.10/M + 4 output × $5/M
    expected = 8 * 1.00 / 1_000_000 + 16502 * 0.10 / 1_000_000 + 4 * 5.00 / 1_000_000
    assert abs(tracker.totals.cost_usd - expected) < 1e-9


def test_cost_tracker_charges_cache_creation_at_premium(tmp_path: Path):
    """Anthropic charges cache_creation_input_tokens at 1.25x the normal input price."""
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)

    # Synthesize a response with cache_creation set
    result = MagicMock()
    result.llm_output = {
        "usage": {
            "input_tokens": 0,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 10000,
        },
        "model_name": "claude-haiku-4-5",
    }
    tracker.on_llm_end(result)

    # 10000 cache_creation tokens × $1/M × 1.25 + 100 output × $5/M
    expected = 10000 * 1.00 / 1_000_000 * 1.25 + 100 * 5.00 / 1_000_000
    assert abs(tracker.totals.cost_usd - expected) < 1e-9
