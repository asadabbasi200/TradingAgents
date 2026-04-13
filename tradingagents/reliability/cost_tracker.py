"""Running cost/token tracker wired as a LangChain callback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.runs.events import EventEmitter, EventType


# Prices in USD per 1M tokens. Source: Anthropic public pricing as of 2026-04.
# Cached reads are ~10% of normal input; cache writes are ~125% of normal input.
MODEL_PRICING: Dict[str, "ModelPricing"] = {}


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float  # cache reads


def register_pricing(model: str, input_: float, output: float, cached: float) -> None:
    MODEL_PRICING[model] = ModelPricing(input_, output, cached)


# Seed with the three Anthropic models used by tiers. Update when prices change.
register_pricing("claude-haiku-4-5",  input_=1.00,  output=5.00,  cached=0.10)
register_pricing("claude-sonnet-4-6", input_=3.00,  output=15.00, cached=0.30)
register_pricing("claude-opus-4-6",   input_=15.00, output=75.00, cached=1.50)


@dataclass
class AgentTotals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0


@dataclass
class RunTotals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    summarizations: int = 0


class CostTracker(BaseCallbackHandler):
    """Accumulates per-call LLM usage and writes llm_call events."""

    def __init__(self, emitter: Optional[EventEmitter] = None):
        self.emitter = emitter
        self.totals = RunTotals()
        self.by_agent: Dict[str, AgentTotals] = {}
        self.current_node: str = "unknown"

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage, model = self._extract_usage(response)
        if usage is None:
            return
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        cached_in = int(usage.get("cache_read_input_tokens", 0) or 0)
        cost = self._cost(model, tokens_in, tokens_out, cached_in)

        self.totals.calls += 1
        self.totals.tokens_in += tokens_in
        self.totals.tokens_out += tokens_out
        self.totals.tokens_cached += cached_in
        self.totals.cost_usd += cost

        agent_totals = self.by_agent.setdefault(self.current_node, AgentTotals())
        agent_totals.calls += 1
        agent_totals.tokens_in += tokens_in
        agent_totals.tokens_out += tokens_out
        agent_totals.tokens_cached += cached_in
        agent_totals.cost_usd += cost

        if self.emitter is not None:
            self.emitter.emit(
                EventType.LLM_CALL,
                node=self.current_node,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cached_in=cached_in,
                cost_usd=round(cost, 6),
            )

    @staticmethod
    def _extract_usage(response: Any):
        # LangChain LLMResult: .llm_output is a dict; individual generations
        # may also have response_metadata. Try both.
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            out = response.llm_output
            return out.get("usage"), out.get("model_name", "unknown")
        return None, "unknown"

    @staticmethod
    def _cost(model: str, tokens_in: int, tokens_out: int, cached_in: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        uncached_in = max(tokens_in - cached_in, 0)
        return (
            uncached_in * pricing.input_per_mtok / 1_000_000
            + cached_in * pricing.cached_input_per_mtok / 1_000_000
            + tokens_out * pricing.output_per_mtok / 1_000_000
        )

    def record_retry(self) -> None:
        self.totals.retries += 1

    def record_summarization(self) -> None:
        self.totals.summarizations += 1
