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
# Register both the bare alias and the dated API identifier — Anthropic's API
# returns the dated form (e.g. "claude-haiku-4-5-20251001") in response metadata,
# so we need to look up by both shapes.
register_pricing("claude-haiku-4-5",            input_=1.00,  output=5.00,  cached=0.10)
register_pricing("claude-haiku-4-5-20251001",   input_=1.00,  output=5.00,  cached=0.10)
register_pricing("claude-sonnet-4-6",           input_=3.00,  output=15.00, cached=0.30)
register_pricing("claude-sonnet-4-6-20251001",  input_=3.00,  output=15.00, cached=0.30)
register_pricing("claude-opus-4-6",             input_=15.00, output=75.00, cached=1.50)
register_pricing("claude-opus-4-6-20251001",    input_=15.00, output=75.00, cached=1.50)


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

    def on_llm_start(self, serialized: Any, prompts: Any, *, metadata: Any = None, **kwargs: Any) -> None:
        """Update current_node from LangGraph metadata before each LLM call.

        LangGraph populates `metadata['langgraph_node']` on every callback
        invocation inside a node, so the cost tracker can attribute calls
        to the correct agent without any per-agent wiring.
        """
        if isinstance(metadata, dict):
            node = metadata.get("langgraph_node")
            if node:
                self.current_node = str(node)

    # LangChain's chat-model callbacks fire on_chat_model_start (not on_llm_start)
    # for ChatModel subclasses. Hook both so we cover both paths.
    on_chat_model_start = on_llm_start

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage, model = self._extract_usage(response)
        if usage is None:
            return
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        cached_in = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cost = self._cost(model, tokens_in, tokens_out, cached_in, cache_creation)

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
                cache_creation=cache_creation,
                cost_usd=round(cost, 6),
            )

    @staticmethod
    def _extract_usage(response: Any):
        """Extract usage and model name from an LLMResult.

        Three response shapes encountered in practice:
        1. Older LangChain: response.llm_output["usage"] (native Anthropic shape)
        2. Newer LangChain ChatModel: generation.message.response_metadata["usage"] (native shape)
        3. LangChain normalized: generation.message.usage_metadata (input_tokens INCLUDES cache reads)

        Prefer (1) and (2) (native Anthropic semantics: input_tokens excludes
        cache reads/writes). Fall back to (3) and convert.
        """
        # Shape 1: legacy llm_output dict
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            out = response.llm_output
            usage = out.get("usage")
            if usage and (usage.get("input_tokens") is not None or usage.get("output_tokens") is not None):
                return usage, out.get("model_name", "unknown")

        # Shapes 2 & 3: pull from the first generation's message
        gens = getattr(response, "generations", None)
        if not gens or not gens[0]:
            return None, "unknown"
        gen = gens[0][0] if isinstance(gens[0], list) else gens[0]
        msg = getattr(gen, "message", None)
        if msg is None:
            return None, "unknown"

        resp_meta = getattr(msg, "response_metadata", None) or {}
        model = resp_meta.get("model_name", "unknown")

        # Shape 2: native Anthropic usage in response_metadata
        native_usage = resp_meta.get("usage")
        if isinstance(native_usage, dict) and "input_tokens" in native_usage:
            return native_usage, model

        # Shape 3: LangChain normalized — input_tokens INCLUDES cache reads, subtract them
        usage_meta = getattr(msg, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            details = usage_meta.get("input_token_details", {}) or {}
            cache_read = int(details.get("cache_read", 0) or 0)
            cache_creation = int(details.get("cache_creation", 0) or 0)
            inclusive_input = int(usage_meta.get("input_tokens", 0) or 0)
            # Convert to native Anthropic shape: fresh input only
            fresh_input = max(inclusive_input - cache_read - cache_creation, 0)
            normalized = {
                "input_tokens": fresh_input,
                "output_tokens": int(usage_meta.get("output_tokens", 0) or 0),
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
            return normalized, model

        return None, "unknown"

    @staticmethod
    def _cost(model: str, tokens_in: int, tokens_out: int, cached_in: int, cache_creation: int = 0) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        # Anthropic billing: input_tokens already EXCLUDES cache reads and writes.
        # cached_in reads are billed at the cached rate (~0.1x normal input).
        # cache_creation writes are billed at 1.25x normal input.
        return (
            tokens_in * pricing.input_per_mtok / 1_000_000
            + cached_in * pricing.cached_input_per_mtok / 1_000_000
            + cache_creation * pricing.input_per_mtok * 1.25 / 1_000_000
            + tokens_out * pricing.output_per_mtok / 1_000_000
        )

    def record_retry(self) -> None:
        self.totals.retries += 1

    def record_summarization(self) -> None:
        self.totals.summarizations += 1
