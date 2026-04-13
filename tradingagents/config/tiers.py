"""Tier presets for cost/quality trade-offs."""
from __future__ import annotations

from typing import Any, Dict, Literal

TierName = Literal["cheap", "balanced", "max"]


TIERS: Dict[str, Dict[str, Any]] = {
    "cheap": {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-haiku-4-5",
        "quick_think_llm": "claude-haiku-4-5",
        "anthropic_effort": "low",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "per_call_input_token_ceiling": 20_000,
        "prompt_caching": True,
        "lazy_summarization": True,
    },
    "balanced": {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-sonnet-4-6",
        "quick_think_llm": "claude-haiku-4-5",
        "anthropic_effort": "medium",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "per_call_input_token_ceiling": 40_000,
        "prompt_caching": True,
        "lazy_summarization": True,
    },
    "max": {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-opus-4-6",
        "quick_think_llm": "claude-sonnet-4-6",
        "anthropic_effort": "high",
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 2,
        "per_call_input_token_ceiling": 80_000,
        "prompt_caching": True,
        "lazy_summarization": True,
    },
}


def apply_tier(config: Dict[str, Any], tier: str) -> Dict[str, Any]:
    """Return a new config dict with the tier preset merged in.

    The input config is not mutated. The `tier` field is recorded so
    downstream code can read which preset was applied.
    """
    if tier not in TIERS:
        raise ValueError(f"Unknown tier: {tier}. Expected one of {list(TIERS)}.")
    merged = dict(config)
    merged.update(TIERS[tier])
    merged["tier"] = tier
    return merged
