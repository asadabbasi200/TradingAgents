import pytest

from tradingagents.config.tiers import TIERS, apply_tier, TierName


def test_three_tiers_exist():
    assert set(TIERS.keys()) == {"cheap", "balanced", "max"}


@pytest.mark.parametrize("tier", ["cheap", "balanced", "max"])
def test_tier_has_required_fields(tier):
    t = TIERS[tier]
    required = {
        "deep_think_llm",
        "quick_think_llm",
        "anthropic_effort",
        "max_debate_rounds",
        "max_risk_discuss_rounds",
        "per_call_input_token_ceiling",
    }
    assert required.issubset(t.keys())


def test_apply_tier_merges_into_config():
    base = {"llm_provider": "anthropic", "foo": "bar"}
    out = apply_tier(base, "cheap")
    assert out["foo"] == "bar"  # untouched
    assert out["deep_think_llm"] == "claude-haiku-4-5"
    assert out["quick_think_llm"] == "claude-haiku-4-5"
    assert out["tier"] == "cheap"


def test_apply_tier_does_not_mutate_input():
    base = {"llm_provider": "anthropic"}
    _ = apply_tier(base, "balanced")
    assert "tier" not in base  # original untouched


def test_apply_tier_rejects_unknown_tier():
    with pytest.raises(ValueError):
        apply_tier({}, "ludicrous")


def test_apply_tier_snapshot_cheap():
    # Snapshot guards against accidental edits to the cheap preset values.
    out = apply_tier({}, "cheap")
    assert out["deep_think_llm"] == "claude-haiku-4-5"
    assert out["quick_think_llm"] == "claude-haiku-4-5"
    assert out["anthropic_effort"] == "low"
    assert out["max_debate_rounds"] == 1
    assert out["max_risk_discuss_rounds"] == 1
    assert out["per_call_input_token_ceiling"] == 20_000
