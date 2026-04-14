"""Verify that running with no tier flag is back-compat with the prior behavior."""
from unittest.mock import patch, MagicMock

from tradingagents.default_config import DEFAULT_CONFIG


def test_graph_without_tier_uses_default_models():
    """No tier set -> models come from DEFAULT_CONFIG, not from any preset."""
    config = DEFAULT_CONFIG.copy()
    assert config.get("tier") is None
    expected_deep = config["deep_think_llm"]
    expected_quick = config["quick_think_llm"]

    with patch("tradingagents.graph.trading_graph.create_llm_client") as mock_create:
        fake_client = MagicMock()
        fake_client.get_llm.return_value = MagicMock()
        mock_create.return_value = fake_client
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        try:
            TradingAgentsGraph(config=config)
        except Exception:
            # Graph setup may fail downstream without real env — we only care
            # that create_llm_client was called with the default models.
            pass

    models_seen = set()
    for call in mock_create.call_args_list:
        kwargs = call.kwargs or {}
        if "model" in kwargs:
            models_seen.add(kwargs["model"])
        elif len(call.args) >= 2:
            models_seen.add(call.args[1])

    # Both default models should appear in the calls; no cheap/balanced/max model substituted in.
    assert expected_deep in models_seen, (
        f"expected default deep_think model {expected_deep}, got {models_seen}"
    )
    assert expected_quick in models_seen, (
        f"expected default quick_think model {expected_quick}, got {models_seen}"
    )
