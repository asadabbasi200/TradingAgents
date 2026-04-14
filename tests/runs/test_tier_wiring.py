from unittest.mock import MagicMock, patch

from tradingagents.default_config import DEFAULT_CONFIG


def test_graph_applies_tier_when_set():
    config = DEFAULT_CONFIG.copy()
    config["tier"] = "cheap"
    # Patch out actual LLM client construction — we only want to check config flow.
    with patch("tradingagents.graph.trading_graph.create_llm_client") as mock_create:
        fake_client = MagicMock()
        fake_client.get_llm.return_value = MagicMock()
        mock_create.return_value = fake_client
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        try:
            TradingAgentsGraph(config=config)
        except Exception:
            # Graph setup may fail downstream without real env — we only care
            # about whether create_llm_client was called with the cheap-tier model.
            pass
        assert mock_create.call_args_list, "create_llm_client was never called"
        models_seen = set()
        for call in mock_create.call_args_list:
            kwargs = call.kwargs or {}
            if "model" in kwargs:
                models_seen.add(kwargs["model"])
            elif len(call.args) >= 2:
                models_seen.add(call.args[1])
        assert "claude-haiku-4-5" in models_seen, f"expected cheap-tier model, got {models_seen}"
