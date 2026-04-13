from tradingagents.reliability.dry_run import DryRunLLM


def test_dry_run_llm_returns_canned_response():
    llm = DryRunLLM()
    out = llm.invoke("anything")
    assert out.content  # non-empty
    assert "FINAL TRANSACTION PROPOSAL" in out.content


def test_dry_run_llm_records_calls():
    llm = DryRunLLM()
    llm.invoke("first")
    llm.invoke("second")
    assert len(llm.call_log) == 2


def test_factory_returns_dry_client_when_dry_run_true():
    from tradingagents.llm_clients import create_llm_client
    client = create_llm_client(
        provider="anthropic", model="claude-haiku-4-5",
        api_key="dummy", dry_run=True,
    )
    llm = client.get_llm()
    out = llm.invoke("hi")
    assert "FINAL TRANSACTION PROPOSAL" in out.content
