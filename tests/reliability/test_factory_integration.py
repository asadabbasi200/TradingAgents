from pathlib import Path

from tradingagents.llm_clients import create_llm_client
from tradingagents.reliability.cost_tracker import CostTracker
from tradingagents.runs.events import EventEmitter


def test_factory_injects_cost_tracker(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)

    client = create_llm_client(
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key="dummy",
        callbacks=[tracker],
    )
    llm = client.get_llm()
    # ChatAnthropic stores callbacks on .callbacks
    assert tracker in (llm.callbacks or [])


def test_invoke_with_retries_uses_policy(monkeypatch):
    from tradingagents.reliability.retry import RetryPolicy
    from tradingagents.llm_clients.base_client import BaseLLMClient

    class Dummy(BaseLLMClient):
        def __init__(self):
            super().__init__("dummy-model")
            self._calls = 0

        def get_llm(self):
            parent = self

            class _LLM:
                def invoke(self, *a, **k):
                    parent._calls += 1
                    if parent._calls < 3:
                        class Transient(Exception):
                            pass
                        raise Transient("timeout")
                    return "ok"

            return _LLM()

        def validate_model(self) -> bool:
            return True

    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = Dummy()
    client.kwargs["retry_policy"] = RetryPolicy(
        max_attempts_transient=5, base_delay_s=0.01,
    )
    assert client.invoke_with_retries("hi") == "ok"
    assert client._calls == 3
