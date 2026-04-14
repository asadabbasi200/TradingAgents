"""Stub LLM client for dry-run mode.

Provides ``DryRunLLM``, a LangChain ``Runnable`` subclass that implements the
minimum surface needed by our agent nodes so that ``prompt | llm.bind_tools(...)``
pipelines work without any real LLM or tool calls.

Responses always contain the literal ``FINAL TRANSACTION PROPOSAL: **HOLD**``
so graph termination conditions trip cleanly.
"""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


def _make_stub_message(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[],
        response_metadata={
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        },
        usage_metadata={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        id="dry-run-stub",
    )


class DryRunLLM(Runnable):
    """Runnable stub LLM — returns a canned terminal decision on every call."""

    canned_response: str = (
        "This is a dry-run stub response. FINAL TRANSACTION PROPOSAL: **HOLD**"
    )

    def __init__(self, canned_response: Optional[str] = None) -> None:
        super().__init__()
        if canned_response is not None:
            self.canned_response = canned_response
        self.call_log: List[Any] = []

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.call_log.append(input)
        return _make_stub_message(self.canned_response)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DryRunLLM":
        # Tools are ignored in dry-run — stub always short-circuits with HOLD.
        return self
