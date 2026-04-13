"""Stub LLM client for dry-run mode."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class _StubResponse:
    content: str
    response_metadata: dict = field(default_factory=lambda: {
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    })


@dataclass
class DryRunLLM:
    canned_response: str = (
        "This is a dry-run stub response. FINAL TRANSACTION PROPOSAL: **HOLD**"
    )
    call_log: List[Any] = field(default_factory=list)

    def invoke(self, input, config=None, **kwargs):
        self.call_log.append(input)
        return _StubResponse(content=self.canned_response)

    def bind_tools(self, tools, **kwargs):
        return self
