"""Shared test fakes for reliability-layer tests."""
from __future__ import annotations

from typing import Any, List


class ScriptedLLM:
    """LLM stand-in that replays a scripted list of responses or exceptions."""

    def __init__(self, script: List[Any]):
        self.script = list(script)
        self.calls: List[Any] = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        if not self.script:
            raise RuntimeError("ScriptedLLM exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
