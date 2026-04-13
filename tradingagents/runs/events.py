"""Append-only JSONL event stream for a run."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    NODE_START = "node_start"
    NODE_END = "node_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRY = "retry"
    SUMMARIZE = "summarize"
    CACHE_HIT = "cache_hit"
    ERROR = "error"


class EventEmitter:
    def __init__(self, path: Path | str, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event_type: EventType, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "type": event_type.value,
            **fields,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
