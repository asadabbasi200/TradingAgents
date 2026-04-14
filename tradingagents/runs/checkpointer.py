"""LangGraph SqliteSaver adapter for per-run checkpoints."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver


def make_checkpointer(path: Path | str) -> SqliteSaver:
    """Create a SqliteSaver at the given path. Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    # Force table creation so the file is materialized immediately.
    saver.setup()
    return saver


def thread_config(run_id: str) -> Dict[str, Any]:
    """Standard LangGraph thread config — keyed on run_id."""
    return {"configurable": {"thread_id": run_id}}
