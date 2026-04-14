from pathlib import Path

from tradingagents.runs.checkpointer import make_checkpointer, thread_config


def test_make_checkpointer_creates_sqlite_file(tmp_path: Path):
    saver = make_checkpointer(tmp_path / "checkpoint.sqlite")
    assert (tmp_path / "checkpoint.sqlite").exists()
    # SqliteSaver exposes get and put
    assert hasattr(saver, "get")
    assert hasattr(saver, "put")


def test_thread_config_shape():
    cfg = thread_config("r1")
    assert cfg == {"configurable": {"thread_id": "r1"}}
