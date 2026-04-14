# Reliability & Cost Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TradingAgents runs survive transient failures, stay within predictable cost envelopes via tier presets, support resume-from-checkpoint, and expose per-run cost/token/latency observability.

**Architecture:** Wrap the existing LLM factory with retry/backoff and a cost-tracking callback; introduce a `run_id`-keyed results layout with a JSON manifest, JSONL event stream, and LangGraph SQLite checkpointer; add tier presets (`cheap`/`balanced`/`max`) that map to models, effort, debate rounds, and per-call token ceilings; enable Anthropic prompt caching on stable blocks; trigger lazy Haiku-based summarization of debate history before calls that would exceed the tier's ceiling.

**Tech Stack:** Python 3.10+, LangGraph (SqliteSaver checkpointer), langchain-anthropic (prompt caching via `cache_control`), Rich (live CLI panel), pytest (new dev dep), tenacity (retry backoff — already transitive via langchain), typer (existing CLI).

**Spec:** `docs/superpowers/specs/2026-04-13-reliability-hardening-design.md`

---

## Working notes for the implementing engineer

- Existing tests use `unittest` in flat `tests/`. We will **add pytest** and put new tests under `tests/reliability/`, `tests/runs/`, `tests/config/`, `tests/integration/`. Existing `unittest` tests keep working (pytest runs them fine).
- Every task ends with a commit. Commit messages follow Conventional Commits.
- All work on branch `feat/reliability-hardening` on `origin` (which points to your fork `asadabbasi200/TradingAgents`).
- TDD: red → green → commit. Never write implementation before the failing test exists.
- Where a test uses a fake LLM, the fakes go in `tests/reliability/_fakes.py` to keep them reusable.
- Do NOT touch the existing `./results/<TICKER>/<DATE>/` output path in this sub-project. New `results/<run_id>/` is additive.

---

## Task 0: Project setup

**Files:**
- Modify: `pyproject.toml`
- Create: `tradingagents/reliability/__init__.py`
- Create: `tradingagents/reliability/trace.py`
- Create: `tradingagents/config/__init__.py`
- Create: `tradingagents/runs/__init__.py`
- Create: `tests/reliability/__init__.py`
- Create: `tests/runs/__init__.py`
- Create: `tests/config/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/reliability/_fakes.py`

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b feat/reliability-hardening
```

- [ ] **Step 2: Add pytest to pyproject.toml**

Add an `[project.optional-dependencies]` table with a `dev` group.

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
    "freezegun>=1.4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

- [ ] **Step 3: Install dev deps**

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: pytest, pytest-mock, freezegun installed.

- [ ] **Step 4: Create new package directories with empty `__init__.py`**

```bash
mkdir -p tradingagents/reliability tradingagents/config tradingagents/runs
touch tradingagents/reliability/__init__.py tradingagents/config/__init__.py tradingagents/runs/__init__.py
mkdir -p tests/reliability tests/runs tests/config tests/integration
touch tests/reliability/__init__.py tests/runs/__init__.py tests/config/__init__.py tests/integration/__init__.py
```

- [ ] **Step 5: Write `tradingagents/reliability/trace.py` stub**

This is the no-op tracer mentioned in the spec — future hook for LangSmith/OTel. Shipped as a stub so later code can import `get_tracer()` without breaking.

```python
"""No-op tracer stub. Future hook for LangSmith/OpenTelemetry."""
from contextlib import contextmanager
from typing import Iterator


class _NoopTracer:
    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield

    def event(self, name: str, **attrs) -> None:
        pass


_tracer = _NoopTracer()


def get_tracer() -> _NoopTracer:
    return _tracer
```

- [ ] **Step 6: Create fakes module**

File: `tests/reliability/_fakes.py`

```python
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
```

- [ ] **Step 7: Verify pytest discovery**

```bash
pytest tests/ --collect-only 2>&1 | tail -20
```

Expected: pytest collects the existing unittest-based tests plus the new empty test files without error.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml tradingagents/reliability tradingagents/config tradingagents/runs tests/reliability tests/runs tests/config tests/integration
git commit -m "chore: scaffold reliability/config/runs packages and pytest dev deps"
```

---

## Task 1: Run ID minting and RunManifest

**Files:**
- Create: `tradingagents/runs/manifest.py`
- Create: `tests/runs/test_manifest.py`

The manifest is the heart of the new results layout. It tracks a run's identity, tier, status, cost, and enough state to support resume.

- [ ] **Step 1: Write failing test for run_id format**

File: `tests/runs/test_manifest.py`

```python
import re
from datetime import datetime

from tradingagents.runs.manifest import mint_run_id, RunManifest, RunStatus


def test_mint_run_id_format():
    rid = mint_run_id(ticker="AAPL", trade_date="2026-04-10", now=datetime(2026, 4, 13, 9, 14))
    # Format: YYYY-MM-DD_TICKER_<4-char hex>
    assert re.match(r"^2026-04-13_AAPL_[0-9a-f]{4}$", rid)


def test_mint_run_id_uniqueness():
    ids = {mint_run_id(ticker="AAPL", trade_date="2026-04-10") for _ in range(100)}
    assert len(ids) == 100  # collisions are astronomically unlikely
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/runs/test_manifest.py -v
```

Expected: `ImportError: cannot import name 'mint_run_id'`.

- [ ] **Step 3: Implement `mint_run_id` and RunStatus/RunManifest skeleton**

File: `tradingagents/runs/manifest.py`

```python
"""Run identity, manifest, and lifecycle state."""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def mint_run_id(ticker: str, trade_date: str, now: Optional[datetime] = None) -> str:
    """Generate a unique run identifier.

    Format: YYYY-MM-DD_TICKER_<4-char hex suffix>
    """
    now = now or datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    suffix = secrets.token_hex(2)
    return f"{date_str}_{ticker.upper()}_{suffix}"


@dataclass
class TokenTotals:
    input: int = 0
    output: int = 0
    cached: int = 0


@dataclass
class RunManifest:
    run_id: str
    ticker: str
    trade_date: str
    tier: str
    started_at: str
    status: RunStatus = RunStatus.RUNNING
    last_completed_node: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens: TokenTotals = field(default_factory=TokenTotals)
    error: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        tokens = data.pop("tokens", {})
        return cls(
            tokens=TokenTotals(**tokens) if isinstance(tokens, dict) else TokenTotals(),
            status=RunStatus(data.pop("status", "running")),
            **data,
        )
```

- [ ] **Step 4: Run the tests — expect pass**

```bash
pytest tests/runs/test_manifest.py::test_mint_run_id_format tests/runs/test_manifest.py::test_mint_run_id_uniqueness -v
```

Expected: both pass.

- [ ] **Step 5: Write failing test for manifest persistence**

Append to `tests/runs/test_manifest.py`:

```python
def test_manifest_roundtrip(tmp_path):
    m = RunManifest(
        run_id="2026-04-13_AAPL_a3f2",
        ticker="AAPL",
        trade_date="2026-04-10",
        tier="balanced",
        started_at="2026-04-13T09:14:22+00:00",
    )
    from tradingagents.runs.manifest import write_manifest, read_manifest
    path = tmp_path / "run_manifest.json"
    write_manifest(m, path)

    loaded = read_manifest(path)
    assert loaded.run_id == m.run_id
    assert loaded.status is RunStatus.RUNNING
    assert loaded.tokens.input == 0


def test_manifest_status_transitions(tmp_path):
    from tradingagents.runs.manifest import write_manifest, update_manifest
    m = RunManifest(
        run_id="r1", ticker="AAPL", trade_date="2026-04-10",
        tier="cheap", started_at="2026-04-13T09:00:00+00:00",
    )
    path = tmp_path / "run_manifest.json"
    write_manifest(m, path)

    update_manifest(path, status=RunStatus.INTERRUPTED, last_completed_node="trader")
    from tradingagents.runs.manifest import read_manifest
    loaded = read_manifest(path)
    assert loaded.status is RunStatus.INTERRUPTED
    assert loaded.last_completed_node == "trader"
```

- [ ] **Step 6: Implement `write_manifest`, `read_manifest`, `update_manifest`**

Append to `tradingagents/runs/manifest.py`:

```python
def write_manifest(manifest: RunManifest, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = datetime.now(timezone.utc).isoformat()
    # Atomic write: temp file + rename, so an interrupted write can't corrupt.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
    os.replace(tmp, path)


def read_manifest(path: Path | str) -> RunManifest:
    path = Path(path)
    data = json.loads(path.read_text())
    return RunManifest.from_dict(data)


def update_manifest(path: Path | str, **fields: Any) -> RunManifest:
    m = read_manifest(path)
    for k, v in fields.items():
        setattr(m, k, v)
    write_manifest(m, path)
    return m
```

- [ ] **Step 7: Run tests — expect pass**

```bash
pytest tests/runs/test_manifest.py -v
```

Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add tradingagents/runs/manifest.py tests/runs/test_manifest.py
git commit -m "feat(runs): add run_id minting and RunManifest with atomic persistence"
```

---

## Task 2: Event stream emitter (events.jsonl)

**Files:**
- Create: `tradingagents/runs/events.py`
- Create: `tests/runs/test_events.py`

Append-only JSONL writer that flushes on every event. Thread-safe so callbacks from async graph execution can't interleave partial lines.

- [ ] **Step 1: Write failing test**

File: `tests/runs/test_events.py`

```python
import json
from pathlib import Path

from tradingagents.runs.events import EventEmitter, EventType


def test_emit_writes_one_line_per_event(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    em = EventEmitter(path, run_id="r1")
    em.emit(EventType.NODE_START, node="market_analyst")
    em.emit(EventType.LLM_CALL, node="market_analyst", model="claude-haiku-4-5",
            tokens_in=120, tokens_out=40, cost_usd=0.001)
    em.emit(EventType.NODE_END, node="market_analyst", duration_ms=2840)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(l) for l in lines]
    assert records[0]["type"] == "node_start"
    assert records[0]["run_id"] == "r1"
    assert "ts" in records[0]
    assert records[1]["tokens_in"] == 120


def test_emit_appends_to_existing_file(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    EventEmitter(path, run_id="r1").emit(EventType.RUN_START)
    EventEmitter(path, run_id="r1").emit(EventType.RUN_END)
    assert len(path.read_text().strip().splitlines()) == 2
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/runs/test_events.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement EventEmitter**

File: `tradingagents/runs/events.py`

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/runs/test_events.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/runs/events.py tests/runs/test_events.py
git commit -m "feat(runs): add thread-safe EventEmitter for events.jsonl"
```

---

## Task 3: Cost tracker callback

**Files:**
- Create: `tradingagents/reliability/cost_tracker.py`
- Create: `tests/reliability/test_cost_tracker.py`

LangChain `BaseCallbackHandler` that accumulates per-call token/cost totals and forwards an `llm_call` event to an `EventEmitter`.

- [ ] **Step 1: Write failing test**

File: `tests/reliability/test_cost_tracker.py`

```python
from pathlib import Path
from unittest.mock import MagicMock

from tradingagents.reliability.cost_tracker import CostTracker, ModelPricing
from tradingagents.runs.events import EventEmitter


def _mk_llm_result(input_tokens=1000, output_tokens=200, cached_in=0):
    result = MagicMock()
    result.llm_output = {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached_in,
            "cache_creation_input_tokens": 0,
        },
        "model_name": "claude-haiku-4-5",
    }
    return result


def test_cost_tracker_accumulates_totals(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.current_node = "market_analyst"

    tracker.on_llm_end(_mk_llm_result(1000, 200))
    tracker.on_llm_end(_mk_llm_result(500, 100))

    assert tracker.totals.calls == 2
    assert tracker.totals.tokens_in == 1500
    assert tracker.totals.tokens_out == 300
    assert tracker.totals.cost_usd > 0


def test_cost_tracker_per_agent_breakdown(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)

    tracker.current_node = "market_analyst"
    tracker.on_llm_end(_mk_llm_result(1000, 200))
    tracker.current_node = "bull_researcher"
    tracker.on_llm_end(_mk_llm_result(800, 300))

    assert tracker.by_agent["market_analyst"].calls == 1
    assert tracker.by_agent["bull_researcher"].calls == 1
    assert tracker.by_agent["market_analyst"].tokens_in == 1000


def test_cost_tracker_applies_cached_discount(tmp_path: Path):
    em = EventEmitter(tmp_path / "events.jsonl", run_id="r1")
    tracker = CostTracker(emitter=em)
    tracker.current_node = "trader"

    # All input cached reads
    tracker.on_llm_end(_mk_llm_result(input_tokens=10000, output_tokens=500, cached_in=10000))

    # Cached reads are 0.1x the normal input price, so cost must be much lower
    # than a call with 10000 uncached input tokens would be.
    cached_cost = tracker.totals.cost_usd
    tracker2 = CostTracker(emitter=em)
    tracker2.current_node = "trader"
    tracker2.on_llm_end(_mk_llm_result(input_tokens=10000, output_tokens=500, cached_in=0))
    uncached_cost = tracker2.totals.cost_usd
    assert cached_cost < uncached_cost * 0.5
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/reliability/test_cost_tracker.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement CostTracker**

File: `tradingagents/reliability/cost_tracker.py`

```python
"""Running cost/token tracker wired as a LangChain callback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.runs.events import EventEmitter, EventType


# Prices in USD per 1M tokens. Source: Anthropic public pricing as of 2026-04.
# Cached reads are ~10% of normal input; cache writes are ~125% of normal input.
MODEL_PRICING: Dict[str, "ModelPricing"] = {}


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float  # cache reads


def register_pricing(model: str, input_: float, output: float, cached: float) -> None:
    MODEL_PRICING[model] = ModelPricing(input_, output, cached)


# Seed with the three Anthropic models used by tiers. Update when prices change.
register_pricing("claude-haiku-4-5",  input_=1.00,  output=5.00,  cached=0.10)
register_pricing("claude-sonnet-4-6", input_=3.00,  output=15.00, cached=0.30)
register_pricing("claude-opus-4-6",   input_=15.00, output=75.00, cached=1.50)


@dataclass
class AgentTotals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0


@dataclass
class RunTotals:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    summarizations: int = 0


class CostTracker(BaseCallbackHandler):
    """Accumulates per-call LLM usage and writes llm_call events."""

    def __init__(self, emitter: Optional[EventEmitter] = None):
        self.emitter = emitter
        self.totals = RunTotals()
        self.by_agent: Dict[str, AgentTotals] = {}
        self.current_node: str = "unknown"

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage, model = self._extract_usage(response)
        if usage is None:
            return
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        cached_in = int(usage.get("cache_read_input_tokens", 0) or 0)
        cost = self._cost(model, tokens_in, tokens_out, cached_in)

        self.totals.calls += 1
        self.totals.tokens_in += tokens_in
        self.totals.tokens_out += tokens_out
        self.totals.tokens_cached += cached_in
        self.totals.cost_usd += cost

        agent_totals = self.by_agent.setdefault(self.current_node, AgentTotals())
        agent_totals.calls += 1
        agent_totals.tokens_in += tokens_in
        agent_totals.tokens_out += tokens_out
        agent_totals.tokens_cached += cached_in
        agent_totals.cost_usd += cost

        if self.emitter is not None:
            self.emitter.emit(
                EventType.LLM_CALL,
                node=self.current_node,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cached_in=cached_in,
                cost_usd=round(cost, 6),
            )

    @staticmethod
    def _extract_usage(response: Any):
        # LangChain LLMResult: .llm_output is a dict; individual generations
        # may also have response_metadata. Try both.
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            out = response.llm_output
            return out.get("usage"), out.get("model_name", "unknown")
        return None, "unknown"

    @staticmethod
    def _cost(model: str, tokens_in: int, tokens_out: int, cached_in: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        uncached_in = max(tokens_in - cached_in, 0)
        return (
            uncached_in * pricing.input_per_mtok / 1_000_000
            + cached_in * pricing.cached_input_per_mtok / 1_000_000
            + tokens_out * pricing.output_per_mtok / 1_000_000
        )

    def record_retry(self) -> None:
        self.totals.retries += 1

    def record_summarization(self) -> None:
        self.totals.summarizations += 1
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/reliability/test_cost_tracker.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/reliability/cost_tracker.py tests/reliability/test_cost_tracker.py
git commit -m "feat(reliability): add CostTracker callback with per-agent and per-run totals"
```

---

## Task 4: Retry wrapper with exponential backoff

**Files:**
- Create: `tradingagents/reliability/retry.py`
- Create: `tests/reliability/test_retry.py`

A function wrapper that retries transient errors with exponential backoff, honors Anthropic `retry-after` headers, and records retries as events.

- [ ] **Step 1: Write failing tests**

File: `tests/reliability/test_retry.py`

```python
import time

import pytest

from tradingagents.reliability.retry import (
    RetryPolicy,
    classify_error,
    with_retries,
    FATAL,
    RETRYABLE_RATE_LIMIT,
    RETRYABLE_TRANSIENT,
)
from tests.reliability._fakes import ScriptedLLM


class FakeRateLimit(Exception):
    def __init__(self, retry_after: float | None = None):
        super().__init__("429 too many requests")
        self.retry_after = retry_after
    status_code = 429


class FakeTimeout(Exception):
    pass


class FakeAuthError(Exception):
    status_code = 401


def test_classify_rate_limit():
    assert classify_error(FakeRateLimit()) == RETRYABLE_RATE_LIMIT


def test_classify_timeout():
    assert classify_error(FakeTimeout()) == RETRYABLE_TRANSIENT


def test_classify_auth_error_is_fatal():
    assert classify_error(FakeAuthError()) == FATAL


def test_retries_until_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    llm = ScriptedLLM([FakeTimeout(), FakeTimeout(), "ok"])
    policy = RetryPolicy(max_attempts_transient=4, base_delay_s=1.0)

    out = with_retries(lambda: llm.invoke("hi"), policy=policy)
    assert out == "ok"
    assert len(llm.calls) == 3
    assert sleeps == [1.0, 2.0]  # backoff 2^0, 2^1


def test_honors_retry_after_header(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    # Stub out jitter so the sleep matches exactly
    import tradingagents.reliability.retry as retry_mod
    monkeypatch.setattr(retry_mod, "_jitter", lambda x: x)

    llm = ScriptedLLM([FakeRateLimit(retry_after=17.0), "ok"])
    out = with_retries(lambda: llm.invoke("hi"))
    assert out == "ok"
    assert sleeps == [17.0]


def test_fatal_error_does_not_retry(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    llm = ScriptedLLM([FakeAuthError()])
    with pytest.raises(FakeAuthError):
        with_retries(lambda: llm.invoke("hi"))
    assert len(llm.calls) == 1


def test_exhaustion_raises_last_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    llm = ScriptedLLM([FakeTimeout()] * 10)
    policy = RetryPolicy(max_attempts_transient=3, base_delay_s=0.1)
    with pytest.raises(FakeTimeout):
        with_retries(lambda: llm.invoke("hi"), policy=policy)
    assert len(llm.calls) == 3
```

- [ ] **Step 2: Run tests — expect failures (ImportError)**

```bash
pytest tests/reliability/test_retry.py -v
```

- [ ] **Step 3: Implement retry wrapper**

File: `tradingagents/reliability/retry.py`

```python
"""Retry/backoff wrapper for LLM calls."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar


log = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorClass(str, Enum):
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"
    RETRYABLE_TRANSIENT = "retryable_transient"
    FATAL = "fatal"


FATAL = ErrorClass.FATAL
RETRYABLE_RATE_LIMIT = ErrorClass.RETRYABLE_RATE_LIMIT
RETRYABLE_TRANSIENT = ErrorClass.RETRYABLE_TRANSIENT


@dataclass
class RetryPolicy:
    max_attempts_rate_limit: int = 5
    max_attempts_transient: int = 4
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    jitter_frac: float = 0.1  # +/- 10%


def classify_error(err: BaseException) -> ErrorClass:
    status = getattr(err, "status_code", None) or getattr(err, "http_status", None)
    if status == 429:
        return RETRYABLE_RATE_LIMIT
    if status is not None and 500 <= status < 600:
        return RETRYABLE_TRANSIENT
    if status in (400, 401, 403, 404, 422):
        return FATAL

    name = type(err).__name__.lower()
    if "ratelimit" in name:
        return RETRYABLE_RATE_LIMIT
    if any(tok in name for tok in ("timeout", "connection", "transient", "internalserver")):
        return RETRYABLE_TRANSIENT
    if any(tok in name for tok in ("authentication", "permission", "badrequest", "validation")):
        return FATAL
    return FATAL  # unknown is fatal by design


def _jitter(delay: float, frac: float = 0.1) -> float:
    return delay * (1 + random.uniform(-frac, frac))


def _sleep_for(err: BaseException, attempt: int, policy: RetryPolicy) -> float:
    retry_after = getattr(err, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    raw = min(policy.base_delay_s * (2 ** (attempt - 1)), policy.max_delay_s)
    return _jitter(raw, policy.jitter_frac)


def with_retries(
    fn: Callable[[], T],
    *,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
) -> T:
    """Invoke fn with exponential-backoff retries on transient errors.

    `on_retry(err, attempt, sleep_s)` is called before sleeping, once per retry.
    Fatal errors are not retried and are re-raised immediately.
    """
    policy = policy or RetryPolicy()
    attempt = 0
    last_err: Optional[BaseException] = None

    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as err:  # noqa: BLE001 — we re-raise below
            last_err = err
            cls = classify_error(err)
            if cls is FATAL:
                raise
            max_attempts = (
                policy.max_attempts_rate_limit
                if cls is RETRYABLE_RATE_LIMIT
                else policy.max_attempts_transient
            )
            if attempt >= max_attempts:
                raise
            sleep_s = _sleep_for(err, attempt, policy)
            if on_retry is not None:
                on_retry(err, attempt, sleep_s)
            log.warning("retry attempt=%d cls=%s sleep=%.1fs err=%s", attempt, cls.value, sleep_s, err)
            time.sleep(sleep_s)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/reliability/test_retry.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/reliability/retry.py tests/reliability/test_retry.py
git commit -m "feat(reliability): add with_retries wrapper with rate-limit and transient backoff"
```

---

## Task 5: Wire CostTracker + retry wrapper into the LLM factory

**Files:**
- Modify: `tradingagents/llm_clients/factory.py`
- Modify: `tradingagents/llm_clients/base_client.py` (add retry/tracker hooks)
- Create: `tests/reliability/test_factory_integration.py`

Now hook the new pieces into the existing factory so every LLM client the graph creates is automatically wrapped.

- [ ] **Step 1: Read existing base_client.py first**

```bash
cat tradingagents/llm_clients/base_client.py
```

Understand its current structure before modifying. It's small.

- [ ] **Step 2: Write failing integration test**

File: `tests/reliability/test_factory_integration.py`

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

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
```

- [ ] **Step 3: Run — expect pass (callbacks already supported)**

```bash
pytest tests/reliability/test_factory_integration.py -v
```

The existing factory already passes `callbacks` through. This test **should pass immediately** on current main. If it does, that's fine — it's a regression guard. If not, the fix is to ensure `callbacks` is in every client's `_PASSTHROUGH_KWARGS`.

- [ ] **Step 4: Verify all provider clients pass `callbacks` through**

```bash
grep -n "callbacks" tradingagents/llm_clients/*.py
```

Confirm each provider client includes `"callbacks"` in its `_PASSTHROUGH_KWARGS`. If any don't, add it.

- [ ] **Step 5: Add retry-wrapping to `BaseLLMClient.invoke_with_retries`**

Create a new small method on `BaseLLMClient` that wraps `get_llm().invoke()` with our retry policy. This lets agent code opt in without LangChain-level changes.

Modify `tradingagents/llm_clients/base_client.py` — add at the bottom of the `BaseLLMClient` class:

```python
    def invoke_with_retries(self, input, config=None, **kwargs):
        """Invoke the underlying LLM with the reliability retry wrapper.

        Agent nodes should prefer this over calling get_llm().invoke() directly
        so retries and per-call observability are consistent.
        """
        from tradingagents.reliability.retry import with_retries, RetryPolicy
        policy: RetryPolicy = self.kwargs.get("retry_policy") or RetryPolicy()
        on_retry = self.kwargs.get("on_retry")
        llm = self.get_llm()
        return with_retries(
            lambda: llm.invoke(input, config=config, **kwargs),
            policy=policy,
            on_retry=on_retry,
        )
```

- [ ] **Step 6: Add test for invoke_with_retries**

Append to `tests/reliability/test_factory_integration.py`:

```python
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
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/reliability/test_factory_integration.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add tradingagents/llm_clients/base_client.py tests/reliability/test_factory_integration.py
git commit -m "feat(llm): add invoke_with_retries on BaseLLMClient"
```

---

## Task 6: Tier presets

**Files:**
- Create: `tradingagents/config/tiers.py`
- Create: `tests/config/test_tiers.py`

Three preset dicts (`cheap`, `balanced`, `max`) and a `apply_tier(config, tier)` helper.

- [ ] **Step 1: Write failing test**

File: `tests/config/test_tiers.py`

```python
import pytest

from tradingagents.config.tiers import TIERS, apply_tier, TierName


def test_three_tiers_exist():
    assert set(TIERS.keys()) == {"cheap", "balanced", "max"}


@pytest.mark.parametrize("tier", ["cheap", "balanced", "max"])
def test_tier_has_required_fields(tier):
    t = TIERS[tier]
    required = {
        "deep_think_llm",
        "quick_think_llm",
        "anthropic_effort",
        "max_debate_rounds",
        "max_risk_discuss_rounds",
        "per_call_input_token_ceiling",
    }
    assert required.issubset(t.keys())


def test_apply_tier_merges_into_config():
    base = {"llm_provider": "anthropic", "foo": "bar"}
    out = apply_tier(base, "cheap")
    assert out["foo"] == "bar"  # untouched
    assert out["deep_think_llm"] == "claude-haiku-4-5"
    assert out["quick_think_llm"] == "claude-haiku-4-5"
    assert out["tier"] == "cheap"


def test_apply_tier_does_not_mutate_input():
    base = {"llm_provider": "anthropic"}
    _ = apply_tier(base, "balanced")
    assert "tier" not in base  # original untouched


def test_apply_tier_rejects_unknown_tier():
    with pytest.raises(ValueError):
        apply_tier({}, "ludicrous")


def test_apply_tier_snapshot_cheap():
    # Snapshot guards against accidental edits to the cheap preset values.
    out = apply_tier({}, "cheap")
    assert out["deep_think_llm"] == "claude-haiku-4-5"
    assert out["quick_think_llm"] == "claude-haiku-4-5"
    assert out["anthropic_effort"] == "low"
    assert out["max_debate_rounds"] == 1
    assert out["max_risk_discuss_rounds"] == 1
    assert out["per_call_input_token_ceiling"] == 20_000
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/config/test_tiers.py -v
```

- [ ] **Step 3: Implement tiers.py**

File: `tradingagents/config/tiers.py`

```python
"""Tier presets for cost/quality trade-offs."""
from __future__ import annotations

from typing import Any, Dict, Literal

TierName = Literal["cheap", "balanced", "max"]


TIERS: Dict[str, Dict[str, Any]] = {
    "cheap": {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-haiku-4-5",
        "quick_think_llm": "claude-haiku-4-5",
        "anthropic_effort": "low",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "per_call_input_token_ceiling": 20_000,
        "prompt_caching": True,
        "lazy_summarization": True,
    },
    "balanced": {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-sonnet-4-6",
        "quick_think_llm": "claude-haiku-4-5",
        "anthropic_effort": "medium",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "per_call_input_token_ceiling": 40_000,
        "prompt_caching": True,
        "lazy_summarization": True,
    },
    "max": {
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-opus-4-6",
        "quick_think_llm": "claude-sonnet-4-6",
        "anthropic_effort": "high",
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 2,
        "per_call_input_token_ceiling": 80_000,
        "prompt_caching": True,
        "lazy_summarization": True,
    },
}


def apply_tier(config: Dict[str, Any], tier: str) -> Dict[str, Any]:
    """Return a new config dict with the tier preset merged in.

    The input config is not mutated. The `tier` field is recorded so
    downstream code can read which preset was applied.
    """
    if tier not in TIERS:
        raise ValueError(f"Unknown tier: {tier}. Expected one of {list(TIERS)}.")
    merged = dict(config)
    merged.update(TIERS[tier])
    merged["tier"] = tier
    return merged
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/config/test_tiers.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Add `tier` to DEFAULT_CONFIG for back-compat**

Modify `tradingagents/default_config.py`. Add one line (keep the file working for callers that don't pass `tier`):

```python
DEFAULT_CONFIG = {
    # ... existing keys unchanged ...
    "tier": None,  # if set, tier preset is applied at graph construction
}
```

- [ ] **Step 6: Commit**

```bash
git add tradingagents/config/tiers.py tests/config/test_tiers.py tradingagents/default_config.py
git commit -m "feat(config): add cheap/balanced/max tier presets and apply_tier helper"
```

---

## Task 7: Wire tier into TradingAgentsGraph

**Files:**
- Modify: `tradingagents/graph/trading_graph.py`
- Create: `tests/runs/test_tier_wiring.py`

The graph constructor should read `config["tier"]` and apply the preset before initializing clients.

- [ ] **Step 1: Write failing test**

File: `tests/runs/test_tier_wiring.py`

```python
from unittest.mock import patch

from tradingagents.default_config import DEFAULT_CONFIG


def test_graph_applies_tier_when_set():
    config = DEFAULT_CONFIG.copy()
    config["tier"] = "cheap"
    # Patch out actual LLM client construction — we only want to check config flow.
    with patch("tradingagents.graph.trading_graph.create_llm_client") as mock_create:
        mock_create.return_value.get_llm.return_value = None
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        # Use a no-op agents setup via selected_analysts=[] if possible,
        # or fall back to catching the first create_llm_client call.
        try:
            TradingAgentsGraph(config=config)
        except Exception:
            pass  # graph setup may fail without real clients; we care about the call args
        first_call = mock_create.call_args_list[0]
        kwargs = first_call.kwargs or first_call[1]
        assert kwargs.get("model") == "claude-haiku-4-5"
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/runs/test_tier_wiring.py -v
```

- [ ] **Step 3: Modify TradingAgentsGraph to apply tier**

Modify `tradingagents/graph/trading_graph.py` near the top of `__init__`, immediately after `self.config = config or DEFAULT_CONFIG.copy()` (or equivalent existing line — verify with grep):

```python
        # Apply tier preset if set — merges preset fields into config.
        if self.config.get("tier"):
            from tradingagents.config.tiers import apply_tier
            self.config = apply_tier(self.config, self.config["tier"])
```

- [ ] **Step 4: Run test — expect pass**

```bash
pytest tests/runs/test_tier_wiring.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tradingagents/graph/trading_graph.py tests/runs/test_tier_wiring.py
git commit -m "feat(graph): apply tier preset when config.tier is set"
```

---

## Task 8: Anthropic prompt caching

**Files:**
- Modify: `tradingagents/llm_clients/anthropic_client.py`
- Create: `tests/reliability/test_prompt_caching.py`

Inject `cache_control: {"type": "ephemeral"}` markers on the stable prefix of messages so Anthropic caches them between calls. The cached block must appear at the start of `messages`.

**Read before implementing**: Anthropic prompt caching requires the cached prefix to be identical across calls and at least 1024 tokens for Sonnet/Opus, 2048 for Haiku. We mark the system message and any message tagged with `metadata={"cache": True}`.

- [ ] **Step 1: Write failing test**

File: `tests/reliability/test_prompt_caching.py`

```python
from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.llm_clients.anthropic_client import inject_cache_markers


def test_system_message_gets_cache_control():
    msgs = [SystemMessage(content="stable analyst report here..." * 300),
            HumanMessage(content="what next?")]
    out = inject_cache_markers(msgs)
    # The system message content should now be a list of blocks with
    # cache_control on the last block.
    sys_content = out[0].content
    assert isinstance(sys_content, list)
    assert any(
        isinstance(b, dict) and b.get("cache_control", {}).get("type") == "ephemeral"
        for b in sys_content
    )
    # Human message is left alone
    assert isinstance(out[1].content, str)


def test_no_system_message_is_noop():
    msgs = [HumanMessage(content="hi")]
    out = inject_cache_markers(msgs)
    assert out == msgs
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/reliability/test_prompt_caching.py -v
```

- [ ] **Step 3: Implement inject_cache_markers and wire into NormalizedChatAnthropic.invoke**

Modify `tradingagents/llm_clients/anthropic_client.py`. Add the helper, then call it in `invoke()` before delegating to `super().invoke()`:

```python
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


def inject_cache_markers(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Mark the system message (if any) as cacheable via cache_control.

    Anthropic cached prefixes must be identical across calls. We mark the
    system message as the single cached block; analyst reports baked into
    the system prompt get ~70% input-token discount on repeat calls.
    """
    if not messages or not isinstance(messages[0], SystemMessage):
        return messages
    sys = messages[0]
    content = sys.content
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list):
        blocks = list(content)
        if blocks and isinstance(blocks[-1], dict):
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    else:
        return messages
    new_sys = SystemMessage(content=blocks, additional_kwargs=sys.additional_kwargs)
    return [new_sys, *messages[1:]]


_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output and prompt caching."""

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, list):
            input = inject_cache_markers(input)
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("anthropic", self.model)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/reliability/test_prompt_caching.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/anthropic_client.py tests/reliability/test_prompt_caching.py
git commit -m "feat(anthropic): inject cache_control on system message for prompt caching"
```

---

## Task 9: Context budgeter and lazy summarization

**Files:**
- Create: `tradingagents/reliability/budgeter.py`
- Create: `tests/reliability/test_budgeter.py`

Pre-flight check before an LLM call. Counts input tokens (via a cheap tokenizer estimate) and, if over the tier ceiling, compresses older debate turns with a Haiku summarizer call. Keeps the last N turns verbatim.

- [ ] **Step 1: Write failing tests**

File: `tests/reliability/test_budgeter.py`

```python
from unittest.mock import MagicMock

from tradingagents.reliability.budgeter import (
    ContextBudgeter,
    estimate_tokens,
    compress_history,
)


def test_estimate_tokens_rough_char_ratio():
    # ~4 chars per token is the rule of thumb we use.
    text = "a" * 4000
    assert 900 <= estimate_tokens(text) <= 1100


def test_budgeter_under_ceiling_passes_through():
    b = ContextBudgeter(ceiling=10_000, summarizer_client=None)
    msgs = [{"role": "system", "content": "short"},
            {"role": "user", "content": "hi"}]
    out = b.ensure_under_ceiling(msgs)
    assert out == msgs  # unchanged


def test_budgeter_over_ceiling_triggers_compression():
    # Build a history above the ceiling with a clear structure:
    # long system message + 6 debate turns of growing size
    msgs = [{"role": "system", "content": "x" * 40_000}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"turn {i}: " + "y" * 5000})

    stub_summary = "Prior debate compressed to 3 key points."
    summarizer = MagicMock()
    summarizer.invoke.return_value = MagicMock(content=stub_summary)

    b = ContextBudgeter(ceiling=15_000, summarizer_client=summarizer, keep_last_n=2)
    out = b.ensure_under_ceiling(msgs)

    # System message is preserved
    assert out[0]["content"].startswith("x" * 100)
    # A summary message replaces the middle turns
    assert any(stub_summary in (m["content"] if isinstance(m["content"], str) else "")
               for m in out)
    # The last 2 turns are kept verbatim
    assert out[-2]["content"].startswith("turn 4:")
    assert out[-1]["content"].startswith("turn 5:")
    summarizer.invoke.assert_called_once()


def test_summarization_failure_falls_back_to_truncation():
    msgs = [{"role": "system", "content": "x" * 40_000}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"turn {i}: " + "y" * 5000})

    summarizer = MagicMock()
    summarizer.invoke.side_effect = RuntimeError("summarizer down")
    b = ContextBudgeter(ceiling=15_000, summarizer_client=summarizer, keep_last_n=2)
    out = b.ensure_under_ceiling(msgs)
    # Fallback: hard truncation to system + last 2 turns, no summary inserted.
    assert len(out) == 3
    assert out[-1]["content"].startswith("turn 5:")
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/reliability/test_budgeter.py -v
```

- [ ] **Step 3: Implement budgeter**

File: `tradingagents/reliability/budgeter.py`

```python
"""Context size guard and lazy history summarization."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Rough token estimate — 4 chars per token for English. Accurate enough for
# a pre-flight budget check; we don't need exact.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str | List[Any]) -> int:
    if isinstance(text, list):
        return sum(estimate_tokens(_content_of(x)) for x in text)
    if not isinstance(text, str):
        text = str(text)
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _content_of(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    if hasattr(msg, "content"):
        return str(msg.content)
    return str(msg)


def _total_tokens(messages: List[Any]) -> int:
    return sum(estimate_tokens(_content_of(m)) for m in messages)


_SUMMARIZE_PROMPT = (
    "You are compressing prior debate turns for another agent to read. "
    "Preserve: the main claims, any numeric evidence cited, and disagreements. "
    "Omit pleasantries. Output at most 400 words. Debate turns:\n\n"
)


def compress_history(
    summarizer_client: Any,
    middle_messages: List[Any],
) -> str:
    """Call the summarizer on a list of messages, return a compact summary."""
    joined = "\n\n".join(f"[{m.get('role', 'msg')}]: {_content_of(m)}" for m in middle_messages)
    resp = summarizer_client.invoke(_SUMMARIZE_PROMPT + joined)
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)


@dataclass
class ContextBudgeter:
    ceiling: int
    summarizer_client: Optional[Any] = None
    keep_last_n: int = 4

    def ensure_under_ceiling(self, messages: List[Any]) -> List[Any]:
        """If messages exceed the ceiling, compress the middle turns.

        Layout after compression:
          [system (if present), summary_message, last_n turns...]

        If summarization itself fails, fall back to hard truncation:
          [system, last_n turns...]
        """
        if _total_tokens(messages) <= self.ceiling:
            return messages
        if not messages:
            return messages

        system_msgs: List[Any] = []
        tail = messages
        if isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system_msgs = [messages[0]]
            tail = messages[1:]
        elif hasattr(messages[0], "type") and messages[0].type == "system":
            system_msgs = [messages[0]]
            tail = messages[1:]

        if len(tail) <= self.keep_last_n:
            return messages  # nothing to compress

        to_compress = tail[: -self.keep_last_n]
        keep = tail[-self.keep_last_n :]

        if self.summarizer_client is None:
            return system_msgs + keep  # hard truncation

        try:
            summary = compress_history(self.summarizer_client, to_compress)
        except Exception as e:  # noqa: BLE001 — budgeter must never raise upward
            log.warning("summarization failed, falling back to truncation: %s", e)
            return system_msgs + keep

        summary_msg = {"role": "user", "content": f"[Prior turns compressed]\n{summary}"}
        return system_msgs + [summary_msg] + keep
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/reliability/test_budgeter.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/reliability/budgeter.py tests/reliability/test_budgeter.py
git commit -m "feat(reliability): add ContextBudgeter with lazy Haiku-based compression"
```

---

## Task 10: LangGraph SqliteSaver checkpointer wiring

**Files:**
- Create: `tradingagents/runs/checkpointer.py`
- Modify: `tradingagents/graph/setup.py` (make `compile()` take a checkpointer)
- Modify: `tradingagents/graph/trading_graph.py` (thread checkpointer + thread_id)
- Create: `tests/runs/test_checkpointer.py`

- [ ] **Step 1: Read langgraph checkpointer docs inline**

```bash
python -c "from langgraph.checkpoint.sqlite import SqliteSaver; help(SqliteSaver)" | head -40
```

Confirm the import path — in langgraph 1.x it is `langgraph.checkpoint.sqlite.SqliteSaver`. If the import fails, pip install `langgraph-checkpoint-sqlite`.

- [ ] **Step 2: Write failing test**

File: `tests/runs/test_checkpointer.py`

```python
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
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/runs/test_checkpointer.py -v
```

- [ ] **Step 4: Implement checkpointer adapter**

File: `tradingagents/runs/checkpointer.py`

```python
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
    return SqliteSaver(conn)


def thread_config(run_id: str) -> Dict[str, Any]:
    """Standard LangGraph thread config — keyed on run_id."""
    return {"configurable": {"thread_id": run_id}}
```

- [ ] **Step 5: Run test — expect pass**

```bash
pytest tests/runs/test_checkpointer.py -v
```

If `langgraph.checkpoint.sqlite` import fails, add to pyproject dev deps:
```toml
"langgraph-checkpoint-sqlite>=2.0.0",
```
and `pip install -e ".[dev]"` again.

- [ ] **Step 6: Thread checkpointer through GraphSetup.compile()**

Modify `tradingagents/graph/setup.py`. Change the method signature and the final return:

```python
def setup_graph(self, selected_analysts, checkpointer=None):
    # ... existing setup ...
    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
```

Do a grep first to find the exact existing signature and match it:
```bash
grep -n "def setup_graph\|workflow.compile" tradingagents/graph/setup.py
```

- [ ] **Step 7: Thread checkpointer through TradingAgentsGraph**

Modify `tradingagents/graph/trading_graph.py`:

1. Add `run_id: Optional[str] = None` and `checkpointer=None` to `__init__`.
2. Store them as `self.run_id` and `self.checkpointer`.
3. When calling `GraphSetup.setup_graph(...)`, pass `checkpointer=self.checkpointer`.
4. In `propagate()`, when calling `.invoke()` or `.stream()`, pass the thread config:
   ```python
   from tradingagents.runs.checkpointer import thread_config
   graph_args = self.propagator.get_graph_args()
   if self.checkpointer is not None and self.run_id:
       graph_args["config"] = {**graph_args.get("config", {}), **thread_config(self.run_id)}
   ```

- [ ] **Step 8: Commit**

```bash
git add tradingagents/runs/checkpointer.py tradingagents/graph/setup.py tradingagents/graph/trading_graph.py tests/runs/test_checkpointer.py
git commit -m "feat(runs): wire LangGraph SqliteSaver checkpointer per run_id"
```

---

## Task 11: Run orchestrator (new + resume lifecycle)

**Files:**
- Create: `tradingagents/runs/orchestrator.py`
- Create: `tests/runs/test_orchestrator.py`

High-level lifecycle: `start_run()` mints an id, writes the initial manifest, returns a context object; `resume_run(run_id)` rehydrates; `finish_run()` flips status; a signal handler flushes `interrupted` on Ctrl-C.

- [ ] **Step 1: Write failing tests**

File: `tests/runs/test_orchestrator.py`

```python
from pathlib import Path

from tradingagents.runs.orchestrator import start_run, resume_run, finish_run, list_runs
from tradingagents.runs.manifest import RunStatus, read_manifest


def test_start_run_writes_manifest(tmp_path: Path):
    ctx = start_run(
        ticker="AAPL",
        trade_date="2026-04-10",
        tier="cheap",
        config_snapshot={"x": 1},
        results_root=tmp_path,
    )
    assert (tmp_path / ctx.run_id / "run_manifest.json").exists()
    m = read_manifest(tmp_path / ctx.run_id / "run_manifest.json")
    assert m.status is RunStatus.RUNNING
    assert m.ticker == "AAPL"
    assert m.tier == "cheap"


def test_finish_run_marks_completed(tmp_path: Path):
    ctx = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    finish_run(ctx, status=RunStatus.COMPLETED)
    m = read_manifest(tmp_path / ctx.run_id / "run_manifest.json")
    assert m.status is RunStatus.COMPLETED


def test_resume_run_requires_interrupted(tmp_path: Path):
    ctx = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    finish_run(ctx, status=RunStatus.COMPLETED)
    import pytest
    with pytest.raises(ValueError, match="not resumable"):
        resume_run(ctx.run_id, results_root=tmp_path)


def test_resume_run_from_interrupted(tmp_path: Path):
    ctx = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    finish_run(ctx, status=RunStatus.INTERRUPTED, last_completed_node="trader")

    resumed = resume_run(ctx.run_id, results_root=tmp_path)
    assert resumed.run_id == ctx.run_id
    assert resumed.manifest.status is RunStatus.RUNNING  # flipped back
    assert resumed.manifest.last_completed_node == "trader"


def test_list_runs_sorted_newest_first(tmp_path: Path):
    import time
    ctx1 = start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path)
    time.sleep(0.01)
    ctx2 = start_run("NVDA", "2026-04-10", "cheap", {}, results_root=tmp_path)
    rows = list_runs(tmp_path)
    assert rows[0].run_id == ctx2.run_id
    assert rows[1].run_id == ctx1.run_id
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/runs/test_orchestrator.py -v
```

- [ ] **Step 3: Implement orchestrator**

File: `tradingagents/runs/orchestrator.py`

```python
"""Run lifecycle: start, resume, finish, list, interrupt handler."""
from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.runs.events import EventEmitter, EventType
from tradingagents.runs.manifest import (
    RunManifest,
    RunStatus,
    TokenTotals,
    mint_run_id,
    read_manifest,
    update_manifest,
    write_manifest,
)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    manifest: RunManifest
    emitter: EventEmitter


def _run_dir(results_root: Path | str, run_id: str) -> Path:
    return Path(results_root) / run_id


def start_run(
    ticker: str,
    trade_date: str,
    tier: str,
    config_snapshot: Dict[str, Any],
    results_root: Path | str = "results",
) -> RunContext:
    run_id = mint_run_id(ticker, trade_date)
    run_dir = _run_dir(results_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    manifest = RunManifest(
        run_id=run_id,
        ticker=ticker,
        trade_date=trade_date,
        tier=tier,
        started_at=now_iso,
        status=RunStatus.RUNNING,
        config_snapshot=config_snapshot,
        tokens=TokenTotals(),
    )
    write_manifest(manifest, run_dir / "run_manifest.json")
    emitter = EventEmitter(run_dir / "events.jsonl", run_id=run_id)
    emitter.emit(EventType.RUN_START, ticker=ticker, trade_date=trade_date, tier=tier)

    ctx = RunContext(run_id=run_id, run_dir=run_dir, manifest=manifest, emitter=emitter)
    _install_interrupt_handler(ctx)
    return ctx


def resume_run(run_id: str, results_root: Path | str = "results") -> RunContext:
    run_dir = _run_dir(results_root, run_id)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run manifest for {run_id} at {manifest_path}")
    manifest = read_manifest(manifest_path)
    if manifest.status is not RunStatus.INTERRUPTED:
        raise ValueError(f"Run {run_id} is not resumable (status={manifest.status.value})")
    manifest.status = RunStatus.RUNNING
    write_manifest(manifest, manifest_path)
    emitter = EventEmitter(run_dir / "events.jsonl", run_id=run_id)
    emitter.emit(EventType.RUN_START, resumed=True, from_node=manifest.last_completed_node)
    ctx = RunContext(run_id=run_id, run_dir=run_dir, manifest=manifest, emitter=emitter)
    _install_interrupt_handler(ctx)
    return ctx


def finish_run(
    ctx: RunContext,
    *,
    status: RunStatus,
    last_completed_node: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    manifest_path = ctx.run_dir / "run_manifest.json"
    update_manifest(
        manifest_path,
        status=status,
        last_completed_node=last_completed_node or ctx.manifest.last_completed_node,
        error=error,
    )
    ctx.emitter.emit(EventType.RUN_END, status=status.value)


@dataclass
class RunRow:
    run_id: str
    ticker: str
    tier: str
    status: str
    started_at: str
    cost_usd: float


def list_runs(results_root: Path | str = "results") -> List[RunRow]:
    root = Path(results_root)
    if not root.exists():
        return []
    rows: List[RunRow] = []
    for child in root.iterdir():
        mpath = child / "run_manifest.json"
        if not mpath.exists():
            continue
        try:
            m = read_manifest(mpath)
        except Exception:  # noqa: BLE001 — skip corrupt manifests
            continue
        rows.append(
            RunRow(
                run_id=m.run_id, ticker=m.ticker, tier=m.tier,
                status=m.status.value, started_at=m.started_at, cost_usd=m.cost_usd,
            )
        )
    rows.sort(key=lambda r: r.started_at, reverse=True)
    return rows


def _install_interrupt_handler(ctx: RunContext) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        try:
            update_manifest(
                ctx.run_dir / "run_manifest.json",
                status=RunStatus.INTERRUPTED,
            )
            ctx.emitter.emit(EventType.RUN_END, status="interrupted")
        finally:
            sys.exit(130)
    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        # signal only works on main thread; tests may not be on main thread
        pass
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/runs/test_orchestrator.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/runs/orchestrator.py tests/runs/test_orchestrator.py
git commit -m "feat(runs): add run lifecycle orchestrator with start/resume/finish/list"
```

---

## Task 12: CLI flags — `--tier`, `--resume`, `--list-runs`, `--dry-run`

**Files:**
- Modify: `cli/main.py`
- Create: `tests/runs/test_cli_flags.py`

Add the four flags and wire them into the existing CLI flow. Keep existing interactive flow working when the flags aren't provided.

- [ ] **Step 1: Read existing CLI entry**

```bash
sed -n '1,60p' cli/main.py
grep -n "^def \|@app.command" cli/main.py
```

Understand how typer commands are wired, where `run_analysis()` is called, where the config is assembled.

- [ ] **Step 2: Add `tier`, `resume`, `list_runs`, `dry_run` typer options**

Modify `cli/main.py`. Change the `analyze` command signature to accept new options, and add a separate `list_runs` command. Exact edit locations depend on the current shape — do a `grep -n '@app.command' cli/main.py` first.

```python
@app.command()
def analyze(
    tier: str = typer.Option(None, "--tier", help="Cost tier: cheap, balanced, or max"),
    resume: str = typer.Option(None, "--resume", help="Resume a prior interrupted run_id (or 'last')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use stub LLM; no API calls"),
):
    run_analysis(tier=tier, resume=resume, dry_run=dry_run)


@app.command("list-runs")
def list_runs_cmd():
    from tradingagents.runs.orchestrator import list_runs
    from rich.console import Console
    from rich.table import Table

    rows = list_runs("results")
    if not rows:
        typer.echo("No runs found.")
        return
    table = Table(title="Recent runs")
    table.add_column("run_id")
    table.add_column("ticker")
    table.add_column("tier")
    table.add_column("status")
    table.add_column("started_at")
    table.add_column("cost_usd", justify="right")
    for r in rows[:30]:
        table.add_row(r.run_id, r.ticker, r.tier, r.status, r.started_at, f"${r.cost_usd:.2f}")
    Console().print(table)
```

Modify `run_analysis(...)` to accept the new kwargs. Where the config is assembled, add:

```python
if tier:
    from tradingagents.config.tiers import apply_tier
    config = apply_tier(config, tier)
```

If `resume` is provided, skip the interactive prompts and call `resume_run(...)` instead of `start_run(...)`.

- [ ] **Step 3: Add test for `list-runs` command output**

File: `tests/runs/test_cli_flags.py`

```python
from pathlib import Path

from tradingagents.runs.orchestrator import start_run


def test_list_runs_cli_shows_started_run(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path / "results")
    from typer.testing import CliRunner
    from cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["list-runs"])
    assert result.exit_code == 0
    assert "AAPL" in result.output
    assert "cheap" in result.output
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/runs/test_cli_flags.py -v
```

- [ ] **Step 5: Commit**

```bash
git add cli/main.py tests/runs/test_cli_flags.py
git commit -m "feat(cli): add --tier, --resume, --dry-run flags and list-runs command"
```

---

## Task 13: Live CLI status panel

**Files:**
- Create: `tradingagents/runs/status_panel.py`
- Modify: `cli/main.py` (attach status panel to the graph stream loop)
- Create: `tests/runs/test_status_panel.py`

Rich-based live display wrapping the existing graph stream loop. Falls back to periodic log lines for non-TTY.

- [ ] **Step 1: Write failing unit test for the renderer**

File: `tests/runs/test_status_panel.py`

```python
from tradingagents.runs.status_panel import render_panel_text
from tradingagents.reliability.cost_tracker import RunTotals


def test_panel_renders_key_fields():
    totals = RunTotals(calls=37, tokens_in=837_800, tokens_out=88_700, tokens_cached=610_000, cost_usd=1.42)
    text = render_panel_text(
        run_id="2026-04-13_AAPL_a3f2", ticker="AAPL", tier="balanced",
        node="Conservative Analyst", elapsed_s=763,
        totals=totals, cache_hit_ratio=0.74, budget=3.0, last_error=None,
    )
    assert "AAPL" in text
    assert "tier=balanced" in text
    assert "Conservative Analyst" in text
    assert "$1.42" in text
    assert "$3.00" in text
    assert "74%" in text
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/runs/test_status_panel.py -v
```

- [ ] **Step 3: Implement render_panel_text (pure text, Rich-formatted)**

File: `tradingagents/runs/status_panel.py`

```python
"""Live CLI status panel and renderer for runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tradingagents.reliability.cost_tracker import RunTotals


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def render_panel_text(
    *,
    run_id: str,
    ticker: str,
    tier: str,
    node: str,
    elapsed_s: float,
    totals: RunTotals,
    cache_hit_ratio: float,
    budget: Optional[float],
    last_error: Optional[str],
) -> str:
    lines = []
    lines.append(f"[bold]{ticker}[/bold] · tier={tier} · run={run_id}")
    lines.append(f"Node: {node}    ⏱ {_fmt_elapsed(elapsed_s)}")
    lines.append(
        f"LLM: {totals.calls} calls · {totals.retries} retries · "
        f"{totals.summarizations} summarize"
    )
    lines.append(
        f"Tokens: {_fmt_tokens(totals.tokens_in)}↑ {_fmt_tokens(totals.tokens_out)}↓   "
        f"Cached: {int(cache_hit_ratio * 100)}%"
    )
    budget_str = f" / budget ${budget:.2f}" if budget is not None else ""
    lines.append(f"Cost: ${totals.cost_usd:.2f}{budget_str}")
    lines.append(f"Last error: {last_error or 'none'}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test — expect pass**

```bash
pytest tests/runs/test_status_panel.py -v
```

- [ ] **Step 5: Attach Live panel around existing graph stream loop**

In `cli/main.py`, locate the `for chunk in graph.graph.stream(...)` block inside `run_analysis()`. Wrap it:

```python
from rich.live import Live
from rich.panel import Panel
import time
import sys

is_tty = sys.stdout.isatty()

def _make_panel():
    return Panel(render_panel_text(
        run_id=ctx.run_id,
        ticker=ticker,
        tier=config.get("tier", "default"),
        node=current_node,
        elapsed_s=time.time() - start_ts,
        totals=cost_tracker.totals,
        cache_hit_ratio=(cost_tracker.totals.tokens_cached /
                         max(cost_tracker.totals.tokens_in, 1)),
        budget=None,
        last_error=last_error,
    ))

start_ts = time.time()
current_node = "starting"
last_error = None

if is_tty:
    with Live(_make_panel(), refresh_per_second=2) as live:
        for chunk in graph.graph.stream(init_agent_state, **args):
            # existing chunk processing...
            current_node = _extract_current_node(chunk) or current_node
            live.update(_make_panel())
else:
    for chunk in graph.graph.stream(init_agent_state, **args):
        # existing chunk processing...
        pass  # non-TTY: events.jsonl remains source of truth
```

Where `_extract_current_node(chunk)` is a small helper you add in the same file:

```python
def _extract_current_node(chunk):
    # LangGraph stream chunks can indicate the active node via keys
    if isinstance(chunk, dict) and "messages" in chunk:
        last = chunk["messages"][-1] if chunk["messages"] else None
        if last is not None and hasattr(last, "name") and last.name:
            return last.name
    return None
```

- [ ] **Step 6: Commit**

```bash
git add tradingagents/runs/status_panel.py cli/main.py tests/runs/test_status_panel.py
git commit -m "feat(runs): add live Rich status panel with TTY/non-TTY split"
```

---

## Task 14: run_metrics.json end-of-run writer

**Files:**
- Create: `tradingagents/runs/metrics.py`
- Create: `tests/runs/test_metrics.py`
- Modify: `tradingagents/runs/orchestrator.py` (call `write_run_metrics` from `finish_run`)

- [ ] **Step 1: Write failing test**

File: `tests/runs/test_metrics.py`

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

from tradingagents.reliability.cost_tracker import CostTracker, AgentTotals, RunTotals
from tradingagents.runs.metrics import write_run_metrics


def test_write_run_metrics_includes_per_agent(tmp_path: Path):
    tracker = CostTracker(emitter=None)
    tracker.totals = RunTotals(
        calls=42, tokens_in=142_300, tokens_out=38_720, tokens_cached=98_400,
        cost_usd=1.83, retries=2, summarizations=1,
    )
    tracker.by_agent["market_analyst"] = AgentTotals(
        calls=3, tokens_in=12_400, cost_usd=0.08,
    )

    write_run_metrics(
        path=tmp_path / "run_metrics.json",
        run_id="r1", ticker="AAPL", tier="balanced",
        duration_s=384, final_decision="BUY",
        tracker=tracker, latencies_ms=[200, 500, 1200, 3000],
        errors=[],
    )

    data = json.loads((tmp_path / "run_metrics.json").read_text())
    assert data["llm"]["calls_total"] == 42
    assert data["llm"]["cost_usd"] == 1.83
    assert data["llm"]["cache_hit_ratio"] > 0
    assert data["by_agent"]["market_analyst"]["calls"] == 3
    assert data["latency"]["p50_ms"] > 0
    assert data["final_decision"] == "BUY"
```

- [ ] **Step 2: Run — expect failure**

- [ ] **Step 3: Implement write_run_metrics**

File: `tradingagents/runs/metrics.py`

```python
"""End-of-run metrics writer."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.reliability.cost_tracker import CostTracker


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def write_run_metrics(
    *,
    path: Path | str,
    run_id: str,
    ticker: str,
    tier: str,
    duration_s: float,
    final_decision: Optional[str],
    tracker: CostTracker,
    latencies_ms: List[float],
    errors: List[Dict[str, Any]],
) -> None:
    totals = tracker.totals
    cache_hit_ratio = (totals.tokens_cached / totals.tokens_in) if totals.tokens_in else 0.0

    doc = {
        "run_id": run_id,
        "ticker": ticker,
        "tier": tier,
        "duration_s": duration_s,
        "final_decision": final_decision,
        "llm": {
            "calls_total": totals.calls,
            "retries": totals.retries,
            "summarizations": totals.summarizations,
            "tokens_in": totals.tokens_in,
            "tokens_out": totals.tokens_out,
            "cached_in": totals.tokens_cached,
            "cache_hit_ratio": round(cache_hit_ratio, 4),
            "cost_usd": round(totals.cost_usd, 4),
        },
        "latency": {
            "p50_ms": round(_percentile(latencies_ms, 0.5)),
            "p95_ms": round(_percentile(latencies_ms, 0.95)),
            "max_ms": round(max(latencies_ms)) if latencies_ms else 0,
        },
        "by_agent": {
            name: asdict(at) for name, at in tracker.by_agent.items()
        },
        "errors": errors,
    }
    Path(path).write_text(json.dumps(doc, indent=2))
```

- [ ] **Step 4: Run test — expect pass**

- [ ] **Step 5: Have `finish_run` call `write_run_metrics`**

Modify `tradingagents/runs/orchestrator.py`. Add optional params to `finish_run`:

```python
def finish_run(
    ctx: RunContext,
    *,
    status: RunStatus,
    last_completed_node: Optional[str] = None,
    error: Optional[str] = None,
    tracker: Optional["CostTracker"] = None,
    latencies_ms: Optional[List[float]] = None,
    final_decision: Optional[str] = None,
    duration_s: Optional[float] = None,
) -> None:
    manifest_path = ctx.run_dir / "run_manifest.json"
    update_manifest(manifest_path, status=status,
                    last_completed_node=last_completed_node or ctx.manifest.last_completed_node,
                    error=error)
    ctx.emitter.emit(EventType.RUN_END, status=status.value)
    if tracker is not None:
        from tradingagents.runs.metrics import write_run_metrics
        write_run_metrics(
            path=ctx.run_dir / "run_metrics.json",
            run_id=ctx.run_id, ticker=ctx.manifest.ticker, tier=ctx.manifest.tier,
            duration_s=duration_s or 0, final_decision=final_decision,
            tracker=tracker, latencies_ms=latencies_ms or [],
            errors=[{"message": error}] if error else [],
        )
```

- [ ] **Step 6: Commit**

```bash
git add tradingagents/runs/metrics.py tradingagents/runs/orchestrator.py tests/runs/test_metrics.py
git commit -m "feat(runs): write run_metrics.json with per-agent and latency stats"
```

---

## Task 15: Dry-run mode with stub LLM

**Files:**
- Create: `tradingagents/reliability/dry_run.py`
- Create: `tests/reliability/test_dry_run.py`

A stub LLM client that returns canned responses, selected via `--dry-run` CLI flag or `config["dry_run"] = True`.

- [ ] **Step 1: Write failing test**

File: `tests/reliability/test_dry_run.py`

```python
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
```

- [ ] **Step 2: Implement DryRunLLM**

File: `tradingagents/reliability/dry_run.py`

```python
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
```

- [ ] **Step 3: Wire into factory when `dry_run=True`**

Modify `tradingagents/llm_clients/factory.py`:

```python
def create_llm_client(
    provider: str, model: str, base_url: Optional[str] = None, **kwargs,
) -> BaseLLMClient:
    if kwargs.pop("dry_run", False):
        from tradingagents.reliability.dry_run import DryRunLLM

        class _DryClient(BaseLLMClient):
            def get_llm(self):
                return DryRunLLM()
            def validate_model(self) -> bool:
                return True
        return _DryClient(model, base_url, **kwargs)
    # ... existing branching unchanged ...
```

- [ ] **Step 4: Thread `dry_run` from CLI**

In `cli/main.py` `run_analysis(..., dry_run: bool = False)`, before creating the graph:

```python
if dry_run:
    config["dry_run"] = True
```

And have `TradingAgentsGraph.__init__` pass `dry_run=self.config.get("dry_run", False)` into its `create_llm_client(...)` calls.

- [ ] **Step 5: Run tests**

```bash
pytest tests/reliability/test_dry_run.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tradingagents/reliability/dry_run.py tradingagents/llm_clients/factory.py cli/main.py tests/reliability/test_dry_run.py
git commit -m "feat(reliability): add dry-run mode with stub LLM for CI and config checks"
```

---

## Task 16: End-to-end smoke test

**Files:**
- Create: `tests/integration/test_smoke.py`

One integration test that spins up `TradingAgentsGraph` in dry-run mode, runs a propagation, and asserts the full results layout is produced.

- [ ] **Step 1: Write the smoke test**

File: `tests/integration/test_smoke.py`

```python
"""End-to-end smoke test: dry-run propagation produces full results layout."""
from pathlib import Path

import pytest


@pytest.mark.integration
def test_dry_run_propagation_produces_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.runs.orchestrator import start_run, finish_run
    from tradingagents.runs.manifest import RunStatus
    from tradingagents.reliability.cost_tracker import CostTracker
    from tradingagents.runs.events import EventEmitter

    config = DEFAULT_CONFIG.copy()
    config["tier"] = "cheap"
    config["dry_run"] = True
    config["results_dir"] = str(tmp_path / "results")

    ctx = start_run(
        ticker="AAPL", trade_date="2026-04-10", tier="cheap",
        config_snapshot={}, results_root=tmp_path / "results",
    )
    tracker = CostTracker(emitter=ctx.emitter)

    try:
        graph = TradingAgentsGraph(
            debug=False, config=config,
            callbacks=[tracker], run_id=ctx.run_id,
        )
        _, decision = graph.propagate("AAPL", "2026-04-10")
    finally:
        finish_run(ctx, status=RunStatus.COMPLETED, tracker=tracker,
                   latencies_ms=[100, 200, 300], final_decision=str(decision)[:20])

    run_dir = tmp_path / "results" / ctx.run_id
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "run_metrics.json").exists()
```

- [ ] **Step 2: Run it**

```bash
pytest tests/integration/test_smoke.py -v
```

Expected: pass. If `TradingAgentsGraph` fails on something unrelated (e.g., data vendor API call during init), make the graph lazy-initialize data tools so dry-run doesn't need network. Document any shim applied.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_smoke.py
git commit -m "test(integration): end-to-end smoke test for dry-run propagation"
```

---

## Task 17: Full test suite + lint pass

- [ ] **Step 1: Run all tests**

```bash
pytest -v
```

Expected: all pass. Investigate and fix any that don't.

- [ ] **Step 2: Verify existing unittest suite still works**

```bash
python -m unittest discover tests
```

Expected: existing tests pass.

- [ ] **Step 3: Smoke-test the CLI**

```bash
source .venv/bin/activate
tradingagents list-runs
```

Expected: command exits cleanly (either shows a run or "No runs found").

```bash
tradingagents analyze --dry-run
```

Follow the prompts (or pipe scripted answers); expected: full run completes using DryRunLLM, produces `results/<run_id>/` with manifest, events.jsonl, and metrics.

- [ ] **Step 4: Commit any fixups**

```bash
git add -A
git commit -m "test: full suite green and CLI smoke verified" || true
```

---

## Task 18: Push and open PR on the fork

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/reliability-hardening
```

- [ ] **Step 2: Open PR against your fork's main**

```bash
gh pr create --base main --head feat/reliability-hardening \
  --title "Reliability & cost hardening (sub-project 1/4)" \
  --body "$(cat <<'EOF'
## Summary
- Retry wrapper with exponential backoff + Anthropic retry-after support
- LangGraph SqliteSaver checkpointer for resumable runs
- Tier presets (cheap / balanced / max) controlling model + effort + budgets
- Anthropic prompt caching on system message blocks
- ContextBudgeter with lazy Haiku-based history summarization
- Per-run observability: run_manifest.json, events.jsonl, run_metrics.json
- Live Rich CLI status panel
- Dry-run mode for CI and config checks
- New CLI flags: --tier, --resume, --dry-run, list-runs command

Spec: docs/superpowers/specs/2026-04-13-reliability-hardening-design.md

## Test plan
- [ ] `pytest -v` — all new tests green
- [ ] `python -m unittest discover tests` — existing tests still green
- [ ] `tradingagents analyze --dry-run` — produces full results layout
- [ ] Manual: trigger a simulated 429 during a real run (see test_retry.py)
- [ ] Manual: Ctrl-C a run, then `--resume <id>` completes it
EOF
)"
```

- [ ] **Step 3: Do not merge until each success criterion from the spec is verified**

From `docs/superpowers/specs/2026-04-13-reliability-hardening-design.md` Section "Success criteria":

1. A `cheap`-tier run completes end-to-end for under $1 and reports accurate cost in `run_metrics.json`.
2. A simulated 429 during a run does not kill the run.
3. A Ctrl-C during a run produces a resumable `interrupted` manifest; `--resume <run_id>` picks up and finishes.
4. A run that would have hit the old per-call token ceiling instead triggers lazy summarization and completes.
5. `events.jsonl`, `run_manifest.json`, and `run_metrics.json` are produced for every run.
6. All existing tests pass; new plumbing tests pass.
7. Running the CLI with no `--tier` flag is back-compat with today's behavior.

Check them off one by one in the PR description.

---

## Appendix: Files created/modified summary

**New files:**
- `tradingagents/reliability/__init__.py`
- `tradingagents/reliability/retry.py`
- `tradingagents/reliability/budgeter.py`
- `tradingagents/reliability/cost_tracker.py`
- `tradingagents/reliability/trace.py`
- `tradingagents/reliability/dry_run.py`
- `tradingagents/config/__init__.py`
- `tradingagents/config/tiers.py`
- `tradingagents/runs/__init__.py`
- `tradingagents/runs/manifest.py`
- `tradingagents/runs/events.py`
- `tradingagents/runs/checkpointer.py`
- `tradingagents/runs/orchestrator.py`
- `tradingagents/runs/metrics.py`
- `tradingagents/runs/status_panel.py`
- 10 new test files across `tests/reliability/`, `tests/runs/`, `tests/config/`, `tests/integration/`
- `tests/reliability/_fakes.py`

**Modified files:**
- `pyproject.toml` (dev deps, pytest config)
- `tradingagents/default_config.py` (add `tier` key)
- `tradingagents/llm_clients/factory.py` (dry-run branch)
- `tradingagents/llm_clients/anthropic_client.py` (prompt caching)
- `tradingagents/llm_clients/base_client.py` (`invoke_with_retries`)
- `tradingagents/graph/trading_graph.py` (tier, run_id, checkpointer, thread config)
- `tradingagents/graph/setup.py` (`compile(checkpointer=...)`)
- `cli/main.py` (flags, status panel wiring, list-runs command)

**Net estimate:** ~1700 LOC added, ~200 edited. 18 tasks, ~90 steps. 2–4 focused implementation sessions with subagent parallelization.
