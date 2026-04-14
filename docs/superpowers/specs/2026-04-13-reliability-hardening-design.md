# Reliability & Cost Hardening — Design Spec

**Date:** 2026-04-13
**Sub-project:** #1 of 4 in the TradingAgents production-readiness roadmap
**Status:** Design approved, pending writing-plans
**Author:** brainstormed with Claude (superpowers:brainstorming)

---

## Context

TradingAgents today is an interesting research prototype but not production-usable as a signal source. A deep analysis in the previous session surfaced four independent sub-projects needed to get it there:

1. **Reliability & cost hardening** (this spec)
2. Data integrity & point-in-time correctness
3. Backtesting + evaluation harness
4. Covered-call strategy adapter

These have a dependency order: 1 → 2 → 3 → 4. This spec covers #1 only.

The immediate trigger is a real failure: a 35-minute run died at the Conservative Analyst step with `RateLimitError 429` (Anthropic Tier 1, 30k input tokens/minute, `claude-sonnet-4-6`). The root cause was not a transient blip — it was that debate stages re-send the full analyst report bundle plus growing history, so a single call's input exceeded the per-minute ceiling. There is no retry wrapper, no prompt caching, no token budgeting, and no resume mechanism. A dead run is simply lost work.

## Goals

1. A run survives transient failures (429s, 5xx, network blips, laptop sleep, Ctrl-C) and can resume from the last completed node, never restart from scratch.
2. A single-ticker run stays within a predictable cost envelope, controlled by a `tier` setting (`cheap` / `balanced` / `max`).
3. Cost, token usage, cache efficiency, retries, and latency are visible during a run and summarized after.
4. Unblocks sub-project #3 (backtesting) by making cheap, resumable, high-volume runs practical.

## Non-goals (explicitly deferred)

- **Look-ahead bias** in fundamentals/insider data → sub-project #2.
- **Parallel multi-ticker execution** and shared rate-limit budgets → post-#4 sub-project. Noted as eventual requirement; not built here.
- **Prompt quality fixes** — structured outputs, debate scoring, memory epistemology → later sub-projects.
- **Provider expansion** beyond Anthropic-first. The existing factory keeps working for OpenAI/Google/xAI/OpenRouter/Ollama, but optimization effort (caching, budgeting) targets Anthropic first.
- **External tracing** (LangSmith, OpenTelemetry). A no-op trace stub is installed so this can be added later as a flag; no integration work in this sub-project.

## Architecture

```
CLI (tradingagents --tier balanced  [--resume <run_id>])
        ↓
Run Orchestrator (NEW)
   - mints run_id, picks tier preset, wires checkpointer,
     writes run_manifest.json, starts or resumes graph
        ↓
TradingAgentsGraph (existing, minimal edits)
        ↓
LLM Factory (existing, wrapped)
   ├── Retry wrapper          ← exponential backoff + jitter,
   │                            respects Anthropic retry-after headers
   ├── Context Budgeter       ← pre-invoke check; if tokens > tier limit,
   │                            calls Haiku summarizer on old debate history
   ├── Prompt Cache markers   ← cache_control:ephemeral on stable blocks
   └── Cost Tracker callback  ← running totals; written to manifest
        ↓
Provider client (Anthropic-first; others keep working but unoptimized)
```

### New modules

All under `tradingagents/`:

| File | Purpose | Est. size |
|---|---|---|
| `reliability/retry.py` | `with_retries()` wrapper around LLM invoke | ~80 lines |
| `reliability/budgeter.py` | `ContextBudgeter` — token count + lazy summarization | ~150 lines |
| `reliability/cost_tracker.py` | LangChain callback → running usage/cost | ~100 lines |
| `reliability/trace.py` | No-op tracer stub (future LangSmith/OTel hook) | ~30 lines |
| `config/tiers.py` | `cheap` / `balanced` / `max` presets | ~60 lines |
| `runs/manifest.py` | `run_id`, `RunManifest`, JSON persist | ~100 lines |
| `runs/checkpointer.py` | LangGraph `SqliteSaver` adapter + thread wiring | ~60 lines |
| `runs/orchestrator.py` | Run lifecycle — new / resume / interrupt | ~120 lines |

### Existing files touched (surgical edits only)

- `llm_clients/factory.py` — wrap every client with retry + cost callback
- `llm_clients/anthropic_client.py` — inject `cache_control` on stable prompt blocks
- `graph/trading_graph.py` — thread `run_id`, pass checkpointer to `compile()`, pass `thread_id` to `.invoke()/.stream()`
- `cli/main.py` — `--tier`, `--resume`, `--list-runs`, `--dry-run` flags; live status panel
- `default_config.py` — new `tier` key; existing keys untouched for back-compat

### Key design rules

- **Retry wrapper** handles ONLY transient errors (429, 5xx, timeouts, `ConnectionError`). It never retries 400-class validation errors — those are bugs, not blips.
- **Context budgeter** runs **before** the provider call, not as a reaction to failure. Pre-flight, not recovery.
- **Summarization is lazy**: only triggered when a projected call exceeds the tier's per-call token ceiling. Small runs never pay the summarization tax.
- **Checkpointing** is per-node (LangGraph's default granularity), persisted to `results/<run_id>/checkpoint.sqlite`.
- **Cost tracking** is additive — the tracker callback is installed on every LLM client from the factory, so nothing in graph code needs to change.

## Tier presets

A single `tier` config key drives everything. Three presets, selectable via CLI (`--tier cheap|balanced|max`), config file, or Python API. Users can still override individual fields; `tier` is just a preset map.

| Knob | `cheap` | `balanced` | `max` |
|---|---|---|---|
| Deep-think model | `claude-haiku-4-5` | `claude-sonnet-4-6` | `claude-opus-4-6` |
| Quick-think model | `claude-haiku-4-5` | `claude-haiku-4-5` | `claude-sonnet-4-6` |
| `anthropic_effort` | `low` | `medium` | `high` |
| `max_debate_rounds` | 1 | 1 | 2 |
| `max_risk_discuss_rounds` | 1 | 1 | 2 |
| Per-call input token ceiling | 20k | 40k | 80k |
| Prompt caching | on | on | on |
| Lazy history summarization | on | on | on |
| Target cost/run (single ticker) | ~$0.25–0.75 | ~$1.50–3 | ~$5–8 |

The per-call ceilings are set below realistic Anthropic tier TPM limits so prompt-size-driven 429s are impossible: 20k on `cheap` leaves headroom on Tier 1 (30k/min); 40k on `balanced` targets Tier 2+; 80k on `max` targets Tier 3+.

## Run manifest & resume UX

### Results layout

```
results/
  <run_id>/                       # e.g., 2026-04-13_AAPL_a3f2
    run_manifest.json             # config, tier, status, timestamps, cost
    checkpoint.sqlite             # LangGraph SqliteSaver state
    events.jsonl                  # structured event stream
    run_metrics.json              # end-of-run summary
    reports/                      # existing agent reports
    final_state.json              # existing full state dump
```

Each run is self-contained: easy to archive, delete, or share. The backtest harness (#3) can blow away a run dir without affecting others. No global schema to migrate as the system evolves.

### `run_manifest.json` shape

```json
{
  "run_id": "2026-04-13_AAPL_a3f2",
  "ticker": "AAPL",
  "trade_date": "2026-04-10",
  "tier": "balanced",
  "started_at": "2026-04-13T09:14:22Z",
  "status": "running",
  "last_completed_node": "trader",
  "config_snapshot": { },
  "cost_usd": 1.83,
  "tokens": { "input": 142300, "output": 38720, "cached": 98400 },
  "error": null
}
```

`status` transitions: `running` → `completed | failed | interrupted`. A `completed` run cannot be resumed; an `interrupted` run can.

### Resume commands

- `tradingagents --resume <run_id>` — rehydrates checkpointer thread, resumes from `last_completed_node`.
- `tradingagents --resume last` — resumes most recent interrupted run.
- `tradingagents --list-runs` — table of recent runs with status, cost, ticker, date.

Ctrl-C is handled by a clean interrupt handler that flushes the manifest to `status=interrupted` before exit, so resume is always possible.

## Observability (tier D details)

### Event stream — `events.jsonl`

Append-only, flushed on each event. One JSON object per line. Event types:

- `run_start`, `run_end`
- `node_start`, `node_end`
- `llm_call` (with `tokens_in`, `tokens_out`, `cached_in`, `latency_ms`, `cost_usd`, `attempt`, `cache_hit`)
- `tool_call` (with `tool`, `latency_ms`)
- `retry` (with `error`, `attempt`, `sleep_s`)
- `summarize` (with `turns_compressed`, `tokens_before`, `tokens_after`)
- `error` (with `error_class`, `fatal`)

Example:

```json
{"ts":"2026-04-13T09:14:25Z","run_id":"2026-04-13_AAPL_a3f2","type":"llm_call","node":"market_analyst","model":"claude-haiku-4-5","tokens_in":4120,"tokens_out":890,"cached_in":3200,"latency_ms":2840,"cost_usd":0.012,"attempt":1}
{"ts":"2026-04-13T09:14:38Z","run_id":"2026-04-13_AAPL_a3f2","type":"retry","node":"conservative","error":"rate_limit_error","attempt":2,"sleep_s":12.4}
{"ts":"2026-04-13T09:14:52Z","run_id":"2026-04-13_AAPL_a3f2","type":"summarize","node":"aggressive","turns_compressed":4,"tokens_before":31200,"tokens_after":9800}
```

### Live CLI status panel

Rich-based live display, extending the existing status header:

```
╭─ AAPL · tier=balanced · run=2026-04-13_AAPL_a3f2 ────────────────╮
│ Node: Conservative Analyst     ⏱ 12:43                           │
│ Progress: 8/12 agents · 6/7 reports                              │
│ LLM: 37 calls · 23 tools · 2 retries · 1 summarize               │
│ Tokens: 837.8k↑ 88.7k↓   Cached: 74% ($0.18 saved)               │
│ Cost: $1.42 / budget $3.00                                       │
│ Last error: none                                                 │
╰──────────────────────────────────────────────────────────────────╯
```

Non-TTY runs (CI, future backtest harness, piped output) fall back to periodic one-line progress logs. `events.jsonl` remains the source of truth in all modes.

### End-of-run `run_metrics.json`

```json
{
  "run_id": "...",
  "ticker": "AAPL",
  "tier": "balanced",
  "duration_s": 384,
  "final_decision": "BUY",
  "llm": {
    "calls_total": 42,
    "retries": 2,
    "summarizations": 1,
    "tokens_in": 142300,
    "tokens_out": 38720,
    "cached_in": 98400,
    "cache_hit_ratio": 0.74,
    "cost_usd": 1.83
  },
  "latency": { "p50_ms": 2840, "p95_ms": 12400, "max_ms": 31200 },
  "by_agent": {
    "market_analyst":   {"calls": 3, "cost": 0.08, "tokens_in": 12400},
    "conservative":     {"calls": 4, "cost": 0.21, "tokens_in": 28100}
  },
  "errors": [{"node": "conservative", "class": "RateLimitError", "fatal": false}]
}
```

This file is what sub-project #3's backtest harness will aggregate across runs.

### Tracing stub (deferred)

`tradingagents/reliability/trace.py` ships as a no-op tracer with a TODO for LangSmith/OpenTelemetry. Enabling external tracing later becomes a config flag, not a refactor. This keeps the door open for sub-project #3's "replay a run in a web UI" use case without building it now.

## Error handling

### Retry policy

| Error class | Action |
|---|---|
| `RateLimitError` (429) | Honor `retry-after` header if present; else exponential backoff 2→4→8→16→32s; max 5 attempts |
| `APITimeoutError`, `APIConnectionError`, `InternalServerError` (5xx) | Exponential backoff; max 4 attempts |
| `BadRequestError` (400) when prompt too long | **No retry.** Trigger summarization once, re-attempt exactly once. If still fails, raise. |
| `AuthenticationError` (401), `PermissionDeniedError` (403) | No retry. Fatal. |
| `ValidationError` from our own schema check | No retry. Bug, not blip. Fatal. |
| Anything unknown | No retry. Fatal with full traceback. |

Every retry or fatal decision emits an `events.jsonl` line so post-mortems are possible.

### Failure modes

1. **Single 429 mid-run** → retry wrapper absorbs it, event logged, run continues.
2. **Sustained 429s (5 consecutive)** → run pauses, manifest → `interrupted`, user resumes later.
3. **Prompt over per-call ceiling** → budgeter triggers summarization *before* the call. If summarization itself fails, fall back to hard truncation (keep last N turns) and log a warning.
4. **Network drop / laptop sleep** → checkpointer has the last completed node persisted; resume picks up from there.
5. **Ctrl-C** → signal handler flushes manifest and checkpoint before exit.
6. **Fatal error** → manifest → `failed`, error recorded, no partial resume offered.
7. **Duplicate `run_id`** → run_id includes a 4-char random suffix; collision is astronomically unlikely.

### Summarization loop guard

The summarizer is itself an LLM call. To prevent infinite recursion:

- Summarizer calls use a hard-coded `claude-haiku-4-5` client with its own token ceiling of 8k input / 1k output.
- Summarizer clients are excluded from the context budgeter.
- If a summarization call itself fails, fall back to hard truncation; never recurse.

## Testing

Focused on plumbing, not on agent reasoning quality (which is sub-project #3's scorecard).

| What | How |
|---|---|
| Retry wrapper | Unit tests with fake LLM client raising scripted exceptions; assert backoff schedule, attempt counts, event emission |
| Context budgeter | Unit tests with synthetic messages of known token counts; assert summarization triggers exactly at ceiling |
| Cost tracker callback | Unit test: feed known token counts, assert running totals and per-agent breakdown |
| Tier presets | Snapshot test on the preset dicts (catches accidental edits) |
| Run manifest lifecycle | Unit test: `running` → `interrupted` → `resumed` → `completed` state transitions |
| Checkpointer resume | Integration test with a 3-node dummy graph: interrupt, resume, assert continuation from correct node |
| End-to-end smoke | One Haiku-only run against recorded API responses (`responses` library cassettes), assert manifest + events.jsonl correct |
| Dry-run mode | `--dry-run` flag exercises the full graph against a stub LLM returning canned responses; runs in CI without spending money |

## Rollout & back-compat

### Back-compat rules

- `default_config.py` keeps every existing key. `tier` is additive — unset falls back to today's defaults.
- `TradingAgentsGraph(config=...)` constructor unchanged. Optional `run_id=None` parameter added; if `None`, one is minted.
- CLI gains `--tier`, `--resume`, `--list-runs`, `--dry-run`. All existing flags keep working.
- Existing `./results/<TICKER>/<DATE>/` output still produced. New `results/<run_id>/` is additive for one sub-project; cutover handled later.
- `propagate()` return value unchanged.

### Branching

All work on a feature branch on the user's fork:

```
git checkout -b feat/reliability-hardening
```

PRs against `main` on the fork. `upstream/main` (TauricResearch) never touched. Changes worth upstreaming can be cherry-picked back later.

### Implementation sequencing (refined in writing-plans)

1. **Foundation** — `run_id`, `RunManifest`, results layout. No behavior change.
2. **Cost tracker + events.jsonl** — immediate visibility without changing behavior.
3. **Retry wrapper** — fixes the 429 crash. Biggest immediate win.
4. **Tier presets + CLI flag** — `--tier` usable.
5. **Prompt caching** — Anthropic `cache_control` on stable blocks. ~70% cost drop on cached stages.
6. **Context budgeter + lazy summarization** — "never crash from prompt size."
7. **Checkpointer + resume** — "never lose 35 minutes."
8. **Live CLI panel + run_metrics.json** — polish observability surface.
9. **Dry-run mode + integration tests** — CI-friendly test harness.

Each step is independently mergeable and leaves the system in a working state. Stopping after step 3 already yields a meaningfully better tool; steps 4–9 compound.

### Effort estimate

Roughly 1500–2000 LOC net add, ~300 LOC edited in existing files. 2–4 focused implementation sessions depending on subagent parallelization.

## Risks

- **LangGraph checkpointer API** can be finicky with the existing `compile()` call in `trading_graph.py`. May require a small refactor to thread the saver in cleanly. Risk, not blocker.
- **Anthropic prompt caching** requires cached content at the start of the `messages` array. Analyst prompts are currently assembled ad-hoc. May need to restructure prompt construction in `agents/utils/` to put stable blocks first.
- **Summarization feedback loop** — summarization uses LLM calls to reduce tokens for other LLM calls. Loop guard (dedicated client, excluded from budgeter, no recursion) mitigates.
- **SqliteSaver concurrency** — not a concern for sub-project #1 (single-run), flagged for the eventual parallel-safe sub-project (C in the original scoping).

## Success criteria

Sub-project #1 is done when:

1. A `cheap`-tier run completes end-to-end for under $1 and reports accurate cost in `run_metrics.json`.
2. A simulated 429 during a run does not kill the run (absorbed by retry wrapper).
3. A Ctrl-C during a run produces a resumable `interrupted` manifest; `--resume <run_id>` picks up from the last completed node and finishes the run.
4. A run that would have hit the old per-call token ceiling instead triggers lazy summarization and completes.
5. `events.jsonl`, `run_manifest.json`, and `run_metrics.json` are produced for every run and contain the fields specified above.
6. All existing tests pass; new tests for retry, budgeter, cost tracker, and resume pass.
7. Running the CLI with no `--tier` flag produces output indistinguishable from today's (back-compat).
