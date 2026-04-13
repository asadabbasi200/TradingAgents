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

    Format: YYYY-MM-DD_TICKER_<6-char hex suffix>. 16M combinations keep
    collision probability negligible across a year of runs.
    """
    now = now or datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    suffix = secrets.token_hex(3)
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
