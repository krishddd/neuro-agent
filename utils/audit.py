"""Append-only HIPAA-safe audit log.

Rules:
- Never write PHI, free-text content, filenames, or question text.
- Patient identifiers are hashed with a salt.
- Each event is one JSON line: {ts, stage, pid_hash, status, duration_ms, meta}.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..config import AUDIT_LOG, HASH_SALT


def hash_pid(pid: str) -> str:
    h = hashlib.sha256((HASH_SALT + "::" + pid).encode()).hexdigest()
    return h[:16]


def log(
    stage: str,
    pid: str | None = None,
    status: str = "ok",
    duration_ms: int | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "pid_hash": hash_pid(pid) if pid else None,
        "status": status,
        "duration_ms": duration_ms,
        "meta": _scrub(meta or {}),
    }
    path = Path(AUDIT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


_ALLOWED_META_KEYS = {
    "phase", "tool", "steps", "tokens", "model", "retries",
    "n_files", "n_chunks", "n_lesions", "severity",
    "urgency", "confidence", "ok", "error_type",
    # Access audit keys
    "role", "caller_name", "endpoint", "method", "action",
    "session_id", "channel",
}


def _scrub(meta: dict[str, Any]) -> dict[str, Any]:
    """Whitelist meta keys to prevent accidental PHI leakage."""
    return {k: v for k, v in meta.items() if k in _ALLOWED_META_KEYS}


def log_access(
    pid: str,
    role: str,
    caller_name: str,
    action: str,
    endpoint: str = "",
    meta: dict[str, Any] | None = None,
) -> None:
    """Log who accessed which patient's data and via what endpoint."""
    log(
        stage="access",
        pid=pid,
        status="ok",
        meta={
            "role": role,
            "caller_name": caller_name,
            "action": action,
            "endpoint": endpoint,
            **(meta or {}),
        },
    )


class stage_timer:
    """Context manager: audits stage start/finish with duration."""

    def __init__(self, stage: str, pid: str | None = None, **meta: Any):
        self.stage = stage
        self.pid = pid
        self.meta = meta
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = int((time.perf_counter() - self.t0) * 1000)
        status = "ok" if exc_type is None else "error"
        meta = dict(self.meta)
        if exc_type is not None:
            meta["error_type"] = exc_type.__name__
        log(self.stage, self.pid, status=status, duration_ms=dur, meta=meta)
        return False
