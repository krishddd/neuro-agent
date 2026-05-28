"""Phase 5.4 / Module 1 — Background LightRAG graph-build worker.

Why this module exists
----------------------
``LightRAG.insert(chunks)`` calls the LLM (qwen3:14b) on every chunk to
extract entities and relations. Synchronous ingestion of a single patient
corpus balloons from ~10 s (Chroma) to several minutes — pushing the
end-to-end pipeline past 30 min.

Mitigation
----------
A single-process ``ThreadPoolExecutor(max_workers=1)`` serialises LightRAG
inserts (LLM-bound, one-at-a-time avoids Ollama contention). The hot path
in ``recist_agent.index_rag()`` performs the Chroma upsert synchronously
(fast) then submits LightRAG insert to the worker and returns
immediately. Each per-patient graph dir carries a sentinel
``.build_status.json`` with one of:

    {"status": "building", "started_at": "..."}
    {"status": "ready",    "finished_at": "...", "n_chunks": int}
    {"status": "failed",   "finished_at": "...", "error": str}

``pharma_agent`` (and ``rag_penalty._lightrag_lookup``) check this file
before issuing a hybrid query: if ``building``, fall back to ChromaDB for
this run (graceful degrade). On the next run for the same patient the
graph is ready and hybrid retrieval kicks in.

On orchestrator shutdown ``flush(timeout)`` waits up to ``LIGHTRAG_FLUSH_TIMEOUT_S``
seconds for inflight builds; partial graphs are append-capable on next run.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# Module-level singleton — ``max_workers=1`` so LLM calls serialise.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
# Track inflight futures keyed by patient_id so flush() can wait on them.
_inflight: dict[str, Future] = {}
_inflight_lock = threading.Lock()

SENTINEL_FILENAME = ".build_status.json"


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="lightrag-builder",
            )
        return _executor


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Sentinel I/O (used by both worker + readers) ──────────────────────────────

def write_status(working_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically write the sentinel JSON. Best-effort; never raises."""
    try:
        working_dir.mkdir(parents=True, exist_ok=True)
        path = working_dir / SENTINEL_FILENAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("graph_worker: status write failed for %s: %s", working_dir, exc)


def read_status(working_dir: Path) -> dict[str, Any]:
    """Read the sentinel JSON. Returns ``{"status": "absent"}`` if missing."""
    path = Path(working_dir) / SENTINEL_FILENAME
    if not path.exists():
        return {"status": "absent"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "absent"}


def is_ready(working_dir: Path) -> bool:
    """True iff the LightRAG graph at ``working_dir`` finished a build."""
    return read_status(working_dir).get("status") == "ready"


def is_building(working_dir: Path) -> bool:
    return read_status(working_dir).get("status") == "building"


# ── Submission ────────────────────────────────────────────────────────────────

def submit_build(
    patient_id: str,
    working_dir: Path,
    build_fn: Callable[[], int],
) -> Future:
    """Submit a LightRAG build job to the background worker.

    ``build_fn`` is a zero-arg callable that performs the actual
    ``lightrag.insert(chunks)`` and returns the chunk count. The worker
    wraps it in sentinel-status updates and exception handling.

    Returns the ``Future`` so callers may wait on it (e.g. flush() at
    shutdown). The same ``patient_id`` submitted twice will queue serially
    — LightRAG is append-capable so this is safe.
    """
    working_dir = Path(working_dir)
    write_status(working_dir, {"status": "building", "started_at": _now_iso()})

    def _run() -> int:
        try:
            n = int(build_fn() or 0)
            write_status(working_dir, {
                "status": "ready",
                "finished_at": _now_iso(),
                "n_chunks": n,
            })
            log.info("graph_worker: %s LightRAG build complete (%d chunks)", patient_id, n)
            return n
        except Exception as exc:
            write_status(working_dir, {
                "status": "failed",
                "finished_at": _now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            })
            log.error("graph_worker: %s LightRAG build failed: %s", patient_id, exc)
            raise

    fut = _get_executor().submit(_run)
    with _inflight_lock:
        _inflight[patient_id] = fut

    def _drop(_):
        with _inflight_lock:
            if _inflight.get(patient_id) is fut:
                _inflight.pop(patient_id, None)
    fut.add_done_callback(_drop)
    return fut


# ── Shutdown ──────────────────────────────────────────────────────────────────

def flush(timeout: float | None = None) -> int:
    """Wait up to ``timeout`` seconds for inflight builds.

    Returns the number of futures that were still running when the
    timeout expired (0 = clean shutdown). Inflight builds keep running
    in the background even after timeout; their sentinel files transition
    from ``building`` → ``ready``/``failed`` opportunistically.
    """
    with _inflight_lock:
        futs = list(_inflight.values())
    if not futs:
        return 0
    log.info("graph_worker: flushing %d inflight LightRAG build(s) (timeout=%s)",
             len(futs), timeout)
    pending = 0
    for fut in futs:
        try:
            fut.result(timeout=timeout)
        except Exception:
            pending += 1
    return pending


def shutdown(wait: bool = False, timeout: float | None = None) -> None:
    """Tear down the executor. Idempotent."""
    global _executor
    with _executor_lock:
        if _executor is None:
            return
        try:
            if wait:
                flush(timeout=timeout)
            _executor.shutdown(wait=wait)
        finally:
            _executor = None


__all__ = [
    "submit_build", "flush", "shutdown",
    "read_status", "write_status", "is_ready", "is_building",
    "SENTINEL_FILENAME",
]
