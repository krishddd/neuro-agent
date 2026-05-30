"""Background segmentation worker (Task 5).

3D nnU-Net inference on CPU runs 10-20 min/patient. Running that inline in
the FastAPI request thread blocks /run until timeout. This module submits
volumetric jobs to a single-worker ``ProcessPoolExecutor`` so the
orchestrator keeps moving — the RECIST/RANO path uses the 2D fallback
while the job runs, and when the volumetric result lands it overwrites
``S04c_volumetric.json`` on disk.

GPU path: if ``torch.cuda.is_available()`` AND ``SEG_WORKER_BACKEND=inline``,
we run in-process (inference finishes in <60s).

Celery backend is not implemented here — the hook is in place (selected
via ``SEG_WORKER_BACKEND=celery``) and raises ``NotImplementedError`` with
a pointer to configure the Redis URL.
"""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ..config import SEG_WORKER_BACKEND

log = logging.getLogger(__name__)

# Single-slot executor (segmentation is RAM-heavy; one job at a time).
_executor: Optional[ProcessPoolExecutor] = None
_jobs: dict[str, Future] = {}


def _gpu_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1)
    return _executor


def _worker_entrypoint(path: str, backend: str | None,
                       output_json: str,
                       mask_out_path: str | None = None) -> dict[str, Any]:
    """Called inside the worker process. Runs segmentation and persists JSON.

    If ``mask_out_path`` is provided, the segmentation backend also writes a
    NIfTI mask there (Phase 5.1 — required for PyRadiomics feature extraction).
    """
    # Re-import inside the child to get a clean module state.
    from .volumetric_seg import segment_path  # type: ignore
    result = segment_path(
        Path(path), backend=backend,
        mask_out_path=(Path(mask_out_path) if mask_out_path else None),
    )
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning("seg_worker: write failed for %s (%s)", out, exc)
    return result


def submit(
    volume_path: Path,
    output_json: Path,
    backend: str | None = None,
    inline_override: bool | None = None,
    mask_out_path: Path | None = None,
) -> dict[str, Any]:
    """Submit a volumetric segmentation job.

    Returns ``{"status": "queued"|"done"|"error", "job_id": ..., ...}``.
    When ``inline_override`` is True or GPU is available or
    ``SEG_WORKER_BACKEND=inline``, runs synchronously and returns the
    full result under ``"result"``.
    """
    be_worker = SEG_WORKER_BACKEND
    run_inline = (
        inline_override
        if inline_override is not None
        else (be_worker == "inline" or _gpu_available())
    )

    if run_inline:
        # Straight-through, same process. Cheap on GPU, risky on CPU.
        result = _worker_entrypoint(
            str(volume_path), backend, str(output_json),
            str(mask_out_path) if mask_out_path else None,
        )
        return {
            "status": "done",
            "job_id": "inline",
            "output_json": str(output_json),
            "result": result,
            "ran_inline": True,
        }

    if be_worker == "celery":
        # Intentional: don't silently fall through to ProcessPool; the user
        # asked for Celery and we can't pretend otherwise.
        raise NotImplementedError(
            "seg_worker: SEG_WORKER_BACKEND=celery requires a Celery app + Redis "
            "broker; set SEG_REDIS_URL and wire the broker in this module."
        )

    # Default: ProcessPoolExecutor with one slot.
    ex = _get_executor()
    job_id = uuid.uuid4().hex[:12]
    fut = ex.submit(
        _worker_entrypoint,
        str(volume_path), backend, str(output_json),
        str(mask_out_path) if mask_out_path else None,
    )
    _jobs[job_id] = fut
    log.info("seg_worker: queued job %s for %s", job_id, volume_path.name)
    return {
        "status": "queued",
        "job_id": job_id,
        "output_json": str(output_json),
        "ran_inline": False,
    }


def poll(job_id: str) -> dict[str, Any]:
    """Non-blocking check of a queued job."""
    fut = _jobs.get(job_id)
    if fut is None:
        return {"status": "unknown", "job_id": job_id}
    if not fut.done():
        return {"status": "running", "job_id": job_id}
    try:
        result = fut.result(timeout=0)
        return {"status": "done", "job_id": job_id, "result": result}
    except Exception as exc:
        return {"status": "error", "job_id": job_id,
                "error": f"{type(exc).__name__}: {exc}"}


def shutdown(wait: bool = False) -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait, cancel_futures=not wait)
        _executor = None
        _jobs.clear()


__all__ = ["submit", "poll", "shutdown"]
