"""Pipeline endpoints.

POST /api/v1/run                    — upload ZIP → full pipeline → complete result
GET  /api/v1/patients               — list all processed patients
GET  /api/v1/patients/{pid}         — patient detail: outputs + pipeline summary
POST /api/v1/calendar/clear         — delete all agent-created calendar events
POST /api/v1/calendar/clear/{pid}   — delete events for one patient only
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...config import APPROVAL_REQUIRED, DATA_ROOT, OUTPUTS_DIR
from ...orchestrator import JOBS, run_patient
from ...utils.audit import log_access

log = logging.getLogger(__name__)

router = APIRouter()

_MAX_BYTES     = 200 * 1024 * 1024   # 200 MB per file
_MAX_ZIP_BYTES = 2  * 1024 * 1024 * 1024  # 2 GB zip


def _safe_pid(pid: str) -> str:
    if not pid or "/" in pid or "\\" in pid or ".." in pid or not pid.isascii():
        raise HTTPException(400, "invalid patient_id — alphanumeric + hyphens only")
    return pid.strip().upper()


def _extract_zip(raw: bytes, pid: str) -> list[dict]:
    """Extract ZIP into DATA_ROOT/<pid>/, return list of written file records."""
    patient_root = Path(DATA_ROOT) / pid
    patient_root.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        top_subdirs: list[str] = []
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if len(parts) >= 2 and parts[0].upper() == pid:
                parts = parts[1:]
            if len(parts) >= 2:
                sd = parts[0].lower()
                if sd not in top_subdirs:
                    top_subdirs.append(sd)

        def _resolve(parts: tuple) -> tuple[str, str]:
            if len(parts) == 1:
                fname = parts[0]
                m = re.search(r"_v(\d+)$", Path(fname).stem.lower())
                if m and int(m.group(1)) >= 2:
                    return f"v{m.group(1)}", fname
                return "", fname
            raw_dir = parts[0].lower()
            fname   = parts[-1]
            m = re.fullmatch(r"v(?:isit)?(\d+)", raw_dir)
            if m:
                n = m.group(1)
                return ("" if n == "1" else f"v{n}"), fname
            idx = top_subdirs.index(raw_dir) if raw_dir in top_subdirs else 0
            return ("" if idx == 0 else f"v{idx + 1}"), fname

        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) >= 2 and parts[0].upper() == pid:
                parts = parts[1:]
            visit_dir, fname = _resolve(tuple(parts))
            if ".." in fname or "/" in fname or "\\" in fname:
                continue
            dest_dir = patient_root / visit_dir if visit_dir else patient_root
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / fname
            data = zf.read(info.filename)
            if len(data) > _MAX_BYTES:
                raise HTTPException(413, f"{fname} exceeds 200 MB")
            dest.write_bytes(data)
            written.append({"name": fname, "visit": visit_dir or "v1", "size": len(data)})

    if not written:
        raise HTTPException(422, "zip contained no extractable files")
    return written


def _run_pipeline(pid: str, job_id: str) -> None:
    """Run the pipeline synchronously. Called inside a thread pool.

    Task 9 safety — when ``APPROVAL_REQUIRED=True`` the legacy ``/run`` endpoint
    silently downgrades to ``mode=prep`` so Gmail / Drive / Calendar CANNOT fire
    without a clinician having posted ``/approve/{pid}`` first. Set
    ``APPROVAL_REQUIRED=False`` in config.py to restore one-shot behaviour.
    """
    try:
        if APPROVAL_REQUIRED:
            run_patient(pid, job_id=job_id, mode="prep")
            cur = JOBS[job_id].get("status")
            if cur not in ("error", "completed_with_warnings"):
                JOBS[job_id]["status"] = "pending_approval"
        else:
            run_patient(pid, job_id=job_id)
            cur = JOBS[job_id].get("status")
            if cur not in ("error", "completed_with_warnings"):
                JOBS[job_id]["status"] = "completed"
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"[:300]


def _collect_outputs(pid: str) -> list[dict]:
    """List all output files generated for a patient."""
    out_dir = Path(OUTPUTS_DIR) / pid
    if not out_dir.exists():
        return []
    files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and "images" not in p.parts:
            files.append({
                "name": p.name,
                "path": str(p.relative_to(out_dir)),
                "size_bytes": p.stat().st_size,
            })
    return files


@router.post("/run")
async def run(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload patient ZIP and run the complete pipeline in one call.

    Stages: ingest → MRI analysis → RECIST → pharma → synthesis
    After synthesis: Gmail patient letter + GP handover, Drive sync, Calendar events,
    Google Chat welcome DM.

    Returns when everything is done. No job_id, no polling.
    """
    pid = _safe_pid(patient_id)

    log_access(pid=pid, role="doctor", caller_name="swagger_ui",
               action="run_pipeline", endpoint="/api/v1/run")

    log.info(">> /api/v1/run  patient=%s  file=%s  size=%.1f MB",
             pid, file.filename, (file.size or 0) / 1_048_576)

    # ── 1. Read and validate ZIP ──────────────────────────────────────────────
    raw = await file.read()
    if len(raw) > _MAX_ZIP_BYTES:
        raise HTTPException(413, "zip exceeds 2 GB limit")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise HTTPException(400, "uploaded file is not a valid zip archive")

    # ── 2. Extract files ──────────────────────────────────────────────────────
    written = _extract_zip(raw, pid)
    visits  = sorted({f["visit"] for f in written})

    # ── 3. Run full pipeline (blocking — runs to completion) ──────────────────
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "patient_id": pid,
        "phase": None,
        "status": "running",
        "phases": [],
        "qa_ready": False,
    }

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_pipeline, pid, job_id)

    # ── 4. Collect results ────────────────────────────────────────────────────
    job     = JOBS[job_id]
    outputs = _collect_outputs(pid)

    # Summarise per-phase status
    stages: dict[str, Any] = {}
    for p in job.get("phases", []):
        name = p.get("phase", "?")
        stages[name] = {
            "ok":       p.get("ok", False),
            "steps":    p.get("steps", 0),
            "skipped":  p.get("skipped", False),
            "reason":   p.get("reason", ""),
        }

    return {
        "ok":            True,
        "patient_id":    pid,
        "status":        job.get("status", "unknown"),
        "files_uploaded": len(written),
        "visits":        visits,
        "stages":        stages,
        "warnings":      job.get("warnings", []),
        "notifications": job.get("notifications", {}),
        "outputs":       outputs,
        "n_outputs":     len(outputs),
        "qa_ready":      job.get("qa_ready", False),
        "error":         job.get("error"),
    }


# ── Patient listing ───────────────────────────────────────────────────────────

def _patient_summary(pid: str) -> dict[str, Any]:
    """Read S3_record.json + S12_summary.json to build a compact patient card."""
    out_dir = Path(OUTPUTS_DIR) / pid
    summary: dict[str, Any] = {"patient_id": pid, "qa_ready": out_dir.exists()}

    # Name / diagnosis — try full pipeline doc first, fall back to S3_record
    pipeline_file = out_dir / f"P{pid}_full_pipeline.json"
    record_file   = out_dir / "S3_record.json"
    for src in (pipeline_file, record_file):
        if src.exists():
            try:
                doc = json.loads(src.read_text(encoding="utf-8"))
                summary["name"]      = doc.get("patient_name") or doc.get("name", "")
                summary["diagnosis"] = (
                    doc.get("tumor_type")
                    or doc.get("diagnosis")
                    or doc.get("primary_diagnosis", "")
                )
                summary["dob"] = doc.get("date_of_birth") or doc.get("dob", "")
                if summary.get("name"):
                    break
            except Exception:
                pass

    # Last processed timestamp — use file mtime of working_memory.json
    wm_file = out_dir / "working_memory.json"
    if wm_file.exists():
        from datetime import datetime, timezone
        mtime = wm_file.stat().st_mtime
        summary["last_processed"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    # Count output files
    summary["n_outputs"] = len([
        p for p in out_dir.rglob("*")
        if p.is_file() and "images" not in p.parts
    ]) if out_dir.exists() else 0

    return summary


@router.get("/patients")
def list_patients() -> dict[str, Any]:
    """List all patients that have been processed (have an outputs folder)."""
    out_root = Path(OUTPUTS_DIR)
    if not out_root.exists():
        return {"ok": True, "patients": [], "n": 0}

    patients = []
    for d in sorted(out_root.iterdir()):
        if d.is_dir():
            patients.append(_patient_summary(d.name))

    return {"ok": True, "patients": patients, "n": len(patients)}


@router.post("/calendar/clear")
def calendar_clear_all() -> dict[str, Any]:
    """Delete ALL calendar events created by this agent (any patient).

    Use this to clean up test/development events from your Google Calendar.
    Searches ±2 years from today for events whose summary starts with '['.
    """
    from ...integrations.calendar_client import CalendarClient
    client = CalendarClient()
    if not client.ready:
        raise HTTPException(
            503,
            "Google Calendar not available — check credentials/token.json "
            "and run setup_oauth.py if needed.",
        )
    result = client.clear_agent_events()
    return {
        "ok":      True,
        "deleted": result["deleted"],
        "failed":  result["failed"],
        "message": (
            f"Deleted {result['deleted']} agent-created calendar event(s). "
            + (f"{result['failed']} deletion(s) failed — check server logs." if result["failed"] else "")
        ).strip(),
    }


@router.post("/calendar/clear/{patient_id}")
def calendar_clear_patient(patient_id: str) -> dict[str, Any]:
    """Delete calendar events for a specific patient only.

    Matches events with summary prefix '[PATIENT_ID]'.
    """
    pid = _safe_pid(patient_id)
    from ...integrations.calendar_client import CalendarClient
    client = CalendarClient()
    if not client.ready:
        raise HTTPException(
            503,
            "Google Calendar not available — check credentials/token.json "
            "and run setup_oauth.py if needed.",
        )
    result = client.clear_agent_events(patient_id=pid)
    return {
        "ok":         True,
        "patient_id": pid,
        "deleted":    result["deleted"],
        "failed":     result["failed"],
        "message": (
            f"Deleted {result['deleted']} calendar event(s) for patient {pid}. "
            + (f"{result['failed']} deletion(s) failed — check server logs." if result["failed"] else "")
        ).strip(),
    }


@router.get("/patients/{patient_id}/history")
def get_patient_history(patient_id: str) -> dict[str, Any]:
    """Phase 5.3 / Module 2 — Longitudinal visit history for a patient.

    Reads ``outputs/<pid>/history/longitudinal.jsonl`` (append-only) and
    returns the ordered list of prior visits plus the 4 derived trajectory
    features as they would be computed against the current SoD = last visit's.
    """
    pid = _safe_pid(patient_id)
    out_dir = Path(OUTPUTS_DIR) / pid
    if not out_dir.exists():
        raise HTTPException(404, f"No outputs found for patient {pid}")

    from ...utils.longitudinal_history import load_history, compute_trajectory_features

    history = load_history(out_dir, pid)
    last_sod = history.visits[-1].sum_of_diameters_mm if history.visits else 0.0
    traj = compute_trajectory_features(history, last_sod)
    return {
        "patient_id": pid,
        "visit_count": history.visit_count,
        "visits": [v.model_dump() for v in history.visits],
        "trajectory_features": traj,
    }


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str) -> dict[str, Any]:
    """Return full detail for one patient: outputs list + pipeline stage summary."""
    pid = _safe_pid(patient_id)
    out_dir = Path(OUTPUTS_DIR) / pid
    if not out_dir.exists():
        raise HTTPException(404, f"No outputs found for patient {pid}")

    summary = _patient_summary(pid)
    outputs = _collect_outputs(pid)

    # Read pipeline stage results from working_memory.json if available
    stages: dict[str, Any] = {}
    wm_file = out_dir / "working_memory.json"
    if wm_file.exists():
        try:
            wm = json.loads(wm_file.read_text(encoding="utf-8"))
            for p in wm.get("phases", []):
                name   = p.get("phase", "?")
                status = p.get("status", "")
                stages[name] = {
                    "ok":      status in ("ok", "completed"),
                    "skipped": status == "skipped",
                    "steps":   p.get("steps", 0),
                }
        except Exception:
            pass

    return {
        "ok":       True,
        **summary,
        "stages":   stages,
        "outputs":  outputs,
        "n_outputs": len(outputs),
    }
