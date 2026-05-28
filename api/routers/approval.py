"""Task 9 — Human-in-the-Loop approval endpoints.

Flow:

    POST /api/v1/run/prep         upload ZIP → phases 1-4 → PENDING_APPROVAL.json
    GET  /api/v1/pending/{pid}    read PENDING_APPROVAL.json
    POST /api/v1/approve/{pid}    clinician signs off → APPROVED.json / REJECTED.json
    POST /api/v1/run/execute/{pid}  phases 5-6 fire Drive / Gmail / Calendar

``POST /api/v1/run`` (existing full-pipeline endpoint) remains for legacy
one-shot runs; production deployments should set ``APPROVAL_REQUIRED=True``
and use the three-step flow above.
"""
from __future__ import annotations

import asyncio
import io
import logging
import uuid
import zipfile
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ...orchestrator import JOBS, run_patient
from ...utils.approval import (
    ApprovalRejectedError,
    ApprovalRequiredError,
    read_approval,
    read_pending,
    record_approval,
)
from ...utils.audit import log_access
from .process import _extract_zip, _safe_pid

log = logging.getLogger(__name__)

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────────

class ApproveBody(BaseModel):
    approver_email:   str
    decision:         Literal["APPROVE", "REJECT", "MODIFY"]
    clinician_notes:  str = ""
    override_regimen: Optional[str] = None


# ── Prep endpoint ─────────────────────────────────────────────────────────────

def _run_prep(pid: str, job_id: str) -> None:
    try:
        run_patient(pid, job_id=job_id, mode="prep")
        cur = JOBS[job_id].get("status")
        if cur not in ("error", "completed_with_warnings"):
            JOBS[job_id]["status"] = "pending_approval"
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = f"{type(exc).__name__}: {exc}"[:300]


@router.post("/run/prep")
async def run_prep(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload patient ZIP and run PREP phases (1-4) only.

    Stops at S18 TreatmentProposal and writes ``PENDING_APPROVAL.json``.
    No Gmail / Drive / Calendar side-effects fire from this endpoint.
    """
    pid = _safe_pid(patient_id)

    log_access(pid=pid, role="doctor", caller_name="swagger_ui",
               action="run_prep", endpoint="/api/v1/run/prep")

    raw = await file.read()
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise HTTPException(400, "uploaded file is not a valid zip archive")

    written = _extract_zip(raw, pid)
    visits  = sorted({f["visit"] for f in written})

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "patient_id": pid,
        "phase":      None,
        "status":     "running",
        "mode":       "prep",
        "phases":     [],
        "qa_ready":   False,
    }

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_prep, pid, job_id)

    job = JOBS[job_id]
    return {
        "ok":               True,
        "patient_id":       pid,
        "mode":             "prep",
        "status":           job.get("status", "unknown"),
        "files_uploaded":   len(written),
        "visits":           visits,
        "pending_approval": job.get("pending_approval"),
        "warnings":         job.get("warnings", []),
        "error":            job.get("error"),
        "next":             f"GET /api/v1/pending/{pid}  →  POST /api/v1/approve/{pid}",
    }


# ── Pending read ──────────────────────────────────────────────────────────────

@router.get("/pending/{patient_id}")
def get_pending(patient_id: str) -> dict[str, Any]:
    pid = _safe_pid(patient_id)
    pending = read_pending(pid)
    if pending is None:
        raise HTTPException(404, f"no PENDING_APPROVAL.json for patient {pid}")
    return {"ok": True, "pending": pending.model_dump()}


# ── Approve / reject / modify ─────────────────────────────────────────────────

@router.post("/approve/{patient_id}")
def approve_patient(patient_id: str, body: ApproveBody) -> dict[str, Any]:
    """Clinician sign-off. Writes APPROVED.json or REJECTED.json."""
    pid = _safe_pid(patient_id)
    if body.decision == "MODIFY" and not (body.override_regimen or "").strip():
        raise HTTPException(422,
            "decision=MODIFY requires a non-empty override_regimen")

    log_access(
        pid=pid, role="doctor",
        caller_name=body.approver_email or "unknown",
        action=f"approve.{body.decision.lower()}",
        endpoint=f"/api/v1/approve/{pid}",
    )

    rec = record_approval(
        pid,
        approver_email=body.approver_email.strip(),
        decision=body.decision,
        clinician_notes=body.clinician_notes,
        override_regimen=body.override_regimen,
    )
    return {"ok": True, "approval": rec.model_dump()}


# ── Execute endpoint ──────────────────────────────────────────────────────────

def _run_execute(pid: str, job_id: str) -> None:
    try:
        run_patient(pid, job_id=job_id, mode="execute")
        cur = JOBS[job_id].get("status")
        if cur not in ("error", "completed_with_warnings"):
            JOBS[job_id]["status"] = "completed"
    except ApprovalRequiredError as exc:
        JOBS[job_id]["status"] = "approval_required"
        JOBS[job_id]["error"]  = str(exc)[:300]
    except ApprovalRejectedError as exc:
        JOBS[job_id]["status"] = "rejected"
        JOBS[job_id]["error"]  = str(exc)[:300]
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = f"{type(exc).__name__}: {exc}"[:300]


@router.post("/run/execute/{patient_id}")
async def run_execute(patient_id: str) -> dict[str, Any]:
    """Run EXECUTE phases (5-6). Requires APPROVED.json on disk."""
    pid = _safe_pid(patient_id)

    # Fail fast with a 409 if no approval marker exists — avoids spinning up
    # a thread pool for a run we already know will be refused.
    approval = read_approval(pid)
    if approval is None:
        raise HTTPException(
            409,
            f"patient {pid} has no APPROVED.json — call "
            f"POST /api/v1/approve/{pid} first",
        )
    if approval.decision == "REJECT":
        raise HTTPException(409, f"patient {pid} was REJECTED by clinician")

    log_access(pid=pid, role="doctor", caller_name=approval.approver_email,
               action="run_execute", endpoint=f"/api/v1/run/execute/{pid}")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "patient_id": pid,
        "phase":      None,
        "status":     "running",
        "mode":       "execute",
        "phases":     [],
        "qa_ready":   False,
    }

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_execute, pid, job_id)

    job = JOBS[job_id]
    return {
        "ok":                True,
        "patient_id":        pid,
        "mode":              "execute",
        "status":            job.get("status", "unknown"),
        "phases":            job.get("phases", []),
        "notifications":     job.get("notifications", {}),
        "approval_archived": job.get("approval_archived", []),
        "approver":          approval.approver_email,
        "warnings":          job.get("warnings", []),
        "error":             job.get("error"),
    }
