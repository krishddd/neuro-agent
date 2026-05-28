"""Task 9 — Human-in-the-Loop approval markers.

The pipeline runs in two phases:

    prep    (ingest → mri → recist → treatment_opt)  →  PENDING_APPROVAL.json
    <clinician review>                                 →  APPROVED.json / REJECTED.json
    execute (pharma → synthesis)                       →  fires Drive / Gmail / Calendar

This module is the on-disk protocol. The orchestrator refuses to enter
"execute" until ``APPROVED.json`` exists for the patient; the execute path
archives the marker into ``outputs/<pid>/audit/`` so we retain a tamper-
evident record of who signed off, when, and on what MDT decision.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..config import OUTPUTS_DIR


# ── Errors ────────────────────────────────────────────────────────────────────

class ApprovalRequiredError(RuntimeError):
    """Raised when execute-mode is invoked without an APPROVED.json marker."""


class ApprovalRejectedError(RuntimeError):
    """Raised when execute-mode is invoked but the patient was rejected."""


# ── Schemas ───────────────────────────────────────────────────────────────────

class PendingApproval(BaseModel):
    patient_id:        str
    generated_at:      str
    mdt_decision:      str                     # APPROVE / MODIFY / REJECT / SKIP
    proposed_regimen:  Optional[str] = None
    consensus_score:   Optional[float] = None
    mdt_discussion_required: bool = False
    summary:           str = ""


class ApprovalRecord(BaseModel):
    patient_id:            str
    approver_email:        str
    decision:              Literal["APPROVE", "REJECT", "MODIFY"]
    clinician_notes:       str = ""
    override_regimen:      Optional[str] = None
    approved_at:           str
    mdt_decision_at_prep:  str = ""


# ── Paths ─────────────────────────────────────────────────────────────────────

def _patient_dir(pid: str) -> Path:
    d = OUTPUTS_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def pending_marker_path(pid: str) -> Path:
    return _patient_dir(pid) / "PENDING_APPROVAL.json"


def approved_marker_path(pid: str) -> Path:
    return _patient_dir(pid) / "APPROVED.json"


def rejected_marker_path(pid: str) -> Path:
    return _patient_dir(pid) / "REJECTED.json"


def audit_dir(pid: str) -> Path:
    d = _patient_dir(pid) / "audit"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Write / read helpers ──────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_pending(pid: str, proposal: dict[str, Any] | None) -> PendingApproval:
    """Persist PENDING_APPROVAL.json summarising the MDT decision.

    Safety: a fresh prep run INVALIDATES any prior approval — stale
    ``APPROVED.json`` / ``REJECTED.json`` markers from an earlier cycle are
    archived into ``audit/`` so the next execute-mode run cannot reuse them.
    """
    # Invalidate any stale approval from a prior cycle before writing the
    # new pending marker — prevents reuse of an old sign-off.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for stale in (approved_marker_path(pid), rejected_marker_path(pid)):
        if stale.exists():
            try:
                dst = audit_dir(pid) / f"{stale.stem}_stale_{stamp}.json"
                shutil.move(str(stale), str(dst))
            except Exception:
                # Last-resort: delete so we fail-closed rather than trust it.
                try:
                    stale.unlink()
                except Exception:
                    pass

    proposal = proposal or {}
    record = PendingApproval(
        patient_id=pid,
        generated_at=_now_iso(),
        mdt_decision=str(proposal.get("decision", "UNKNOWN")),
        proposed_regimen=proposal.get("proposed_regimen"),
        consensus_score=proposal.get("consensus_score"),
        mdt_discussion_required=bool(proposal.get("mdt_discussion_required", False)),
        summary=str(proposal.get("clinical_narrative", ""))[:1000],
    )
    pending_marker_path(pid).write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )
    return record


def read_pending(pid: str) -> PendingApproval | None:
    p = pending_marker_path(pid)
    if not p.exists():
        return None
    try:
        return PendingApproval.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_approval(
    pid: str,
    *,
    approver_email: str,
    decision: Literal["APPROVE", "REJECT", "MODIFY"],
    clinician_notes: str = "",
    override_regimen: Optional[str] = None,
) -> ApprovalRecord:
    """Write APPROVED.json or REJECTED.json based on the decision."""
    pending = read_pending(pid)
    rec = ApprovalRecord(
        patient_id=pid,
        approver_email=approver_email,
        decision=decision,
        clinician_notes=clinician_notes[:2000],
        override_regimen=override_regimen,
        approved_at=_now_iso(),
        mdt_decision_at_prep=(pending.mdt_decision if pending else ""),
    )
    target = rejected_marker_path(pid) if decision == "REJECT" else approved_marker_path(pid)
    target.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    return rec


def read_approval(pid: str) -> ApprovalRecord | None:
    p = approved_marker_path(pid)
    if not p.exists():
        return None
    try:
        return ApprovalRecord.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_rejected(pid: str) -> bool:
    return rejected_marker_path(pid).exists()


# ── Guards used by the orchestrator execute-mode entry ────────────────────────

def require_approval(pid: str) -> ApprovalRecord:
    """Return the active APPROVED.json record or raise.

    Raises
    ------
    ApprovalRejectedError  — a REJECTED.json exists for this patient
    ApprovalRequiredError  — no APPROVED.json yet
    """
    if is_rejected(pid):
        raise ApprovalRejectedError(
            f"patient {pid} was REJECTED by a clinician; execute-mode refused"
        )
    rec = read_approval(pid)
    if rec is None:
        raise ApprovalRequiredError(
            f"patient {pid} has no APPROVED.json — run prep mode and have a "
            f"clinician call POST /api/v1/approve/{pid} first"
        )
    return rec


# ── Archival into audit/ (called after execute completes) ─────────────────────

def archive_markers(pid: str) -> list[str]:
    """Move PENDING_APPROVAL.json + APPROVED.json into outputs/<pid>/audit/.

    Run at the tail end of execute-mode — leaves the audit trail intact
    while clearing the patient root for the next cycle.
    """
    audit = audit_dir(pid)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved: list[str] = []
    for src in (pending_marker_path(pid), approved_marker_path(pid)):
        if src.exists():
            dst = audit / f"{src.stem}_{stamp}.json"
            try:
                shutil.move(str(src), str(dst))
                moved.append(dst.name)
            except Exception:
                # Don't let archival fail the pipeline — copy & leave original.
                try:
                    shutil.copy2(str(src), str(dst))
                    moved.append(dst.name + " (copy)")
                except Exception:
                    pass
    return moved
