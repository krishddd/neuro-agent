"""Job control: kick off the guarded pipeline asynchronously and poll status.

For v1 we use FastAPI's BackgroundTasks + the in-memory JOBS dict from
orchestrator.py. The pipeline is GPU-blocking, so concurrent jobs share
the GPU sequentially through Ollama's queue.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ...config import OUTPUTS_DIR
from ...orchestrator import JOBS, run_patient

router = APIRouter()


class RunRequest(BaseModel):
    patient_id: str
    stop_after: Optional[str] = None  # ingest|mri|recist|pharma|synthesis


def _run(patient_id: str, stop_after: Optional[str], job_id: str) -> None:
    try:
        run_patient(patient_id, job_id=job_id, stop_after=stop_after)
        # Preserve "completed_with_warnings" set by the orchestrator;
        # only promote to "completed" if nothing bad was recorded.
        cur = JOBS[job_id].get("status")
        if cur not in ("error", "completed_with_warnings"):
            JOBS[job_id]["status"] = "completed"
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"{type(e).__name__}: {e}"[:300]


@router.post("/jobs")
def create_job(req: RunRequest, background: BackgroundTasks) -> dict:
    import uuid
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "patient_id": req.patient_id,
        "phase": None,
        "status": "queued",
        "phases": [],
        "qa_ready": False,
    }
    background.add_task(_run, req.patient_id, req.stop_after, job_id)
    return {"ok": True, "job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {"ok": True, "job_id": job_id, **job}


@router.get("/outputs/{patient_id}")
def list_outputs(patient_id: str) -> dict:
    """List all output files generated for a patient.

    Returns JSON/MD paths grouped by file type so the caller can retrieve
    the report, FHIR bundle, patient letter, etc. without knowing the timestamp.
    """
    out_dir = OUTPUTS_DIR / patient_id
    if not out_dir.exists():
        raise HTTPException(404, f"no outputs found for {patient_id}")

    files: list[dict] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out_dir))
        ext = p.suffix.lower()
        kind = (
            "report" if p.name == "report.md"
            else "patient_letter" if "patient_letter" in p.name
            else "gp_handover" if "gp_handover" in p.name
            else "fhir_bundle" if "fhir_bundle" in p.name
            else "mdt_package" if "mdt_package" in p.name
            else "qa_examples" if p.name == "S11_qa_examples.json"
            else "laboratory_results" if p.name == "laboratory_results.json"
            else "working_memory" if p.name == "working_memory.json"
            else "extended" if "extended" in p.parts
            else "stage_output" if p.name.startswith("S") and ext == ".json"
            else "full_pipeline" if "_full_pipeline.json" in p.name
            else "image" if ext in {".png", ".jpg"}
            else "other"
        )
        files.append({
            "name": p.name,
            "path": str(p),
            "relative": rel,
            "size_bytes": p.stat().st_size,
            "kind": kind,
        })

    return {
        "ok": True,
        "patient_id": patient_id,
        "n_files": len(files),
        "files": files,
    }


@router.get("/jobs")
def list_jobs() -> dict:
    return {
        "ok": True,
        "n": len(JOBS),
        "jobs": [
            {
                "job_id": jid,
                "patient_id": j.get("patient_id"),
                "phase": j.get("phase"),
                "status": j.get("status"),
                "qa_ready": j.get("qa_ready", False),
            }
            for jid, j in JOBS.items()
        ],
    }
