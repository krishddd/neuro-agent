"""Upload patient files into the dataset directory.

Two upload modes
----------------
1. **Multi-file upload** (existing):
   POST /api/v1/upload
   - `patient_id` (form field)
   - `visit`      (form field, default "v1")
   - `files[]`    (one or more UploadFile)

   Writes each file to DATA_ROOT/<patient_id>/ (or /<visit>/ for v2+).

2. **ZIP folder upload** (new):
   POST /api/v1/upload/zip
   - `patient_id` (form field)
   - `file`       (single zip UploadFile containing the patient folder)

   Extracts the zip preserving relative paths so that files inside a
   visit2/ subfolder land in DATA_ROOT/<patient_id>/visit2/.
   Top-level files land in DATA_ROOT/<patient_id>/.

Both endpoints accept an optional `auto_run=true` form flag that kicks off
the orchestrator pipeline immediately after the files are written.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ...config import DATA_ROOT
from ...orchestrator import JOBS, run_patient

router = APIRouter()

_MAX_BYTES = 200 * 1024 * 1024  # 200 MB per file
_MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB total zip


def _safe_patient_id(pid: str) -> str:
    if not pid or "/" in pid or "\\" in pid or ".." in pid or not pid.isascii():
        raise HTTPException(400, "invalid patient_id — use alphanumeric + hyphens only")
    return pid.strip()


def _safe_visit(visit: str) -> str:
    v = (visit or "v1").strip() or "v1"
    if "/" in v or "\\" in v or ".." in v:
        raise HTTPException(400, "invalid visit value")
    return v


# ---------- multi-file upload ----------
@router.post("/upload")
async def upload_files(
    background: BackgroundTasks,
    patient_id: str = Form(...),
    visit: Optional[str] = Form("v1"),
    auto_run: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> dict:
    """Upload individual files for one patient visit.

    Set `auto_run=true` to immediately start the orchestrator pipeline
    as a background task after the files are written.
    """
    pid = _safe_patient_id(patient_id)
    visit = _safe_visit(visit or "v1")

    target_dir = Path(DATA_ROOT) / pid
    if visit != "v1":
        target_dir = target_dir / visit
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    for f in files:
        if not f.filename:
            continue
        dest = target_dir / Path(f.filename).name
        size = 0
        with dest.open("wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"{f.filename} exceeds 200 MB")
                out.write(chunk)
        written.append({"name": dest.name, "visit": visit, "size": size, "path": str(dest)})

    job_id: Optional[str] = None
    if auto_run:
        import uuid
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"patient_id": pid, "phase": None, "status": "queued",
                        "phases": [], "qa_ready": False}
        background.add_task(run_patient, pid, job_id=job_id)

    return {
        "ok": True,
        "patient_id": pid,
        "visit": visit,
        "n_files": len(written),
        "files": written,
        **({"job_id": job_id, "pipeline_status": "queued"} if job_id else {}),
    }


# ---------- ZIP folder upload ----------
@router.post("/upload/zip")
async def upload_zip(
    background: BackgroundTasks,
    patient_id: str = Form(...),
    auto_run: bool = Form(False),
    file: UploadFile = File(...),
) -> dict:
    """Upload the entire patient folder as a single ZIP archive.

    The ZIP must contain files at the root level for visit 1, and inside a
    'visit2/' (or 'v2/') subdirectory for visit 2. Nested directories deeper
    than one level are flattened into their parent visit directory.

    Example ZIP layout:
        brain_axial.dcm
        mri_brain_report.pdf
        prescription_oncology_clinic.pdf
        visit2/brain_axial_v2.dcm
        visit2/mri_brain_report_v2.pdf

    Set `auto_run=true` to start the pipeline immediately after extraction.
    """
    pid = _safe_patient_id(patient_id)

    # Read the entire zip into memory (cap at 2 GB).
    raw = await file.read()
    if len(raw) > _MAX_ZIP_BYTES:
        raise HTTPException(413, "zip file exceeds 2 GB limit")

    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise HTTPException(400, "uploaded file is not a valid zip archive")

    patient_root = Path(DATA_ROOT) / pid
    patient_root.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Pre-scan: collect all subdirectory names present at the top level
        # (after stripping a matching patient_id wrapper folder).
        # This lets us map unknown subdirs → visit numbers by their order.
        top_subdirs: list[str] = []
        for info in zf.infolist():
            zpath = Path(info.filename)
            parts = zpath.parts
            if len(parts) >= 2 and parts[0].upper() == pid.upper():
                parts = parts[1:]
            if len(parts) >= 2:
                sd = parts[0].lower()
                if sd not in top_subdirs:
                    top_subdirs.append(sd)

        def _resolve_visit(parts: tuple) -> tuple[str, str]:
            """Return (visit_dir, filename) for a zip entry's path parts."""
            if len(parts) == 1:
                fname = parts[0]
                # Flat file: detect _v2, _v3 … suffix in stem
                import re as _re
                m = _re.search(r"_v(\d+)$", Path(fname).stem.lower())
                if m and int(m.group(1)) >= 2:
                    return f"v{m.group(1)}", fname
                return "", fname

            raw_dir = parts[0].lower()
            fname   = parts[-1]

            # Explicit visit directory names: v1, v2, visit1, visit2, …
            import re as _re
            m = _re.fullmatch(r"v(?:isit)?(\d+)", raw_dir)
            if m:
                n = m.group(1)
                return ("" if n == "1" else f"v{n}"), fname

            # Unknown subdir — map by order of first appearance (first=v1, second=v2, …)
            idx = top_subdirs.index(raw_dir) if raw_dir in top_subdirs else 0
            return ("" if idx == 0 else f"v{idx + 1}"), fname

        for info in zf.infolist():
            if info.is_dir():
                continue

            zpath = Path(info.filename)
            parts = zpath.parts

            # Strip matching top-level wrapper folder.
            if len(parts) >= 2 and parts[0].upper() == pid.upper():
                parts = parts[1:]

            visit_dir, fname = _resolve_visit(tuple(parts))

            # Guard against path traversal.
            if ".." in fname or "/" in fname or "\\" in fname:
                continue

            dest_dir = patient_root / visit_dir if visit_dir else patient_root
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / fname

            data = zf.read(info.filename)
            if len(data) > _MAX_BYTES:
                raise HTTPException(413, f"{fname} inside zip exceeds 200 MB")
            dest.write_bytes(data)

            written.append({
                "name": fname,
                "visit": visit_dir or "v1",
                "size": len(data),
                "path": str(dest),
            })

    if not written:
        raise HTTPException(422, "zip contained no extractable files")

    # NOTE: auto_run background task is queued AFTER extraction is fully
    # complete and all bytes are flushed to disk.  Do NOT move this block
    # above the zip-extraction loop — doing so creates a race where the
    # pipeline's Phase 1 file-walk runs before files are written.
    job_id: Optional[str] = None
    if auto_run:
        import uuid
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"patient_id": pid, "phase": None, "status": "queued",
                        "phases": [], "qa_ready": False}
        background.add_task(run_patient, pid, job_id=job_id)

    return {
        "ok": True,
        "patient_id": pid,
        "n_files": len(written),
        "visits_found": sorted({f["visit"] for f in written}),
        "files": written,
        **({"job_id": job_id, "pipeline_status": "queued"} if job_id else {}),
    }
