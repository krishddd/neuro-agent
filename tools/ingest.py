"""Stage 1 — deterministic multi-source ingestion.

Walks the patient folder, classifies each file by extension + filename
keywords, extracts text from PDFs, and normalizes DICOM/PNG/JPG inputs
into PNGs the vision model can consume. All outputs go into
WorkingMemory under `ingestion`.

No LLM calls are made here — ingestion must be fast, deterministic and
PHI-safe so that every downstream phase gets a consistent view.
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from ..config import DATA_ROOT, FILE_KEYWORDS, OUTPUTS_DIR, PHASE4_DATA_ROOT
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.schemas import FileKind, IngestedFile, IngestionResult
from . import register

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_PDF_EXT = {".pdf"}
_DICOM_EXT = {".dcm", ".dicom"}
# NIfTI volumetric brain MRI (Task 5) — handled by volumetric_seg pipeline,
# not the 2D image normaliser.
_NIFTI_EXT = {".nii"}  # .nii.gz detected by name check below


def _is_nifti(name: str) -> bool:
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _classify(path: Path) -> FileKind:
    name = path.name.lower()
    ext = path.suffix.lower()

    if ext in _DICOM_EXT or ext in _IMAGE_EXT or _is_nifti(name):
        return "mri_image"

    if ext in _PDF_EXT:
        for kind, kws in FILE_KEYWORDS.items():
            for kw in kws:
                if kw in name:
                    if kind == "mri_image":
                        continue
                    return kind  # type: ignore[return-value]
        return "other"
    return "other"


def _extract_pdf_text(path: Path) -> str:
    try:
        import pdfplumber
    except Exception as exc:
        log.warning("ingest: pdfplumber unavailable — PDF text extraction disabled (%s)",
                    type(exc).__name__)
        return ""
    try:
        with pdfplumber.open(str(path)) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            return "\n\n".join(parts)
    except Exception as exc:
        log.warning("ingest: PDF text extraction failed for %s: %s (%s)",
                    path.name, type(exc).__name__, str(exc)[:120])
        return ""


def _normalize_image(path: Path, out_dir: Path) -> Path | None:
    """Convert DICOM/other images into a standard PNG under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in _DICOM_EXT:
        from ..utils.dicom_anon import dicom_to_png

        out = out_dir / (path.stem + ".png")
        try:
            return dicom_to_png(path, out)
        except Exception as exc:
            log.warning("ingest: DICOM→PNG failed for %s: %s (%s)",
                        path.name, type(exc).__name__, str(exc)[:120])
            return None
    if ext in _IMAGE_EXT:
        out = out_dir / (path.stem + "_norm.png")
        try:
            from PIL import Image
            import numpy as np

            with Image.open(path) as im:
                mode = im.mode
                # I;16 and I;16B are 16-bit grayscale — PIL cannot directly
                # convert or save them via the standard .convert() path.
                if mode in ("I;16", "I;16B", "I;16S"):
                    arr = np.frombuffer(im.tobytes(), dtype=np.uint16).reshape(
                        im.size[1], im.size[0]
                    )
                    # Min-max scale to 8-bit.
                    lo, hi = float(arr.min()), float(arr.max())
                    if hi <= lo:
                        hi = lo + 1.0
                    arr8 = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
                    Image.fromarray(arr8, mode="L").save(out, format="PNG")
                elif mode == "I":
                    # 32-bit signed integer mode — same treatment.
                    arr = np.array(im, dtype=np.int32)
                    lo, hi = float(arr.min()), float(arr.max())
                    if hi <= lo:
                        hi = lo + 1.0
                    arr8 = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
                    Image.fromarray(arr8, mode="L").save(out, format="PNG")
                else:
                    # Standard modes (RGB, L, RGBA, etc.) — convert to L or RGB.
                    target = "L" if mode not in ("L", "RGB", "RGBA") else mode
                    im.convert(target).save(out, format="PNG")
            return out
        except Exception as exc:
            log.warning("ingest: image normalization failed for %s: %s (%s)",
                        path.name, type(exc).__name__, str(exc)[:120])
            return path  # fall back to original path
    return None


def _safe_rel(path: Path, base: Path) -> str:
    """Return path relative to base; falls back to patient-dir-relative on ValueError."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        # File is outside base (e.g., from phase4_patient_data) — use parent dir as anchor
        try:
            return str(path.relative_to(base.parent.parent))
        except ValueError:
            return path.name


def _walk_patient(pid: str) -> tuple[Path, list[tuple[Path, str]]]:
    """Return (patient_root, [(file_path, visit_id), ...]).

    Handles both layouts:
      v2 (unified)  : Datasets/patients/{pid}/{clinical,phase4}/*
      v1 (legacy)   : Datasets/raw_docs/{pid}/* + Datasets/phase4_patient_data/{pid}/*
    """
    # v2: DATA_ROOT == Datasets/patients. Per-patient files split under
    # {pid}/clinical and {pid}/phase4.
    v2_patient = DATA_ROOT / pid
    v2_clinical = v2_patient / "clinical"
    v2_phase4   = v2_patient / "phase4"

    results: list[tuple[Path, str]] = []

    def _collect(search_root: Path, default_visit: str = "v1") -> None:
        for p in sorted(search_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(search_root)
            visit = rel.parts[0] if len(rel.parts) > 1 else default_visit
            v = visit.lower()
            if v.startswith("visit"):
                tail = v.replace("visit", "").strip() or "1"
                visit = f"v{tail}"
            elif v in {"v1", "v2", "v3", "v4"}:
                visit = v
            else:
                visit = default_visit
            results.append((p, visit))

    # v2 unified layout — preferred path
    if v2_clinical.exists():
        _collect(v2_clinical)
        if v2_phase4.exists():
            _collect(v2_phase4, default_visit="v1")
        return v2_patient, results

    # Fall back to v1 layout: DATA_ROOT may be raw_docs/, or v2 root with missing clinical/.
    legacy_root = v2_patient  # same path in v1 if DATA_ROOT==raw_docs
    if not legacy_root.exists():
        raise FileNotFoundError(f"patient folder not found: {legacy_root}")
    _collect(legacy_root)

    # Separately scan the old phase4_patient_data/{pid}/ if present
    phase4_root = PHASE4_DATA_ROOT / pid
    if phase4_root.exists() and phase4_root != legacy_root:
        _collect(phase4_root, default_visit="v1")

    return legacy_root, results


@register("ingest_patient_files")
def ingest_patient_files(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    """Scan the patient folder, extract text / normalize images, store in memory."""
    pid = memory.patient_id
    with stage_timer("ingest", pid=pid) as _t:
        root, files = _walk_patient(pid)
        out_imgs = OUTPUTS_DIR / pid / "images"

        ingested: list[IngestedFile] = []
        visits_seen: set[str] = set()
        for path, visit in files:
            visits_seen.add(visit)
            kind = _classify(path)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            text = None
            img = None

            if kind == "mri_image":
                if _is_nifti(path.name.lower()):
                    # NIfTI: don't try to render a 2D PNG here; the
                    # volumetric_seg tool will consume the raw .nii(.gz).
                    img = None
                else:
                    img_path = _normalize_image(path, out_imgs / visit)
                    img = str(img_path) if img_path else None
            elif path.suffix.lower() in _PDF_EXT:
                text = _extract_pdf_text(path)
            elif path.suffix.lower() in _IMAGE_EXT:
                img_path = _normalize_image(path, out_imgs / visit)
                img = str(img_path) if img_path else None
            elif path.suffix.lower() == ".json" and kind in ("wearable", "intake_form"):
                # Phase 4 structured JSON — store as text for downstream tools.
                # Force utf-8 so Windows cp1252 default doesn't trip on UTF-8 payloads.
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    # Fallback for files written with BOM or other encodings
                    text = path.read_text(encoding="utf-8-sig", errors="replace")
                except Exception as exc:
                    log.warning("ingest: JSON read failed for %s: %s (%s)",
                                path.name, type(exc).__name__, str(exc)[:120])
                    text = None

            ingested.append(
                IngestedFile(
                    path=str(path),
                    kind=kind,
                    visit=visit,
                    mime=mime,
                    size_bytes=path.stat().st_size,
                    text=text,
                    image_path=img,
                    meta={
                        "rel": _safe_rel(path, root),
                        "is_nifti": _is_nifti(path.name.lower()),
                    },
                )
            )

        result = IngestionResult(
            patient_id=pid,
            files=ingested,
            visits=sorted(visits_seen),
            has_prior_scans=len(visits_seen) > 1,
        )
        memory.set(WorkingMemory.INGESTION, result)
        _t.meta["n_files"] = len(ingested)

        # Compact return for the LLM — full data stays in memory.
        return {
            "ok": True,
            "n_files": len(ingested),
            "visits": result.visits,
            "has_prior_scans": result.has_prior_scans,
            "kinds": {
                k: sum(1 for f in ingested if f.kind == k)
                for k in set(f.kind for f in ingested)
            },
        }
