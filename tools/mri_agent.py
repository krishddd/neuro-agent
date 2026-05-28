"""Phase 2 — MRI sub-agent.

Four registered tools:

    analyze_scan(visit)            -> VisionObservation written to memory
    compare_scans(baseline, curr)  -> dict summary of interval change
    flag_discrepancy(visit)        -> dict highlighting report vs vision deltas
    extract_patient_record()       -> PatientRecord written to memory

Each tool reads/writes WorkingMemory by reference and returns a compact
dict for the orchestrator's tool-call loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import BRATS_BACKEND, OUTPUTS_DIR
from ..llm import json_call, vision_json
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.schemas import PatientRecord, VisionObservation
from ..utils.tool_helpers import (
    get_ingestion,
    load_prompt,
    report_text,
    scan_images,
)
from . import register


def _store_vision(memory: WorkingMemory, obs: VisionObservation) -> None:
    bag = memory.get(WorkingMemory.VISION) or {}
    if not isinstance(bag, dict):
        bag = {}
    bag[obs.visit] = obs
    memory.set(WorkingMemory.VISION, bag)


# ---------- analyze_scan ----------
_ANALYZE_SCHEMA_HINT = """
{
  "visit": "v1",
  "findings": [
    {"description": "...", "location": "...", "size_mm": 0.0,
     "enhancement": "none|homogeneous|heterogeneous|ring|nodular"}
  ],
  "impression": "...",
  "mass_effect": false,
  "hemorrhage": false,
  "discrepancy_with_report": false,
  "discrepancy_notes": null
}
"""


@register("analyze_scan")
def analyze_scan(memory: WorkingMemory, visit: str = "v1", **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("mri.analyze_scan", pid=pid, tool="analyze_scan") as _t:
        ing = get_ingestion(memory)
        images = scan_images(ing, visit)
        if not images:
            return {"ok": False, "reason": f"no MRI images for visit {visit}"}

        report = report_text(ing, visit)
        sys_msg = load_prompt("mri_system.md")

        prompt = (
            f"{sys_msg}\n\n"
            f"VISIT: {visit}\n"
            f"RADIOLOGY REPORT (verbatim, may be empty):\n"
            f"-----\n{report or '(none)'}\n-----\n\n"
            f"Analyze the attached MRI image(s). Return ONE JSON object with "
            f"this exact shape (fill `visit` with '{visit}'):\n"
            f"{_ANALYZE_SCHEMA_HINT}"
        )

        try:
            obs = vision_json(prompt, images, VisionObservation)
        except Exception as vision_err:
            # Vision failed (model error / unsupported image format).
            # Fall back: derive a synthetic observation from the text report so
            # extract_patient_record() can still produce a PatientRecord.
            fallback_prompt = (
                f"{sys_msg}\n\n"
                f"VISIT: {visit}\n"
                f"NOTE: MRI images could not be analysed by the vision model "
                f"({type(vision_err).__name__}). Use the radiology report text ONLY.\n\n"
                f"RADIOLOGY REPORT:\n-----\n{report or '(none)'}\n-----\n\n"
                f"Return ONE JSON object with this exact shape:\n{_ANALYZE_SCHEMA_HINT}"
            )
            try:
                obs = json_call(
                    [{"role": "user", "content": fallback_prompt}],
                    VisionObservation,
                )
            except Exception as e2:
                _t.meta["ok"] = False
                return {
                    "ok": False,
                    "error": f"vision: {vision_err!s:.120} | text-fallback: {e2!s:.120}",
                    "fallback_attempted": True,
                }

        # Force the visit field even if the model wandered.
        obs = obs.model_copy(update={"visit": visit})
        _store_vision(memory, obs)
        _t.meta["ok"] = True
        return {
            "ok": True,
            "visit": visit,
            "n_findings": len(obs.findings),
            "mass_effect": obs.mass_effect,
            "hemorrhage": obs.hemorrhage,
            "discrepancy": obs.discrepancy_with_report,
            "impression": obs.impression[:240],
        }


# ---------- compare_scans ----------
@register("compare_scans")
def compare_scans(
    memory: WorkingMemory,
    baseline_visit: str = "v1",
    current_visit: str = "v2",
    **_: Any,
) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("mri.compare_scans", pid=pid, tool="compare_scans") as _t:
        ing = get_ingestion(memory)
        base_imgs = scan_images(ing, baseline_visit)
        curr_imgs = scan_images(ing, current_visit)
        if not base_imgs or not curr_imgs:
            _t.meta["ok"] = False
            return {
                "ok": False,
                "reason": "missing images for one of the visits",
                "have": {"baseline": len(base_imgs), "current": len(curr_imgs)},
            }

        sys_msg = load_prompt("mri_system.md")
        prompt = (
            f"{sys_msg}\n\n"
            f"You will receive a baseline MRI followed by a current MRI for the "
            f"same patient.\n"
            f"Baseline visit: {baseline_visit}\n"
            f"Current visit:  {current_visit}\n\n"
            f"Return ONE JSON object describing INTERVAL CHANGE only, with shape:\n"
            "{\n"
            '  "visit": "%s",\n'
            '  "findings": [{"description":"interval change ...", "location":"...", '
            '"size_mm": 0.0, "enhancement":"..."}],\n'
            '  "impression": "stable|improved|worsened — short rationale",\n'
            '  "mass_effect": false, "hemorrhage": false,\n'
            '  "discrepancy_with_report": false, "discrepancy_notes": null\n'
            "}\n" % current_visit
        )
        images = base_imgs[:1] + curr_imgs[:1]  # one slice each, keep prompt small
        try:
            obs = vision_json(prompt, images, VisionObservation)
        except Exception as e:
            _t.meta["ok"] = False
            return {"ok": False, "error": str(e)[:200]}

        obs = obs.model_copy(update={"visit": f"{baseline_visit}->{current_visit}"})
        bag = memory.get("vision_compare") or {}
        bag[f"{baseline_visit}->{current_visit}"] = obs
        memory.set("vision_compare", bag)

        _t.meta["ok"] = True
        return {
            "ok": True,
            "baseline_visit": baseline_visit,
            "current_visit": current_visit,
            "impression": obs.impression[:240],
            "n_changes": len(obs.findings),
        }


# ---------- flag_discrepancy ----------
@register("flag_discrepancy")
def flag_discrepancy(memory: WorkingMemory, visit: str = "v1", **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("mri.flag_discrepancy", pid=pid, tool="flag_discrepancy") as _t:
        ing = get_ingestion(memory)
        report = report_text(ing, visit)
        bag = memory.get(WorkingMemory.VISION) or {}
        obs = bag.get(visit) if isinstance(bag, dict) else None
        if obs is None:
            _t.meta["ok"] = False
            return {"ok": False, "reason": f"analyze_scan({visit}) must run first"}
        if isinstance(obs, dict):
            obs = VisionObservation.model_validate(obs)
        if not report:
            return {"ok": True, "visit": visit, "discrepancy": False, "reason": "no written report"}

        prompt = (
            "Compare the radiology report below with the vision findings JSON. "
            "Return ONE JSON object: "
            '{"visit": "%s", "findings": [], "impression": "...", '
            '"mass_effect": false, "hemorrhage": false, '
            '"discrepancy_with_report": true|false, '
            '"discrepancy_notes": "short note or null"}\n\n'
            "REPORT:\n-----\n%s\n-----\n\nVISION JSON:\n%s\n"
        ) % (visit, report, obs.model_dump_json())

        try:
            verdict = json_call(
                [{"role": "user", "content": prompt}], VisionObservation
            )
        except Exception as e:
            _t.meta["ok"] = False
            return {"ok": False, "error": str(e)[:200]}

        # Update the stored observation with the discrepancy verdict.
        merged = obs.model_copy(update={
            "discrepancy_with_report": verdict.discrepancy_with_report,
            "discrepancy_notes": verdict.discrepancy_notes,
        })
        bag[visit] = merged
        memory.set(WorkingMemory.VISION, bag)

        _t.meta["ok"] = True
        return {
            "ok": True,
            "visit": visit,
            "discrepancy": merged.discrepancy_with_report,
            "notes": (merged.discrepancy_notes or "")[:240],
        }


# ---------- segment_volumetric (Task 5) ----------
def _find_nifti_files(ing, visit: str) -> list[Path]:
    """Return all NIfTI files for the given visit from the ingested manifest."""
    paths = []
    for f in ing.files:
        if f.visit != visit:
            continue
        if f.kind != "mri_image":
            continue
        p = Path(f.path)
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            paths.append(p)
    return paths


@register("segment_volumetric")
def segment_volumetric(
    memory: WorkingMemory, visit: str = "v1", **_: Any
) -> dict[str, Any]:
    """Submit a 3D volumetric segmentation job for brain MRI (Task 5).

    * GPU present → runs inline (fast, <60 s).
    * CPU only    → queued in a ProcessPoolExecutor; returns immediately with
                    status="queued". The orchestrator continues with the 2D
                    RECIST/RANO path; S04c_volumetric.json is overwritten on
                    disk when the job finishes.
    * No NIfTI    → skips silently (patient has DICOM/PNG only) and writes a
                    placeholder with ``volumetric_unavailable=true``.

    When the result is ``done`` (inline or already-complete queued job), the
    RANO helper ``_assess_rano`` is patched via memory so subsequent RANO
    calls use the accurate ``largest_axial_bidim_mm2`` instead of the
    1D-diameter approximation.
    """
    pid = memory.patient_id
    with stage_timer("mri.segment_volumetric", pid=pid,
                     tool="segment_volumetric") as _t:
        ing = get_ingestion(memory)
        nifti_paths = _find_nifti_files(ing, visit)

        out_dir = OUTPUTS_DIR / pid / "stages"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "S04c_volumetric.json"
        # Phase 5.1 — persist the segmentation mask next to the JSON so
        # radiomics can pick it up without re-running segmentation.
        mask_out = out_dir / f"S04c_mask_{visit}.nii.gz"

        if not nifti_paths:
            # No NIfTI — write a well-formed placeholder so downstream always
            # finds the file; mark unavailable so RANO uses its approximation.
            placeholder = {
                "stage": 4,
                "stage_name": "3D Volumetric Tumour Segmentation (BraTS)",
                "patient_id": pid,
                "visit": visit,
                "volumetric_unavailable": True,
                "reason": "no_nifti_files_in_manifest",
                "backend": BRATS_BACKEND,
                "enhancing_volume_mm3": 0.0,
                "necrotic_volume_mm3":  0.0,
                "edema_volume_mm3":     0.0,
                "largest_axial_bidim_mm2": 0.0,
            }
            out_json.write_text(
                json.dumps(placeholder, indent=2), encoding="utf-8"
            )
            memory.set(WorkingMemory.VOLUMETRIC, placeholder)
            _t.meta["ok"] = True
            _t.meta["skipped"] = True
            return {
                "ok": True,
                "skipped": True,
                "reason": "no_nifti_files",
                "visit": visit,
                "output_json": str(out_json),
            }

        # Use the first NIfTI (typically the post-contrast T1 series).
        volume_path = nifti_paths[0]

        from ..utils.seg_worker import submit

        dispatch = submit(
            volume_path=volume_path,
            output_json=out_json,
            backend=BRATS_BACKEND,
            mask_out_path=mask_out,
        )

        status = dispatch.get("status", "unknown")
        result_payload: dict[str, Any] | None = dispatch.get("result")

        if status == "done" and result_payload:
            # Inline (GPU) run completed — enrich memory immediately.
            _store_volumetric(memory, pid, result_payload, out_json)
            _t.meta["ok"] = True
            _t.meta["ran_inline"] = True
            return {
                "ok": True,
                "status": "done",
                "visit": visit,
                "job_id": dispatch.get("job_id", "inline"),
                "enhancing_volume_mm3":    result_payload.get("enhancing_volume_mm3"),
                "largest_axial_bidim_mm2": result_payload.get("largest_axial_bidim_mm2"),
                "output_json": str(out_json),
            }

        # Queued (CPU path) — store a "pending" placeholder in memory.
        pending = {
            "stage": 4,
            "stage_name": "3D Volumetric Tumour Segmentation (BraTS)",
            "patient_id": pid,
            "visit": visit,
            "status": "queued",
            "job_id": dispatch.get("job_id"),
            "volumetric_unavailable": False,  # will be updated when job finishes
            "backend": BRATS_BACKEND,
            "enhancing_volume_mm3": 0.0,
            "necrotic_volume_mm3":  0.0,
            "edema_volume_mm3":     0.0,
            "largest_axial_bidim_mm2": 0.0,
        }
        out_json.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        memory.set(WorkingMemory.VOLUMETRIC, pending)

        _t.meta["ok"] = True
        _t.meta["queued"] = True
        return {
            "ok": True,
            "status": "queued",
            "visit": visit,
            "job_id": dispatch.get("job_id"),
            "note": (
                "Volumetric segmentation queued in background process. "
                "RANO assessment uses 1D-diameter approximation in the interim. "
                f"Result will land at: {out_json}"
            ),
        }


def _store_volumetric(
    memory: WorkingMemory,
    pid: str,
    result: dict[str, Any],
    out_json: Path,
) -> None:
    """Persist a completed volumetric result to memory and disk."""
    import logging as _log_mod
    log = _log_mod.getLogger(__name__)
    payload = {
        "stage": 4,
        "stage_name": "3D Volumetric Tumour Segmentation (BraTS)",
        "patient_id": pid,
        "volumetric_unavailable": result.get("volumetric_unavailable", False),
        "backend": result.get("backend", BRATS_BACKEND),
        "enhancing_volume_mm3":    result.get("enhancing_volume_mm3", 0.0),
        "necrotic_volume_mm3":     result.get("necrotic_volume_mm3",  0.0),
        "edema_volume_mm3":        result.get("edema_volume_mm3",     0.0),
        "largest_axial_bidim_mm2": result.get("largest_axial_bidim_mm2", 0.0),
        # BUG-FIX (Phase 5.1): mask_path + source were dropped here, so
        # extract_radiomics always saw vol.get("mask_path") == None and
        # silently degraded to "no_mask_available" on every run. Pass them
        # through so the radiomics tool can locate the geometry-aligned
        # mask the segmenter just persisted.
        "mask_path": result.get("mask_path"),
        "source":    result.get("source"),
        "note": result.get("note"),
        "reason": result.get("reason"),
    }
    try:
        out_json.write_text(json.dumps(payload, indent=2, default=str),
                            encoding="utf-8")
    except Exception as exc:
        log.warning("mri: volumetric persist failed: %s", exc)
    memory.set(WorkingMemory.VOLUMETRIC, payload)

    # ── Patch RANO with the accurate bidim (if RANO already in memory) ──────
    bidim = float(result.get("largest_axial_bidim_mm2") or 0.0)
    if bidim > 0:
        rano_raw = memory.get(WorkingMemory.RANO)
        if rano_raw is not None:
            from ..utils.schemas import RANOAssessment
            try:
                rano = (rano_raw if isinstance(rano_raw, RANOAssessment)
                        else RANOAssessment.model_validate(rano_raw))
                rano_updated = rano.model_copy(
                    update={"bidirectional_product_mm2": round(bidim, 1)}
                )
                memory.set(WorkingMemory.RANO, rano_updated)
                log.info(
                    "mri: RANO bidim updated from volumetric segmentation "
                    "(%.1f mm²)", bidim
                )
            except Exception as exc:
                log.warning("mri: RANO patch failed: %s", exc)


# ---------- extract_radiomics (Phase 5.1 / Module 4) ----------
@register("extract_radiomics")
def extract_radiomics(
    memory: WorkingMemory, visit: str = "v1", **_: Any
) -> dict[str, Any]:
    """Compute 5 PyRadiomics features (GLCM texture, shape, first-order
    intensity) from the segmentation mask produced by
    :func:`segment_volumetric`.

    Graceful degrade:
      * pyradiomics / SimpleITK missing   → radiomics_unavailable=true
      * mask file missing (seg skipped or nnU-Net not configured) → idem
      * geometry alignment fails beyond repair → idem

    The result is stored in WorkingMemory under the ``"radiomics"`` key,
    which STAGE_FILE_MAP persists to ``stages/S04d_radiomics.json``. The
    Phase-4 patient-state builder then merges these features into the
    25-dim PatientStateVector (dims 20..24), falling back to population
    medians when any feature is missing.
    """
    from ..config import RADIOMICS_ENABLED
    from ..utils.radiomics_extractor import (
        RADIOMIC_FEATURE_NAMES,
        extract_radiomic_features,
    )

    pid = memory.patient_id
    with stage_timer("mri.extract_radiomics", pid=pid,
                     tool="extract_radiomics") as _t:
        if not RADIOMICS_ENABLED:
            payload = {
                "stage": 4,
                "stage_name": "Radiomic Features Extraction",
                "patient_id": pid,
                "visit": visit,
                "radiomics_unavailable": True,
                "reason": "radiomics_disabled_by_config",
                **{k: None for k in RADIOMIC_FEATURE_NAMES},
            }
            memory.set(WorkingMemory.RADIOMICS, payload)
            _t.meta["ok"] = True
            _t.meta["skipped"] = True
            return {"ok": True, "skipped": True, "reason": payload["reason"]}

        # Pull mask_path from volumetric result stored by segment_volumetric.
        vol_raw = memory.get(WorkingMemory.VOLUMETRIC)
        vol = vol_raw if isinstance(vol_raw, dict) else (
            vol_raw.model_dump() if vol_raw else {}
        )

        # BUG-FIX #2 (Phase 5.1): on CPU/queued runs the worker writes its
        # final result to disk asynchronously, but memory still holds the
        # initial "queued" payload (no mask). Re-read S04c_volumetric.json
        # so a worker that completed since segment_volumetric returned is
        # picked up. Falls back to the in-memory copy when the file is
        # missing or unreadable.
        if vol.get("status") == "queued" or not vol.get("mask_path"):
            try:
                disk_path = OUTPUTS_DIR / pid / "stages" / "S04c_volumetric.json"
                if disk_path.exists():
                    with disk_path.open("r", encoding="utf-8") as f:
                        on_disk = json.load(f)
                    # If the worker completed (mask_path now populated), trust
                    # disk over memory. Otherwise keep the in-memory snapshot.
                    if on_disk.get("mask_path"):
                        vol = on_disk
            except Exception:
                pass

        mask_path = vol.get("mask_path")
        # BUG-FIX #3 (Phase 5.1): prefer the source NIfTI the segmenter
        # actually ran on (recorded in vol["source"]), not the first
        # _find_nifti_files() hit. With multi-sequence visits these can
        # diverge → mask geometry won't match the volume PyRadiomics opens.
        source_path = vol.get("source")
        if not source_path:
            ing = get_ingestion(memory)
            nifti_paths = _find_nifti_files(ing, visit)
            source_path = str(nifti_paths[0]) if nifti_paths else None

        if not mask_path or not source_path:
            payload = {
                "stage": 4,
                "stage_name": "Radiomic Features Extraction",
                "patient_id": pid,
                "visit": visit,
                "radiomics_unavailable": True,
                "reason": (
                    "no_mask_available" if not mask_path
                    else "no_source_nifti"
                ),
                **{k: None for k in RADIOMIC_FEATURE_NAMES},
            }
            memory.set(WorkingMemory.RADIOMICS, payload)
            _t.meta["ok"] = True
            _t.meta["skipped"] = True
            return {"ok": True, "skipped": True, "reason": payload["reason"]}

        features = extract_radiomic_features(source_path, mask_path)
        payload = {
            "stage": 4,
            "stage_name": "Radiomic Features Extraction",
            "patient_id": pid,
            "visit": visit,
            **features,
        }
        memory.set(WorkingMemory.RADIOMICS, payload)
        _t.meta["ok"] = True
        _t.meta["radiomics_unavailable"] = bool(
            features.get("radiomics_unavailable", False)
        )
        return {
            "ok": True,
            "visit": visit,
            "radiomics_unavailable": payload.get("radiomics_unavailable", False),
            "reason": payload.get("reason"),
            **{k: payload.get(k) for k in RADIOMIC_FEATURE_NAMES},
        }


# ---------- extract_patient_record ----------
_RECORD_SCHEMA_HINT = """
{
  "patient_id": "P001",
  "patient_name": null,
  "age": null, "sex": null,
  "diagnosis": null, "diagnosis_date": null,
  "findings": [
    {"description": "...", "location": "...", "size_mm": 0.0, "enhancement": "..."}
  ],
  "impression": "..."
}
"""


@register("extract_patient_record")
def extract_patient_record(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("mri.extract_record", pid=pid, tool="extract_patient_record") as _t:
        ing = get_ingestion(memory)

        # MULTI-PATIENT-FIX: pre-populate name/age/sex/diagnosis_date from
        # the structured intake JSON before letting the LLM hallucinate.
        # Previously the LLM only saw the radiology report → name fell
        # through to "Dear Patient" for every patient and age/sex/dob
        # were filled with whatever the LLM guessed from PDF text.
        from ..utils.patient_state import load_phase4_json
        intake = load_phase4_json(pid, "patient_intake_form.json") or {}
        intake_overrides: dict[str, Any] = {}
        if intake.get("patient_name"):
            intake_overrides["patient_name"] = str(intake["patient_name"]).strip()
        if intake.get("age") is not None:
            try:
                intake_overrides["age"] = int(intake["age"])
            except (TypeError, ValueError):
                pass
        if intake.get("sex"):
            sex = str(intake["sex"]).strip().upper()[:1]
            if sex in ("M", "F"):
                intake_overrides["sex"] = sex
        # P001-MULTI-FIX follow-up: parse ISO date strings into ``date``
        # objects up-front so Pydantic doesn't emit a serialiser warning
        # ("Expected `date` but got `str` with value '2023-10-21'") when
        # PatientRecord round-trips through model_dump.
        from datetime import date as _date
        for k in ("diagnosis_date", "surgery_date", "visit_date"):
            raw = intake.get(k)
            if not raw:
                continue
            try:
                if hasattr(raw, "isoformat"):
                    intake_overrides.setdefault("diagnosis_date", raw)
                else:
                    intake_overrides.setdefault(
                        "diagnosis_date", _date.fromisoformat(str(raw)[:10])
                    )
                break
            except (TypeError, ValueError):
                # Bad/unparseable date — let downstream see no override.
                continue
        if intake.get("cancer_type"):
            intake_overrides.setdefault("diagnosis", str(intake["cancer_type"]))

        # Use the latest visit's report as primary; fall back to all reports.
        latest = sorted({f.visit for f in ing.files})[-1] if ing.files else "v1"
        report = report_text(ing, latest) or "\n\n".join(
            (f.text or "") for f in ing.files if f.kind == "mri_report" and f.text
        )

        bag = memory.get(WorkingMemory.VISION) or {}
        obs_obj = bag.get(latest) if isinstance(bag, dict) else None
        if isinstance(obs_obj, dict):
            obs_obj = VisionObservation.model_validate(obs_obj)
        vision_json_str = obs_obj.model_dump_json() if obs_obj else "{}"

        sys_msg = load_prompt("record_system.md")
        intake_hint = ""
        if intake_overrides:
            intake_hint = (
                "STRUCTURED INTAKE (use these values verbatim — do NOT "
                f"override from text):\n{json.dumps(intake_overrides)}\n\n"
            )
        prompt = (
            f"{sys_msg}\n\n"
            f"PATIENT_ID: {pid}\n\n"
            f"{intake_hint}"
            f"RADIOLOGY REPORT:\n-----\n{report or '(none)'}\n-----\n\n"
            f"VISION OBSERVATION JSON (latest visit):\n{vision_json_str}\n\n"
            f"Return ONE JSON object matching this shape "
            f"(use patient_id='{pid}'):\n{_RECORD_SCHEMA_HINT}"
        )

        try:
            record = json_call(
                [{"role": "user", "content": prompt}], PatientRecord
            )
        except Exception as e:
            _t.meta["ok"] = False
            return {"ok": False, "error": str(e)[:200]}

        # Force the structured intake values to take precedence over the
        # LLM's free-text guesses for identity fields.
        update = {"patient_id": pid, **intake_overrides}
        record = record.model_copy(update=update)
        memory.set(WorkingMemory.RECORD, record)
        _t.meta["ok"] = True
        return {
            "ok": True,
            "patient_name": record.patient_name,
            "diagnosis": record.diagnosis,
            "n_findings": len(record.findings),
            "impression": (record.impression or "")[:240],
        }
