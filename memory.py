"""WorkingMemory — the orchestrator's scratchpad.

All sub-agents write full structured objects here by reference.  The LLM
never sees the full memory dump; it sees compact summaries from
`snapshot_for_llm()`.  Memory is persisted to `outputs/<pid>/` using the
reference naming convention:

    S1_ingestion.json      stage 1 output (file manifest)
    S2_vision.json         stage 2 output (MRI vision)
    S3_record.json         stage 3 output (patient record)
    S4_recist.json         stage 4 output (RECIST/RANO)
    S5_index.json          stage 5 output (ChromaDB index stats)
    S6_urgency.json        stage 6 output (urgency triage)
    S7_medications.json    stage 7 output (medication list)
    S8_interactions.json   stage 8 output (drug interactions)
    S9_correlation.json    stage 9 output (treatment correlation)
    S10_timeline.json      stage 10 output (longitudinal timeline)
    S11_qa_examples.json   stage 11 output (Q&A pairs — written by synthesis)
    S12_summary.json       stage 12 output (patient letter + GP handover)
    S13_export.json        stage 13 output (FHIR R4 export)
    P{pid}_full_pipeline.json   all stage outputs merged into one document
    working_memory.json         raw internal state (job metadata + full store)

Each SN_*.json file is wrapped in a standard stage envelope:
    {
      "stage": <N>,
      "stage_name": "<label>",
      "model": "<MODEL_PRIMARY>",
      "patient_id": "<pid>",
      "generated_at": "<ISO timestamp>",
      ... actual payload keys ...
    }
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import MODEL_PRIMARY, OUTPUTS_DIR, STAGE_FILE_MAP

log = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkingMemory:
    """Typed key-value scratchpad shared across the phase DAG."""

    # Canonical keys written by the phases.
    INGESTION    = "ingestion"
    VISION       = "vision"
    RECORD       = "record"
    RECIST       = "recist"
    RANO         = "rano"          # Task 4: neuro-oncology response criteria
    VOLUMETRIC   = "volumetric"    # Task 5: 3D BraTS/nnUNet segmentation
    RADIOMICS    = "radiomics"     # Phase 5.1: PyRadiomics texture/shape/intensity
    RAG          = "rag"
    URGENCY      = "urgency"
    MEDICATIONS  = "medications"
    INTERACTIONS = "interactions"
    CORRELATION  = "correlation"
    TIMELINE     = "timeline"
    SUMMARY      = "summary"
    EXPORT       = "export"
    QA_EXAMPLES  = "qa_examples"   # S11 — written by synthesis or eval harness
    # Phase 4 — Treatment Optimization (SMBO v3.0)
    PATIENT_STATE      = "patient_state"        # S14
    PREDICTION         = "prediction"           # S15
    OPTIMIZATION       = "optimization"         # S16
    SHAP               = "shap"                 # S17
    TRIAL_MATCHES      = "trial_matches"         # S17b — Task 8
    PUBMED_EVIDENCE    = "pubmed_evidence"       # S17c — Phase 5.5 / Module 3
    FAERS_SIGNALS      = "faers_signals"         # S17d — Phase 5.7 / Extra B
    TREATMENT_PROPOSAL = "treatment_proposal"   # S18

    def __init__(self, job_id: str, patient_id: str):
        self.job_id     = job_id
        self.patient_id = patient_id
        self.created_at = time.time()
        self._store: dict[str, Any] = {}
        self._phase_log: list[dict[str, Any]] = []
        self.out_dir: Path = OUTPUTS_DIR / patient_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- read / write ----------
    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._persist_key(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self._store]
        if missing:
            raise RuntimeError(f"WorkingMemory missing required keys: {missing}")

    def mark_phase(self, phase: str, status: str, steps: int = 0) -> None:
        self._phase_log.append(
            {"phase": phase, "status": status, "steps": steps, "ts": time.time()}
        )

    # ---------- compact view for the LLM ----------
    def snapshot_for_llm(self) -> dict[str, Any]:
        """Small token-friendly view passed as orchestrator context."""
        snap: dict[str, Any] = {
            "patient_id":  self.patient_id,
            "phases_done": [p["phase"] for p in self._phase_log if p["status"] == "ok"],
        }
        ing = self._store.get(self.INGESTION)
        if ing is not None:
            ing_d = _jsonable(ing)
            snap["ingestion"] = {
                "n_files":        len(ing_d.get("files", [])),
                "visits":         ing_d.get("visits", []),
                "has_prior_scans": ing_d.get("has_prior_scans", False),
            }
        rec = self._store.get(self.RECORD)
        if rec is not None:
            rec_d = _jsonable(rec)
            snap["record"] = {
                "diagnosis":  rec_d.get("diagnosis"),
                "impression": (rec_d.get("impression") or "")[:300],
                "n_findings": len(rec_d.get("findings", [])),
            }
        recist = self._store.get(self.RECIST)
        if recist is not None:
            r = _jsonable(recist)
            snap["recist"] = {
                "response":             r.get("response"),
                "pct_change":           r.get("pct_change"),
                "new_lesion_detected":  r.get("new_lesion_detected", False),
                "confirmation_required": r.get("confirmation_required", False),
            }
        vol = self._store.get(self.VOLUMETRIC)
        if vol is not None:
            vd = _jsonable(vol)
            snap["volumetric"] = {
                "enhancing_volume_mm3":    vd.get("enhancing_volume_mm3"),
                "largest_axial_bidim_mm2": vd.get("largest_axial_bidim_mm2"),
                "backend":                 vd.get("backend"),
                "volumetric_unavailable":  vd.get("volumetric_unavailable", True),
                "status":                  vd.get("status", "done"),
            }
        rano = self._store.get(self.RANO)
        if rano is not None:
            ro = _jsonable(rano)
            snap["rano"] = {
                "response":                 ro.get("response"),
                "bidirectional_product_mm2": ro.get("bidirectional_product_mm2"),
                "delta_product_pct":        ro.get("delta_product_pct"),
                "t2_flair_change":          ro.get("t2_flair_change"),
                "corticosteroid_dose_change": ro.get("corticosteroid_dose_change"),
                "neurologic_status":        ro.get("neurologic_status"),
                "criteria_used":            ro.get("criteria_used"),
            }
        urg = self._store.get(self.URGENCY)
        if urg is not None:
            u = _jsonable(urg)
            snap["urgency"] = {"score": u.get("score"), "level": u.get("level")}
        meds = self._store.get(self.MEDICATIONS)
        if meds is not None:
            m = _jsonable(meds)
            snap["medications"] = {
                "n_current":    len(m.get("current", [])),
                "n_historical": len(m.get("historical", [])),
            }
        inter = self._store.get(self.INTERACTIONS)
        if inter is not None:
            i = _jsonable(inter)
            snap["interactions"] = {
                "highest_severity": i.get("highest_severity"),
                "n": len(i.get("interactions", [])),
            }
        # Phase 4 — compact summaries for orchestrator context
        pred = self._store.get(self.PREDICTION)
        if pred is not None:
            p = _jsonable(pred)
            snap["prediction"] = {
                "recist_delta_pred":      p.get("recist_delta_pred"),
                "recist_sigma":           p.get("recist_sigma"),
                "pfs_median_weeks":       p.get("pfs_median_weeks"),
                "optimization_triggered": p.get("optimization_triggered"),
            }
        tm = self._store.get(self.TRIAL_MATCHES)
        if tm is not None:
            td = _jsonable(tm)
            snap["clinical_trials"] = {
                "triggered":     td.get("triggered"),
                "n_top_matches": len(td.get("top_matches", []) or []),
                "top_nct_id": (
                    (td.get("top_matches") or [{}])[0].get("nct_id")
                    if td.get("top_matches") else None
                ),
            }
        prop = self._store.get(self.TREATMENT_PROPOSAL)
        if prop is not None:
            pr = _jsonable(prop)
            snap["treatment_proposal"] = {
                "decision":               pr.get("decision"),
                "proposed_regimen":       pr.get("proposed_regimen"),
                "mdt_discussion_required": pr.get("mdt_discussion_required"),
            }
        return snap

    # ---------- persistence ----------
    def _persist_key(self, key: str, value: Any) -> None:
        """Write one stage output to disk using the S{NN}_name.json convention.

        S01–S18 stages land in ``outputs/<pid>/stages/``; internal/transient
        keys (e.g. ``recist_lesions``) stay at the patient root for debug.
        """
        from .config import STAGE_SUBDIR_MAP, patient_out_dir
        payload = _jsonable(value)

        # Determine filename and stage envelope from the map.
        if key in STAGE_FILE_MAP:
            stem, stage_num, stage_label = STAGE_FILE_MAP[key]
            filename = f"{stem}.json"
            # Wrap in standard stage envelope (mirroring reference dataset format).
            wrapped: dict[str, Any] = {
                "stage":        stage_num,
                "stage_name":   stage_label,
                "model":        MODEL_PRIMARY,
                "patient_id":   self.patient_id,
                "generated_at": _now_iso(),
            }
            # Merge payload — if it's a dict, spread it; otherwise store under "data".
            if isinstance(payload, dict):
                wrapped.update(payload)
            else:
                wrapped["data"] = payload
        else:
            # Internal or transient key (e.g. "recist_lesions") — plain JSON, no envelope.
            filename = f"{key}.json"
            wrapped = payload if isinstance(payload, dict) else {"data": payload}

        # Route into the correct subfolder (stages/ for numbered stages, patient
        # root for internal keys) — legacy files at the root are removed.
        kind = STAGE_SUBDIR_MAP.get(key)
        target_dir = patient_out_dir(self.patient_id, kind) if kind else self.out_dir
        path = target_dir / filename
        path.write_text(
            json.dumps(wrapped, indent=2, default=str),
            encoding="utf-8",
        )
        # Back-compat: remove a stale copy at the patient root if we just wrote to a subdir.
        if kind:
            stale = self.out_dir / filename
            if stale.exists() and stale.resolve() != path.resolve():
                try:
                    stale.unlink()
                except Exception:
                    pass

    def persist_internal_snapshot(self) -> Path:
        """Write a current snapshot of working_memory.json on disk.

        Idempotent — overwriting itself is fine. ``finalize()`` will
        re-write it later with the same path.

        Why this exists
        ---------------
        Drive sync runs inside ``synthesis_agent`` *before* ``finalize()``.
        Without this, working_memory.json doesn't yet exist when the sync
        iterates the outputs directory — so the skip-list in
        ``DriveClient.sync_patient_outputs`` (which excludes internal-state
        files by filename) has nothing to skip. By calling this method
        immediately before drive sync, the file exists on disk and the
        skip-list correctly excludes it from the patient-shared folder.
        """
        from .config import patient_out_dir
        master_dir = patient_out_dir(self.patient_id, "master")
        master_dir.mkdir(parents=True, exist_ok=True)
        wm_path = master_dir / "working_memory.json"
        wm_path.write_text(
            json.dumps(
                {
                    "job_id":     self.job_id,
                    "patient_id": self.patient_id,
                    "phases":     self._phase_log,
                    "store":      _jsonable(self._store),
                    "_snapshot_only": True,   # cleared by finalize()
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        return wm_path

    def finalize(self) -> Path:
        """Write master/working_memory.json, master/{pid}_master.json,
        master/run_manifest.json, and the legacy P{pid}_full_pipeline.json."""
        from .config import patient_out_dir
        master_dir = patient_out_dir(self.patient_id, "master")

        # 1. Raw internal state — for debugging / eval harness.
        wm_path = master_dir / "working_memory.json"
        wm_path.write_text(
            json.dumps(
                {
                    "job_id":     self.job_id,
                    "patient_id": self.patient_id,
                    "phases":     self._phase_log,
                    "store":      _jsonable(self._store),
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        # Remove legacy copy at patient root if it exists.
        legacy_wm = self.out_dir / "working_memory.json"
        if legacy_wm.exists() and legacy_wm.resolve() != wm_path.resolve():
            try:
                legacy_wm.unlink()
            except Exception:
                pass

        # 2. Full pipeline document — mirrors P{pid}_full_pipeline.json in the
        #    reference dataset: top-level patient summary + all stage outputs merged.
        pipeline: dict[str, Any] = {
            "patient_id":     self.patient_id,
            "generated_at":   _now_iso(),
            "pipeline_outputs": {},
        }

        # Pull basic demographics from record if available.
        rec_raw = self._store.get(self.RECORD)
        if rec_raw:
            rec = _jsonable(rec_raw)
            patient_d = rec.get("patient") or {}
            if not isinstance(patient_d, dict):
                patient_d = {}
            # MULTI-PATIENT-FIX: prefer the flat ``patient_name`` field on
            # PatientRecord (populated from intake by extract_patient_record).
            # Fall back to legacy nested ``patient.name`` for back-compat,
            # then to patient_id as a last resort.
            pipeline["patient_name"] = (
                rec.get("patient_name")
                or patient_d.get("name")
                or rec.get("patient_id", self.patient_id)
            )
            # diagnosis may be a nested dict {"histology": ..., "who_grade": ...}
            # or a plain string from LLM variability — handle both safely.
            diag = rec.get("diagnosis")
            if isinstance(diag, dict):
                pipeline["tumor_type"] = diag.get("histology") or "unknown"
                pipeline["who_grade"]  = diag.get("who_grade") or "unknown"
            elif isinstance(diag, str):
                pipeline["tumor_type"] = diag or "unknown"
                pipeline["who_grade"]  = "unknown"
            else:
                pipeline["tumor_type"] = "unknown"
                pipeline["who_grade"]  = "unknown"
        else:
            pipeline["patient_name"]  = self.patient_id
            pipeline["tumor_type"]    = "unknown"
            pipeline["who_grade"]     = "unknown"

        # Disease trend from RECIST.
        recist_raw = self._store.get(self.RECIST)
        if recist_raw:
            r = _jsonable(recist_raw)
            resp = r.get("response", "NE") if isinstance(r, dict) else "NE"
            pipeline["disease_trend"] = (
                "progressive" if resp == "PD"
                else "responding"  if resp in {"PR", "CR"}
                else "stable"      if resp == "SD"
                else "not_evaluable"
            )
        else:
            pipeline["disease_trend"] = "unknown"

        # Embed all stage outputs under pipeline_outputs (from stages/ subdir).
        stages_dir = patient_out_dir(self.patient_id, "stages")
        for key, (stem, stage_num, stage_label) in STAGE_FILE_MAP.items():
            # Prefer new stages/ location; fall back to patient root for older runs.
            candidates = [stages_dir / f"{stem}.json", self.out_dir / f"{stem}.json"]
            for stage_file in candidates:
                if stage_file.exists():
                    try:
                        pipeline["pipeline_outputs"][stem] = json.loads(
                            stage_file.read_text(encoding="utf-8")
                        )
                        break
                    except Exception as exc:
                        log.warning("finalize: failed to load %s: %s", stage_file.name, exc)

        # Include lab results (synthesis extended output).
        ext_dir = patient_out_dir(self.patient_id, "extended")
        lab_candidates = [ext_dir / "laboratory_results.json",
                          self.out_dir / "laboratory_results.json"]
        for lab_file in lab_candidates:
            if lab_file.exists():
                try:
                    pipeline["laboratory_results"] = json.loads(
                        lab_file.read_text(encoding="utf-8")
                    )
                    break
                except Exception as exc:
                    log.warning("finalize: lab load failed: %s", exc)

        # Include extended/ subfolder files.
        if ext_dir.exists():
            pipeline["extended"] = {}
            for ext_file in sorted(ext_dir.iterdir()):
                if not ext_file.is_file() or ext_file.name == "laboratory_results.json":
                    continue
                try:
                    if ext_file.suffix == ".json":
                        pipeline["extended"][ext_file.stem] = json.loads(
                            ext_file.read_text(encoding="utf-8")
                        )
                    else:
                        pipeline["extended"][ext_file.stem] = ext_file.read_text(encoding="utf-8")
                except Exception as exc:
                    log.warning("finalize: extended load failed for %s: %s",
                                ext_file.name, exc)

        # ---- 2a. Canonical master JSON (new) ----
        master_path = master_dir / f"{self.patient_id}_master.json"
        master_path.write_text(
            json.dumps(pipeline, indent=2, default=str),
            encoding="utf-8",
        )

        # ---- 2b. Legacy full_pipeline.json at the patient root for back-compat ----
        pp_path = self.out_dir / f"{self.patient_id}_full_pipeline.json"
        pp_path.write_text(
            json.dumps(pipeline, indent=2, default=str),
            encoding="utf-8",
        )

        # ---- 3. Executive summary (qwen3:14b, best-effort plain-English) ----
        try:
            exec_text = self._generate_executive_summary(pipeline)
            if exec_text:
                (master_dir / "executive_summary.txt").write_text(
                    exec_text, encoding="utf-8"
                )
        except Exception as exc:
            log.warning("finalize: executive summary failed: %s", exc)

        # ---- 4. Legacy-root cleanup: remove stale unpadded / misplaced copies ----
        #    Safe only when a canonical version exists under its new subdir.
        try:
            self._cleanup_legacy_root()
        except Exception as exc:
            log.warning("finalize: legacy cleanup failed: %s", exc)

        # ---- 4b. Phase 5.3 / Module 2 — Longitudinal history append ----
        try:
            self._append_longitudinal_visit()
        except Exception as exc:
            log.warning("finalize: longitudinal append failed: %s", exc)

        # ---- 4c. Phase 5.4 / Module 1 — best-effort flush of background
        #          LightRAG graph builds (timeout-bounded; partial graphs
        #          are append-capable on next run, so we never block long).
        try:
            from .config import LIGHTRAG_FLUSH_TIMEOUT_S
            from .utils import graph_worker
            still_pending = graph_worker.flush(timeout=float(LIGHTRAG_FLUSH_TIMEOUT_S))
            if still_pending:
                log.info("finalize: %d LightRAG build(s) still running after flush — "
                         "they continue in background and will reach 'ready' on disk",
                         still_pending)
        except Exception as exc:
            log.warning("finalize: LightRAG flush failed: %s", exc)

        # ---- 5. Run manifest (file inventory + phase log) ----
        manifest = self._build_manifest(master_dir)
        (master_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )

        return wm_path

    def _append_longitudinal_visit(self) -> None:
        """Phase 5.3 / Module 2 — append the current visit to
        ``outputs/<pid>/history/longitudinal.jsonl``.

        Pulls SoD, RECIST response, predicted PFS, and the normalised vector
        from working memory. Silent no-op if those slots are empty (e.g.
        early-exit runs that never reached treatment-opt).
        """
        from .utils.longitudinal_history import append_visit

        # Pull what we can from working memory; everything is optional.
        sod = 0.0
        recist_resp = None
        try:
            recist_raw = self._store.get(self.RECIST)
            if recist_raw:
                rec = _jsonable(recist_raw)
                sod = float(rec.get("current_sum_mm") or 0.0)
                recist_resp = rec.get("response")
        except Exception:
            pass

        pfs_med = None
        try:
            pred_raw = self._store.get(self.PREDICTION)
            if pred_raw:
                pred = _jsonable(pred_raw)
                pfs_med = pred.get("pfs_median_weeks")
        except Exception:
            pass

        cancer_type = None
        normalized: list[float] = []
        try:
            ps_raw = self._store.get(self.PATIENT_STATE)
            if ps_raw:
                ps = _jsonable(ps_raw)
                cancer_type = ps.get("cancer_type")
                normalized = list(ps.get("normalized") or [])
        except Exception:
            pass

        path = append_visit(
            self.out_dir, self.patient_id,
            visit_id=self.job_id,
            sum_of_diameters_mm=sod,
            pfs_median_weeks=pfs_med,
            recist_response=recist_resp,
            cancer_type=cancer_type,
            normalized=normalized,
        )
        log.info("longitudinal: appended visit %s for %s -> %s",
                 self.job_id, self.patient_id, path)

    def _cleanup_legacy_root(self) -> None:
        """Remove stale pre-rename files at the patient root.

        Only deletes when the canonical copy exists under the expected subdir —
        never destroys data that isn't duplicated elsewhere.
        """
        from .config import patient_out_dir
        root = self.out_dir
        stages_dir   = patient_out_dir(self.patient_id, "stages")
        reports_dir  = patient_out_dir(self.patient_id, "reports")
        fhir_dir     = patient_out_dir(self.patient_id, "fhir")
        extended_dir = patient_out_dir(self.patient_id, "extended")
        notif_dir    = patient_out_dir(self.patient_id, "notifications")

        # Every known zero-padded stage filename from STAGE_FILE_MAP — plus its
        # unpadded legacy form (e.g. S04_recist -> S4_recist).
        canonical_stems: set[str] = set()
        for _k, (stem, num, _lbl) in STAGE_FILE_MAP.items():
            canonical_stems.add(stem)                      # S04_recist
            canonical_stems.add(f"S{num}_{stem.split('_', 1)[1]}")  # S4_recist

        # Root-level files with a known new home
        root_to_subdir = {
            "fhir_bundle.json":        fhir_dir,
            "mdt_package.json":        reports_dir,
            "report.md":               reports_dir,
            "patient_letter.txt":      reports_dir,
            "gp_handover.txt":         reports_dir,
            "laboratory_results.json": extended_dir,
            "notifications_gmail.json":    notif_dir,
            "notifications_sync.json":     notif_dir,
            "notifications_phase4.json":   notif_dir,
            "working_memory.json":     patient_out_dir(self.patient_id, "master"),
        }

        # Task 9 HITL markers MUST survive cleanup — they live at the patient
        # root by design and are the protocol between prep and execute modes.
        HITL_PROTECTED = {"PENDING_APPROVAL.json", "APPROVED.json", "REJECTED.json"}

        removed = 0
        for p in list(root.iterdir()):
            if not p.is_file():
                continue
            stem = p.stem
            name = p.name
            if name in HITL_PROTECTED:
                continue

            # 4a. Stage file at root — move into stages/ if canonical missing, else delete.
            if stem in canonical_stems and p.suffix == ".json":
                # Derive the padded canonical filename (S04_recist etc.)
                pid_num = stem.split("_", 1)[0].lstrip("S")
                rest = stem.split("_", 1)[1] if "_" in stem else ""
                try:
                    padded = f"S{int(pid_num):02d}_{rest}.json"
                except ValueError:
                    continue
                padded_path = stages_dir / padded
                if padded_path.exists() and padded_path.resolve() != p.resolve():
                    p.unlink()
                    removed += 1
                else:
                    stages_dir.mkdir(parents=True, exist_ok=True)
                    p.rename(padded_path)
                    removed += 1
                continue

            # 4b. Other well-known root files — move into proper subdir (or delete dup).
            if name in root_to_subdir:
                sub = root_to_subdir[name]
                dup = sub / name
                if dup.exists() and dup.resolve() != p.resolve():
                    p.unlink()
                    removed += 1
                else:
                    sub.mkdir(parents=True, exist_ok=True)
                    p.rename(dup)
                    removed += 1

            # 4c. Stray ancient outputs like recist_lesions.json (never canonical).
            if name == "recist_lesions.json":
                # canonical copy lives in stages/S05_index.json era or not at all
                p.unlink()
                removed += 1

        if removed:
            log.info("finalize: legacy cleanup removed %d stale root files for %s",
                     removed, self.patient_id)

    def _generate_executive_summary(self, pipeline: dict[str, Any]) -> str:
        """Build a compact snapshot and call qwen3:14b for a ≤200-word summary."""
        from .utils.llm_enrichment import executive_summary
        # Extract a minimal high-signal snapshot from the pipeline dict.
        outputs = pipeline.get("pipeline_outputs", {}) or {}
        recist = (outputs.get("S04_recist") or {}) if isinstance(outputs, dict) else {}
        urgency = (outputs.get("S06_urgency") or {}) if isinstance(outputs, dict) else {}
        pred   = (outputs.get("S15_prediction") or {})
        prop   = (outputs.get("S18_treatment_proposal") or {})
        shap   = (outputs.get("S17_shap") or {})
        top5   = shap.get("top_5_drivers") or []
        snap = {
            "patient_id":       pipeline.get("patient_id"),
            "diagnosis":        pipeline.get("tumor_type"),
            "who_grade":        pipeline.get("who_grade"),
            "disease_trend":    pipeline.get("disease_trend"),
            "recist_response":  recist.get("response"),
            "new_lesion":       recist.get("new_lesion_detected"),
            "urgency_level":    urgency.get("level"),
            "urgency_score":    urgency.get("score"),
            "pfs_median_weeks": pred.get("pfs_median_weeks"),
            "recist_delta_pred": pred.get("recist_delta_pred"),
            "mdt_decision":     prop.get("decision"),
            "proposed_regimen": prop.get("proposed_regimen"),
            "mdt_discussion_required": prop.get("mdt_discussion_required"),
            "top_shap_drivers": [
                {"feature": d.get("feature"),
                 "impact_weeks": round(float(d.get("shap_value", 0.0)), 1),
                 "direction": d.get("direction")}
                for d in top5[:5]
            ],
            "shap_narrative":   shap.get("narrative", ""),
            "audit_concerns":   prop.get("audit_concerns", []),
            "wearable_trend":   self._store.get("wearable_narrative") or "",
        }
        return executive_summary(snap)

    def _build_manifest(self, master_dir: Path) -> dict[str, Any]:
        """Snapshot every file written under outputs/<pid>/ with size + mtime."""
        inventory: dict[str, list[dict[str, Any]]] = {}
        base = self.out_dir
        if base.exists():
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    rel_parts = p.relative_to(base).parts
                except ValueError:
                    continue
                bucket = rel_parts[0] if len(rel_parts) > 1 else "_root"
                inventory.setdefault(bucket, []).append({
                    "path":       str(p.relative_to(base)).replace("\\", "/"),
                    "size_bytes": p.stat().st_size,
                    "mtime":      _now_iso() if not p.exists() else None,
                })
        return {
            "job_id":       self.job_id,
            "patient_id":   self.patient_id,
            "generated_at": _now_iso(),
            "phases":       self._phase_log,
            "output_root":  str(base),
            "inventory":    inventory,
        }

    @classmethod
    def load(cls, patient_id: str) -> "WorkingMemory":
        from .config import patient_out_dir
        mem = cls(job_id="restored", patient_id=patient_id)
        # Canonical location is master/working_memory.json; legacy copies at
        # the patient root are checked as a fallback.
        candidates = [
            patient_out_dir(patient_id, "master") / "working_memory.json",
            mem.out_dir / "working_memory.json",
        ]
        for path in candidates:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                mem._store     = data.get("store", {})
                mem._phase_log = data.get("phases", [])
                break
        return mem
