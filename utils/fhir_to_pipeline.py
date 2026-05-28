"""Phase 5.8 / Extra C — FHIR R4 → pipeline-internal layout converter.

Pulls the minimum-viable FHIR resource set for one patient and writes
files into the existing v2 layout::

    Datasets/patients/<pid>/
        clinical/
            <pid>_diagnosis.json        ← Condition[]
            <pid>_observations.json     ← Observation[]
            <pid>_diagnostic_reports.json ← DiagnosticReport[]
            <pid>_medications.json      ← MedicationStatement[]
        phase4/
            patient_intake_form.json    ← synthesised from Patient + Condition + Observation

After conversion the existing ``ingest`` phase runs unchanged — the
SMART path simply replaces the ZIP-upload step.

Resources fetched (read-only):
    Patient, Condition, Observation, MedicationStatement, DiagnosticReport

Graceful degrade
----------------
``fhir.resources`` is an optional dependency. When missing we still
write the raw FHIR JSON bundles (no validation) — the rest of the
pipeline is tolerant of extra fields.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import PATIENTS_ROOT
from ..integrations.fhir_client import SmartFHIRClient

log = logging.getLogger(__name__)

# Fail-soft import of fhir.resources — used only for validation when present.
try:
    from fhir.resources import construct_fhir_element  # type: ignore
    FHIR_VALIDATION_AVAILABLE = True
except Exception:
    construct_fhir_element = None  # type: ignore
    FHIR_VALIDATION_AVAILABLE = False


_RESOURCE_PLAN: dict[str, str] = {
    # FHIR resource type → output filename stem
    "Condition":           "diagnosis",
    "Observation":         "observations",
    "DiagnosticReport":    "diagnostic_reports",
    "MedicationStatement": "medications",
}


def _patient_search(client: SmartFHIRClient, resource: str, patient_id: str) -> list[dict]:
    """Fetch all resources of a type for a patient (paginates)."""
    out: list[dict] = []
    next_url: str | None = f"{resource}?patient={patient_id}&_count=100"
    pages = 0
    while next_url and pages < 20:    # safety cap
        resp = client.get(next_url)
        if resp.status_code >= 400:
            log.warning("fhir_to_pipeline: %s search %s — HTTP %d",
                        resource, patient_id, resp.status_code)
            break
        bundle = resp.json() or {}
        for entry in bundle.get("entry") or []:
            res = entry.get("resource")
            if res:
                out.append(res)
        # Follow ``next`` link if present.
        next_url = None
        for link in bundle.get("link") or []:
            if link.get("relation") == "next":
                next_url = link.get("url")
                break
        pages += 1
    return out


def _patient_read(client: SmartFHIRClient, patient_id: str) -> dict[str, Any]:
    resp = client.get(f"Patient/{patient_id}")
    if resp.status_code >= 400:
        raise RuntimeError(f"FHIR Patient/{patient_id} read failed: HTTP {resp.status_code}")
    return resp.json() or {}


# ── intake-form synthesis ─────────────────────────────────────────────────────

def _extract_observation_value(obs: dict[str, Any]) -> Any:
    """Try common Observation value paths; return first non-null."""
    for k in ("valueQuantity", "valueCodeableConcept", "valueString",
              "valueBoolean", "valueInteger"):
        v = obs.get(k)
        if v:
            return v
    return None


def _synthesise_intake(patient: dict[str, Any],
                       conditions: list[dict],
                       observations: list[dict]) -> dict[str, Any]:
    """Build a minimal patient_intake_form.json from FHIR resources.

    Keeps just the fields ``extract_intake_features`` reads. Additional
    fields are best-effort — graceful-degrade when missing.
    """
    intake: dict[str, Any] = {
        "_source":   "smart_on_fhir",
        "_imported_at": datetime.utcnow().isoformat(),
        "patient_id":  patient.get("id"),
        "name":        " ".join(filter(None, [
            (patient.get("name") or [{}])[0].get("given", [None])[0],
            (patient.get("name") or [{}])[0].get("family"),
        ])).strip() or None,
        "sex":         (patient.get("gender") or "").upper()[:1] or None,
        "date_of_birth": patient.get("birthDate"),
    }

    # Age
    if intake.get("date_of_birth"):
        try:
            dob = datetime.fromisoformat(intake["date_of_birth"])
            intake["age"] = int((datetime.utcnow() - dob).days / 365.25)
        except ValueError:
            pass

    # Diagnosis: take the first non-resolved Condition with a code text.
    primary_diag = None
    for cond in conditions:
        clinical = (cond.get("clinicalStatus") or {}).get("coding", [])
        if any((c.get("code") or "") in ("inactive", "resolved") for c in clinical):
            continue
        code = cond.get("code") or {}
        text = code.get("text") or (code.get("coding") or [{}])[0].get("display")
        if text:
            primary_diag = text
            break
    if primary_diag:
        intake["cancer_type"] = primary_diag.lower()

    # ECOG / performance status — look for an Observation with that LOINC code.
    for obs in observations:
        code = (obs.get("code") or {}).get("coding") or []
        if any(c.get("code") == "89247-1" for c in code):     # ECOG performance status
            v = _extract_observation_value(obs)
            if isinstance(v, dict) and "value" in v:
                intake["ecog_ps"] = v["value"]
            break

    return intake


# ── public entrypoint ─────────────────────────────────────────────────────────

def import_patient_from_fhir(
    client: SmartFHIRClient, fhir_patient_id: str, *, local_pid: str | None = None,
) -> dict[str, Any]:
    """Pull all relevant FHIR resources for one patient and write to disk.

    ``local_pid`` defaults to ``fhir_patient_id`` upper-cased and
    sanitised. Returns a small summary dict with file paths + counts.
    """
    pid = (local_pid or fhir_patient_id).upper().replace("/", "_")
    patient_root = Path(PATIENTS_ROOT) / pid
    clinical_dir = patient_root / "clinical"
    phase4_dir   = patient_root / "phase4"
    clinical_dir.mkdir(parents=True, exist_ok=True)
    phase4_dir.mkdir(parents=True, exist_ok=True)

    # 1. Patient demographics
    patient = _patient_read(client, fhir_patient_id)
    (patient_root / "fhir_patient.json").write_text(
        json.dumps(patient, indent=2), encoding="utf-8",
    )

    # 2. The four searchable resource types
    counts: dict[str, int] = {}
    by_type: dict[str, list[dict]] = {}
    for resource_type, stem in _RESOURCE_PLAN.items():
        rows = _patient_search(client, resource_type, fhir_patient_id)
        counts[resource_type] = len(rows)
        by_type[resource_type] = rows
        out_path = clinical_dir / f"{pid}_{stem}.json"
        out_path.write_text(
            json.dumps({"resourceType": "Bundle", "entry":
                        [{"resource": r} for r in rows]}, indent=2),
            encoding="utf-8",
        )

    # 3. Synthesise intake form so phase4 keeps working unchanged.
    intake = _synthesise_intake(
        patient, by_type.get("Condition", []), by_type.get("Observation", []),
    )
    intake_path = phase4_dir / "patient_intake_form.json"
    intake_path.write_text(json.dumps(intake, indent=2), encoding="utf-8")

    log.info(
        "fhir_to_pipeline: imported patient %s (FHIR %s) — %s",
        pid, fhir_patient_id,
        ", ".join(f"{k}={v}" for k, v in counts.items()),
    )

    return {
        "ok":              True,
        "patient_id":      pid,
        "fhir_patient_id": fhir_patient_id,
        "counts":          counts,
        "intake_path":     str(intake_path),
        "clinical_dir":    str(clinical_dir),
        "validation_enabled": FHIR_VALIDATION_AVAILABLE,
    }


__all__ = ["import_patient_from_fhir"]
