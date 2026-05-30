"""Patient State Vectorization — Sub-step 4a utility.

Builds the 20-dimensional normalised PatientState vector from:
  • patient_intake_form.json  (structured JSON from phase4_patient_data/)
  • wearable_data.json        (Fitbit stream from phase4_patient_data/)
  • lab PDF text              (fallback when intake_form absent)
  • WorkingMemory             (RECIST, record already extracted by phases 2-3)

Normalisation: min-max [0,1] using reference_ranges.json population norms.
Missing values are imputed from the cancer-type median and flagged in
imputation_mask (1 = imputed, 0 = observed).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import (
    BIOMARKER_REQUIRED_CANCERS,
    PHASE4_DATA_ROOT,
    REFERENCE_RANGES_PATH,
)
from ..memory import WorkingMemory
from ..utils.schemas import PatientStateVector

log = logging.getLogger(__name__)

# ── dimension order (must stay stable — models trained on this ordering) ───────
FEATURE_NAMES: list[str] = [
    # Tumour burden (4)
    "sum_of_diameters_mm", "delta_sod_pct", "lesion_count", "new_lesion_flag",
    # Lab markers (6)
    "ldh_u_per_l", "crp_mg_per_l", "nlr", "hemoglobin_g_per_dl",
    "albumin_g_per_dl", "egfr_ml_per_min",
    # Wearable vitals (5)
    "daily_steps_7d_avg", "resting_hr_bpm", "sleep_hours_7d_avg",
    "hrv_ms", "ecog_ps_score",
    # Treatment history (5)
    "treatment_cycles_completed", "days_since_last_dose",
    "total_prior_lines", "dose_reduction_flag", "treatment_duration_weeks",
    # Radiomics (5) — Phase 5.1 / Module 4
    "glcm_contrast", "glcm_correlation",
    "shape_sphericity", "shape_surface_volume_ratio",
    "firstorder_entropy",
    # Longitudinal trajectory (4) — Phase 5.3 / Module 2
    "sod_growth_rate_mm_per_week", "pfs_trajectory_slope",
    "treatment_response_streak", "visit_count",
]
assert len(FEATURE_NAMES) == 29, "Feature vector must be exactly 29 dimensions"

# ── reference ranges cache ──────────────────────────────────────────────────────
_REF_CACHE: dict[str, Any] | None = None

# Hardcoded fallback ranges (used when reference_ranges.json is absent)
_FALLBACK_RANGES: dict[str, dict[str, float]] = {
    "sum_of_diameters_mm":        {"p5": 5.0,   "p95": 120.0, "median": 35.0},
    "delta_sod_pct":              {"p5": -50.0, "p95": 50.0,  "median": 5.0},
    "lesion_count":               {"p5": 1.0,   "p95": 6.0,   "median": 2.0},
    "new_lesion_flag":            {"p5": 0.0,   "p95": 1.0,   "median": 0.0},
    "ldh_u_per_l":                {"p5": 120.0, "p95": 520.0, "median": 218.0},
    "crp_mg_per_l":               {"p5": 1.0,   "p95": 80.0,  "median": 12.0},
    "nlr":                        {"p5": 1.5,   "p95": 9.5,   "median": 3.5},
    "hemoglobin_g_per_dl":        {"p5": 8.5,   "p95": 14.5,  "median": 11.5},
    "albumin_g_per_dl":           {"p5": 2.8,   "p95": 4.5,   "median": 3.6},
    "egfr_ml_per_min":            {"p5": 45.0,  "p95": 110.0, "median": 75.0},
    "daily_steps_7d_avg":         {"p5": 400.0, "p95": 7500.0,"median": 3200.0},
    "resting_hr_bpm":             {"p5": 52.0,  "p95": 92.0,  "median": 72.0},
    "sleep_hours_7d_avg":         {"p5": 4.5,   "p95": 9.0,   "median": 6.5},
    "hrv_ms":                     {"p5": 12.0,  "p95": 55.0,  "median": 28.0},
    "ecog_ps_score":              {"p5": 0.0,   "p95": 4.0,   "median": 1.0},
    "treatment_cycles_completed": {"p5": 1.0,   "p95": 12.0,  "median": 4.0},
    "days_since_last_dose":       {"p5": 0.0,   "p95": 90.0,  "median": 14.0},
    "total_prior_lines":          {"p5": 0.0,   "p95": 4.0,   "median": 1.0},
    "dose_reduction_flag":        {"p5": 0.0,   "p95": 1.0,   "median": 0.0},
    "treatment_duration_weeks":   {"p5": 2.0,   "p95": 52.0,  "median": 16.0},
    # Radiomics (Phase 5.1). Ranges seeded from published GBM cohorts
    # (Hajianfar 2022, Fathi Kazerooni 2021, BraTS 2021 validation set);
    # refine when we accumulate our own distribution.
    "glcm_contrast":              {"p5": 5.0,   "p95": 450.0, "median": 80.0},
    "glcm_correlation":           {"p5": 0.05,  "p95": 0.85,  "median": 0.40},
    "shape_sphericity":           {"p5": 0.30,  "p95": 0.90,  "median": 0.60},
    "shape_surface_volume_ratio": {"p5": 0.10,  "p95": 1.20,  "median": 0.45},
    "firstorder_entropy":         {"p5": 2.5,   "p95": 6.5,   "median": 4.5},
    # Longitudinal trajectory (Phase 5.3). Ranges seeded from typical
    # GBM cohort visit cadence (4-8 weeks) and CTRP/REMBRANDT trajectory
    # distributions; refine once we accumulate longitudinal observations.
    "sod_growth_rate_mm_per_week":{"p5": -3.0,  "p95": 4.0,   "median": 0.0},
    "pfs_trajectory_slope":       {"p5": -2.5,  "p95": 1.0,   "median": -0.2},
    "treatment_response_streak":  {"p5": 0.0,   "p95": 6.0,   "median": 1.0},
    "visit_count":                {"p5": 0.0,   "p95": 8.0,   "median": 1.0},
}

# The 5 radiomic feature names — referenced by build_patient_state_vector.
RADIOMIC_FEATURE_NAMES: list[str] = [
    "glcm_contrast", "glcm_correlation",
    "shape_sphericity", "shape_surface_volume_ratio",
    "firstorder_entropy",
]

# Brand-name → generic name alias map for common oncology drugs
_DRUG_ALIASES: dict[str, str] = {
    "avastin":    "bevacizumab",
    "temodar":    "temozolomide",
    "ccnu":       "lomustine",
    "decadron":   "dexamethasone",
    "keppra":     "levetiracetam",
    "dilantin":   "phenytoin",
    "rituxan":    "rituximab",
    "opdivo":     "nivolumab",
    "keytruda":   "pembrolizumab",
    "tagrisso":   "osimertinib",
}


def _load_reference_ranges() -> dict[str, Any]:
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    try:
        data = json.loads(REFERENCE_RANGES_PATH.read_text(encoding="utf-8"))
        _REF_CACHE = data
        return data
    except Exception as exc:
        log.warning("patient_state: could not load reference_ranges.json: %s — using fallback", exc)
        _REF_CACHE = {}
        return {}


def _get_range(feature: str, cancer_type: str) -> dict[str, float]:
    """Return {p5, p95, median} for this feature × cancer_type.

    Falls back: cancer-type-specific → 'all' → hardcoded fallback.
    """
    ref = _load_reference_ranges()
    ct_key = cancer_type.lower().replace(" ", "_").replace("-", "_")

    # Try cancer-type-specific ranges first
    for block in ref.get("cancer_types", []):
        if block.get("cancer_type", "").lower().replace(" ", "_") == ct_key:
            dims = block.get("dimensions", {})
            if feature in dims:
                d = dims[feature]
                return {
                    "p5":    float(d.get("p5",    _FALLBACK_RANGES.get(feature, {}).get("p5",    0.0))),
                    "p95":   float(d.get("p95",   _FALLBACK_RANGES.get(feature, {}).get("p95",   1.0))),
                    "median":float(d.get("median", _FALLBACK_RANGES.get(feature, {}).get("median",0.5))),
                }

    # Fallback to global hardcoded ranges
    fb = _FALLBACK_RANGES.get(feature, {"p5": 0.0, "p95": 1.0, "median": 0.5})
    return dict(fb)


def _normalize_value(val: float | None, feature: str, cancer_type: str) -> tuple[float, int]:
    """Normalise a raw value to [0,1] min-max. Returns (norm_val, imputed_flag)."""
    ranges = _get_range(feature, cancer_type)
    p5, p95, median = ranges["p5"], ranges["p95"], ranges["median"]

    if val is None or (isinstance(val, float) and (val != val)):  # None or NaN
        # Impute with population median
        val = median
        imputed = 1
    else:
        imputed = 0

    denom = p95 - p5
    if denom <= 0:
        return 0.5, imputed
    norm = (float(val) - p5) / denom
    return max(0.0, min(1.0, norm)), imputed


# ── wearable data extraction ────────────────────────────────────────────────────

def extract_wearable_features(wearable_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract wearable vitals from wearable_data.json.

    The actual data format stores parallel arrays (daily_steps, resting_hr_bpm, …)
    plus a pre-computed computed_averages dict.  We prefer the pre-computed averages
    (which already exclude anomalous days like hospital admissions), then fall back
    to computing our own mean from the raw arrays.
    """
    out: dict[str, float | None] = {
        "daily_steps_7d_avg":  None,
        "resting_hr_bpm":      None,
        "sleep_hours_7d_avg":  None,
        "hrv_ms":              None,
        "ecog_ps_score":       None,
    }

    try:
        avgs = wearable_data.get("computed_averages", {}) or {}

        # Use pre-computed averages first — they already handle exclusions
        out["daily_steps_7d_avg"] = _safe_float(
            avgs.get("daily_steps_6d_avg") or avgs.get("daily_steps_7d_avg"))
        out["resting_hr_bpm"] = _safe_float(
            avgs.get("resting_hr_bpm_6d_avg") or avgs.get("resting_hr_bpm_7d_avg"))
        out["sleep_hours_7d_avg"] = _safe_float(
            avgs.get("sleep_hours_6d_avg") or avgs.get("sleep_hours_7d_avg"))
        out["hrv_ms"] = _safe_float(
            avgs.get("hrv_ms_6d_avg") or avgs.get("hrv_ms_7d_avg"))

        # Fall back to raw array mean if pre-computed not available
        if out["daily_steps_7d_avg"] is None:
            arr = wearable_data.get("daily_steps") or []
            if arr:
                out["daily_steps_7d_avg"] = sum(arr) / len(arr)
        if out["resting_hr_bpm"] is None:
            arr = wearable_data.get("resting_hr_bpm") or []
            if arr:
                out["resting_hr_bpm"] = sum(arr) / len(arr)
        if out["sleep_hours_7d_avg"] is None:
            arr = wearable_data.get("sleep_hours") or []
            if arr:
                out["sleep_hours_7d_avg"] = sum(arr) / len(arr)
        if out["hrv_ms"] is None:
            arr = wearable_data.get("hrv_ms") or []
            if arr:
                out["hrv_ms"] = sum(arr) / len(arr)

        # ECOG: prefer wearable-estimated ECOG when mismatch detected (more objective)
        flags = wearable_data.get("clinical_flags", {}) or {}
        ecog_mismatch   = flags.get("ecog_mismatch_flag", False)
        wearable_ecog   = flags.get("ecog_wearable_estimate")
        clinician_ecog  = flags.get("ecog_clinician_reported")

        if ecog_mismatch and wearable_ecog is not None:
            out["ecog_ps_score"] = float(wearable_ecog)
        elif clinician_ecog is not None:
            out["ecog_ps_score"] = float(clinician_ecog)

    except Exception as exc:
        log.warning("patient_state: wearable extraction error: %s", exc)

    return out


# ── intake_form extraction ──────────────────────────────────────────────────────

def extract_intake_features(intake_form: dict[str, Any]) -> dict[str, Any]:
    """Extract raw (un-normalised) feature values from patient_intake_form.json.

    Handles the real flat structure used in phase4_patient_data/:
      - Top-level keys: ecog_ps, cancer_type, prior_lines (list), ...
      - phase4_treatment_history: nested dict with treatment cycle data
      - phase4_renal_hepatic: nested dict with eGFR / creatinine
    """
    out: dict[str, Any] = {}
    try:
        # Cancer type (flat top-level key)
        out["cancer_type"] = intake_form.get("cancer_type", "unknown")

        # ECOG / KPS (flat top-level)
        out["ecog_ps_score"] = _safe_float(
            intake_form.get("ecog_ps") or intake_form.get("ecog"))

        # Treatment history lives in phase4_treatment_history nested dict
        tx = intake_form.get("phase4_treatment_history", {}) or {}
        out["treatment_cycles_completed"] = _safe_float(
            tx.get("treatment_cycles_completed"))
        out["treatment_duration_weeks"]   = _safe_float(
            tx.get("treatment_duration_weeks"))
        out["days_since_last_dose"]       = _safe_float(
            tx.get("days_since_last_dose"))
        out["dose_reduction_flag"]        = int(bool(
            tx.get("dose_reduction_flag", False)))

        # prior_lines: flat top-level — may be int OR a list of regimen strings
        prior_raw = intake_form.get("prior_lines", 0)
        if isinstance(prior_raw, list):
            out["total_prior_lines"] = float(len(prior_raw))
        elif isinstance(prior_raw, (int, float)):
            out["total_prior_lines"] = float(prior_raw)
        else:
            # current_line_of_therapy is 1-based; prior = current - 1
            cl = _safe_float(intake_form.get("current_line_of_therapy"))
            out["total_prior_lines"] = float(max(0, (cl or 1) - 1))

        # eGFR from phase4_renal_hepatic (real data uses this path)
        rh = intake_form.get("phase4_renal_hepatic", {}) or {}
        egfr = _safe_float(rh.get("egfr_override"))
        if egfr is None:
            # Estimate from creatinine using CKD-EPI approximation if available
            cr_umol = _safe_float(rh.get("latest_creatinine_umol_l"))
            age     = _safe_float(intake_form.get("age"))
            sex     = intake_form.get("sex", "M")
            if cr_umol and age:
                egfr = _estimate_egfr_ckdepi(cr_umol, age, sex)
        out["egfr_ml_per_min"] = egfr

    except Exception as exc:
        log.warning("patient_state: intake_form extraction error: %s", exc)

    return out


def extract_pgx_profile(intake_form: dict[str, Any]):
    """Extract ``PatientPharmacogenomics`` from intake JSON (Phase 5.2 / Module 5).

    Accepts several common shapes so legacy intakes keep working:
      - ``intake_form["pharmacogenomics"] = {"cyp2d6": "poor_metabolizer", ...}``
      - ``intake_form["phase4_pharmacogenomics"] = {...}``
      - ``intake_form["pgx"] = {...}``
      - absent entirely → defaults to ``normal_metabolizer`` with
        ``pgx_unavailable=True``.

    Missing fields → ``normal_metabolizer``. Missing block entirely →
    profile with ``pgx_unavailable=True`` so apply_pgx() can log graceful
    fallback.
    """
    from .schemas import PatientPharmacogenomics  # local import: avoid cycle

    block = (
        intake_form.get("pharmacogenomics")
        or intake_form.get("phase4_pharmacogenomics")
        or intake_form.get("pgx")
        or {}
    )
    if not isinstance(block, dict):
        block = {}

    # Canonicalise keys to lowercase with underscores.
    norm: dict[str, str] = {}
    for k, v in block.items():
        if not isinstance(v, str) or not v.strip():
            continue
        key = k.strip().lower().replace("-", "_").replace("*", "_")
        norm[key] = v.strip().lower().replace(" ", "_")

    fields = {
        "cyp2d6":  norm.get("cyp2d6"),
        "cyp2c19": norm.get("cyp2c19"),
        "cyp2c9":  norm.get("cyp2c9"),
        "cyp3a4":  norm.get("cyp3a4"),
        "cyp2b6":  norm.get("cyp2b6"),
        "cyp2c8":  norm.get("cyp2c8"),
        "tpmt":    norm.get("tpmt"),
        "dpyd":    norm.get("dpyd"),
        "ugt1a1":  norm.get("ugt1a1"),
        "mthfr":   norm.get("mthfr"),
        "gstp1":   norm.get("gstp1"),
        "cbr3":    norm.get("cbr3"),
        "hla_b":   norm.get("hla_b") or norm.get("hla_b_1502"),
    }
    supplied = {k: v for k, v in fields.items() if v is not None}
    defaults = {k: (v or "normal_metabolizer") for k, v in fields.items()}

    return PatientPharmacogenomics(
        **defaults,
        pgx_unavailable=(len(supplied) == 0),
        raw=block,
    )


def _estimate_egfr_ckdepi(creatinine_umol_l: float, age: float, sex: str) -> float:
    """Simplified CKD-EPI eGFR estimate from creatinine (umol/L).

    Returns eGFR in mL/min/1.73m². Approximate — sufficient for normalisation.
    """
    cr_mg_dl = creatinine_umol_l / 88.42
    is_female = str(sex).upper() in ("F", "FEMALE")
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    sex_mult = 1.012 if is_female else 1.0
    ratio = cr_mg_dl / kappa
    if ratio < 1:
        egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_mult
    else:
        egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_mult
    return max(5.0, min(150.0, egfr))


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── current medication extraction (clinical safety pre-seed) ───────────────────

def extract_current_drug_names(intake_form: dict[str, Any] | None) -> list[str]:
    """Extract current drug names from intake_form for RAG penalty pre-seeding.

    Uses structured JSON — no LLM needed.  Brand names are normalised to
    generics via _DRUG_ALIASES so the KB lookup succeeds.
    """
    drugs: list[str] = []
    if not intake_form:
        return drugs
    try:
        # Real path: phase4_treatment_history.current_regimen (free-text string)
        tx = intake_form.get("phase4_treatment_history", {}) or {}
        current = (
            tx.get("current_regimen")
            or intake_form.get("current_regimen")
            or intake_form.get("current_medications")
            or []
        )
        if isinstance(current, str):
            # Sometimes it's a free-text string like "Bevacizumab + Lomustine"
            current = [s.strip() for s in current.replace("+", ",").split(",")]
        for item in current:
            name = ""
            if isinstance(item, dict):
                name = item.get("name") or item.get("drug") or ""
            elif isinstance(item, str):
                name = item
            name = name.strip().lower()
            if name:
                name = _DRUG_ALIASES.get(name, name)
                drugs.append(name)
    except Exception as exc:
        log.warning("patient_state: drug name extraction error: %s", exc)
    return drugs


# ── molecular biomarker extraction (MGMT / IDH) — Task 3 ──────────────────────

_MGMT_RE = re.compile(
    r"MGMT[^.\n]{0,40}?(methylated|unmethylated|methylation\s+(?:positive|negative))",
    re.IGNORECASE,
)
_IDH_RE  = re.compile(
    r"IDH[\s-]?[12]?[^.\n]{0,40}?(mutant|mutation\s+positive|wild[\s-]?type|wildtype|R132[A-Z]?|negative)",
    re.IGNORECASE,
)


def _classify_mgmt(text: str) -> str | None:
    m = _MGMT_RE.search(text or "")
    if not m:
        return None
    val = m.group(1).lower()
    if "unmethylated" in val or "negative" in val:
        return "unmethylated"
    if "methylated" in val or "positive" in val:
        return "methylated"
    return None


def _classify_idh(text: str) -> str | None:
    m = _IDH_RE.search(text or "")
    if not m:
        return None
    val = m.group(1).lower()
    if "wild" in val or "negative" in val:
        return "wildtype"
    if "mutant" in val or "positive" in val or val.startswith("r132"):
        return "mutant"
    return None


def extract_molecular_biomarkers(
    intake_form: dict[str, Any] | None,
    memory: WorkingMemory,
) -> tuple[str | None, str | None]:
    """Return (mgmt_status, idh_status) — each "methylated"/"unmethylated"/"unknown"/None.

    None means the marker was never mentioned (treated the same as 'unknown'
    for the hard-stop decision, but kept distinct so we can tell whether the
    pathology report was even available).

    Priority:
      1. Structured intake_form.molecular_markers (or .biomarkers) field.
      2. Pathology / discharge / lab text in memory.
    """
    mgmt: str | None = None
    idh:  str | None = None

    # Source 1: structured intake JSON.
    #
    # MULTI-PATIENT-FIX (P001-run analysis): intakes for P001..P020 store
    # ``mgmt_status`` and ``idh_status`` as TOP-LEVEL keys (not nested
    # inside a ``biomarkers`` or ``molecular_markers`` block). The original
    # extractor only looked at the nested block, so it returned (None, None)
    # for every patient → biomarker hard-stop fired on every single run
    # → DEV_MODE bypassed it → SMBO ran without the most clinically
    # critical inputs. Now we check the top-level keys first and fall
    # through to the legacy nested lookup for back-compat.
    # ``not_applicable`` / ``not applicable`` is a real clinical value used
    # for non-glioma tumours where MGMT/IDH testing isn't indicated
    # (meningioma, ependymoma, brain mets). We treat it the same as
    # ``unknown`` so ``_evaluate_biomarker_hard_stop`` (which only fires
    # for GBM/astrocytoma) doesn't penalise these patients.
    _NOT_PRESENT = {
        "unknown", "pending", "not tested", "na", "n/a",
        "not applicable", "not_applicable", "not-applicable", "n.a.",
    }

    def _classify_mgmt(raw: str) -> str | None:
        raw = raw.strip().lower()
        if not raw:
            return None
        if raw in _NOT_PRESENT:
            return "unknown"
        if "unmethyl" in raw or raw in ("negative", "neg"):
            return "unmethylated"
        if "methyl" in raw or raw in ("positive", "pos"):
            return "methylated"
        return None

    def _classify_idh(raw: str) -> str | None:
        raw = raw.strip().lower()
        if not raw:
            return None
        if raw in _NOT_PRESENT:
            return "unknown"
        if "wild" in raw or raw in ("negative", "neg"):
            return "wildtype"
        if (
            "mutant" in raw or "mutation" in raw
            or raw in ("positive", "pos") or raw.startswith("r132")
            or raw.startswith("idh1") or raw.startswith("idh2")
        ):
            return "mutant"
        return None

    if intake_form:
        # 1a. Top-level intake keys (the v2 schema for P001..P020).
        for key in ("mgmt_status", "mgmt_methylation", "mgmt"):
            v = intake_form.get(key)
            if v:
                m = _classify_mgmt(str(v))
                if m:
                    mgmt = m
                    break
        for key in ("idh_status", "idh_mutation", "idh1", "idh"):
            v = intake_form.get(key)
            if v:
                m = _classify_idh(str(v))
                if m:
                    idh = m
                    break

        # 1b. Legacy nested-block lookup (back-compat).
        mol = (
            intake_form.get("molecular_markers")
            or intake_form.get("biomarkers")
            or intake_form.get("phase4_molecular")
            or {}
        )
        if isinstance(mol, dict):
            if mgmt is None:
                for key in ("mgmt", "mgmt_methylation", "mgmt_status"):
                    if mol.get(key):
                        mgmt = _classify_mgmt(str(mol[key])) or mgmt
                        if mgmt:
                            break
            if idh is None:
                for key in ("idh", "idh_mutation", "idh1", "idh_status"):
                    if mol.get(key):
                        idh = _classify_idh(str(mol[key])) or idh
                        if idh:
                            break

    # Source 2: free-text path/discharge in memory (regex on stored ingestion docs)
    if mgmt is None or idh is None:
        try:
            ing = memory.get(WorkingMemory.INGESTION) or memory.get("ingestion")
            files = (ing or {}).get("files", []) if isinstance(ing, dict) else []
            text_blob = ""
            for f in files:
                kind = (f.get("kind") or "").lower()
                if kind in ("pathology", "discharge", "lab", "mri_report"):
                    text_blob += "\n" + (f.get("text") or "")
            if text_blob:
                if mgmt is None:
                    mgmt = _classify_mgmt(text_blob)
                if idh is None:
                    idh = _classify_idh(text_blob)
        except Exception as exc:
            log.warning("patient_state: biomarker text-scan error: %s", exc)

    return mgmt, idh


def _evaluate_biomarker_hard_stop(
    cancer_type: str,
    mgmt: str | None,
    idh: str | None,
) -> tuple[bool, str | None]:
    """Decide whether Phase 4 should HARD STOP because critical biomarkers are
    missing for this cancer type.  Returns (hard_stop, reason)."""
    ct = (cancer_type or "").strip().lower()
    requires = any(req in ct for req in BIOMARKER_REQUIRED_CANCERS)
    if not requires:
        return False, None

    missing: list[str] = []
    if mgmt in (None, "unknown"):
        missing.append("MGMT promoter methylation")
    if idh in (None, "unknown"):
        missing.append("IDH mutation")

    if not missing:
        return False, None

    reason = (
        f"missing_critical_biomarkers: "
        f"{', '.join(missing)} unknown for {cancer_type}. "
        "Order molecular testing (IHC + MGMT methylation PCR) before "
        "treatment optimisation can be performed safely."
    )
    return True, reason


# ── main build function ─────────────────────────────────────────────────────────

def build_patient_state_vector(
    memory: WorkingMemory,
    intake_form: dict[str, Any] | None,
    wearable_data: dict[str, Any] | None,
) -> PatientStateVector:
    """Assemble the full 20-dimensional PatientState vector.

    Priority order per feature:
      1. Structured intake_form.json (most reliable for labs + treatment hist)
      2. wearable_data.json (for vitals, ECOG mismatch detection)
      3. Pipeline memory (RECIST for tumour burden, record for demographics)
      4. Population median imputation (last resort — sets imputation_mask bit)
    """
    # Collect raw values from all sources
    raw: dict[str, Any] = {}

    # ── Source 1: intake_form ──────────────────────────────────────────────────
    intake_vals = extract_intake_features(intake_form or {})
    raw.update({k: v for k, v in intake_vals.items() if v is not None})

    cancer_type = raw.pop("cancer_type", "unknown")

    # ── Source 2: wearable ─────────────────────────────────────────────────────
    wearable_vals = extract_wearable_features(wearable_data or {})
    for k, v in wearable_vals.items():
        if v is not None and k not in raw:
            raw[k] = v
        elif v is not None and k == "ecog_ps_score":
            # Always prefer wearable ECOG when available (more objective)
            raw[k] = v

    # ── Source 3: pipeline memory ──────────────────────────────────────────────
    try:
        from .schemas import RECISTAssessment
        recist_raw = memory.get(WorkingMemory.RECIST)
        if recist_raw:
            recist = (recist_raw if isinstance(recist_raw, RECISTAssessment)
                      else RECISTAssessment.model_validate(recist_raw))
            raw.setdefault("sum_of_diameters_mm", recist.current_sum_mm or 0.0)
            raw.setdefault("delta_sod_pct",
                           (recist.pct_change or 0.0) * 100.0)
            raw.setdefault("lesion_count",
                           len(recist.lesions_current))
            raw.setdefault("new_lesion_flag", int(recist.new_lesion_detected))
    except Exception as exc:
        log.warning("patient_state: RECIST extraction error: %s", exc)

    # ── Source 3a: longitudinal trajectory (Phase 5.3) — first-visit→imputed ──
    longitudinal_count = 0
    try:
        from .longitudinal_history import compute_trajectory_features, load_history
        history = load_history(memory.out_dir, memory.patient_id)
        longitudinal_count = history.visit_count
        # current SoD: prefer raw["sum_of_diameters_mm"] just set above.
        current_sod = float(raw.get("sum_of_diameters_mm") or 0.0)
        traj = compute_trajectory_features(history, current_sod)
        # Only seed the 4 trajectory dims when we actually have prior visits;
        # on first run let normalisation impute to population median.
        if longitudinal_count > 0:
            for k, v in traj.items():
                raw.setdefault(k, v)
        else:
            # Always populate visit_count even on first run (it's deterministic).
            raw.setdefault("visit_count", 0.0)
    except Exception as exc:
        log.warning("patient_state: longitudinal extraction error: %s", exc)

    # ── Source 3b: radiomics (Phase 5.1) — optional; population-median imputed if absent ──
    try:
        rad_raw = memory.get(WorkingMemory.RADIOMICS)
        if rad_raw:
            rad = rad_raw if isinstance(rad_raw, dict) else dict(rad_raw)
            if not rad.get("radiomics_unavailable", True):
                for feat in RADIOMIC_FEATURE_NAMES:
                    val = rad.get(feat)
                    if val is not None:
                        raw.setdefault(feat, float(val))
    except Exception as exc:
        log.warning("patient_state: radiomics extraction error: %s", exc)

    # ── Try to get cancer_type from record if not from intake ─────────────────
    if cancer_type == "unknown":
        try:
            from .schemas import PatientRecord
            rec_raw = memory.get(WorkingMemory.RECORD)
            if rec_raw:
                rec = (rec_raw if isinstance(rec_raw, PatientRecord)
                       else PatientRecord.model_validate(rec_raw))
                diag = rec.diagnosis or ""
                if diag:
                    cancer_type = diag.lower()
        except Exception:
            pass

    # ── Normalise all 20 features ──────────────────────────────────────────────
    normalized: list[float] = []
    imputation_mask: list[int] = []

    for feat in FEATURE_NAMES:
        val = raw.get(feat)
        norm_val, imputed = _normalize_value(
            None if val is None else float(val),
            feat, cancer_type,
        )
        normalized.append(norm_val)
        imputation_mask.append(imputed)

    # ── Build the Pydantic model ───────────────────────────────────────────────
    def _r(key: str, default: Any = 0.0) -> Any:
        v = raw.get(key, default)
        return default if v is None else v

    # ── Pharmacogenomics (Phase 5.2 / Module 5) — optional ────────────────────
    try:
        pgx_profile = extract_pgx_profile(intake_form or {})
    except Exception as exc:
        log.warning("patient_state: pgx extraction error: %s", exc)
        pgx_profile = None

    # ── Molecular biomarkers (Task 3 — never imputed) ─────────────────────────
    mgmt_status, idh_status = extract_molecular_biomarkers(intake_form, memory)
    hard_stop, hard_stop_reason = _evaluate_biomarker_hard_stop(
        cancer_type, mgmt_status, idh_status,
    )
    # DEV_MODE: demote the hard-stop to a warning so the pipeline continues
    # into sub-steps 4b–4e (RSF/GP/RFR get trained, MDT debate runs, pkls
    # cache to disk). Missing markers stay "unknown" in the vector — we do
    # NOT impute them; we just stop refusing to proceed.
    from ..config import DEV_MODE as _DEV_MODE
    if hard_stop and _DEV_MODE:
        log.warning(
            "patient_state: BIOMARKER HARD-STOP bypassed by DEV_MODE for %s — %s "
            "(NEVER do this in production)",
            memory.patient_id, hard_stop_reason,
        )
        hard_stop = False
    elif hard_stop:
        log.warning(
            "patient_state: BIOMARKER HARD STOP for %s — %s",
            memory.patient_id, hard_stop_reason,
        )

    vec = PatientStateVector(
        patient_id=memory.patient_id,
        cancer_type=cancer_type,
        mgmt_methylation=mgmt_status or "unknown",
        idh_mutation=idh_status or "unknown",
        biomarker_hard_stop=hard_stop,
        biomarker_hard_stop_reason=hard_stop_reason,
        pgx_profile=pgx_profile,
        # Tumour burden
        sum_of_diameters_mm  = float(_r("sum_of_diameters_mm")),
        delta_sod_pct        = float(_r("delta_sod_pct")),
        lesion_count         = int(_r("lesion_count", 0)),
        new_lesion_flag      = int(_r("new_lesion_flag", 0)),
        # Lab markers
        ldh_u_per_l          = float(_r("ldh_u_per_l")),
        crp_mg_per_l         = float(_r("crp_mg_per_l")),
        nlr                  = float(_r("nlr")),
        hemoglobin_g_per_dl  = float(_r("hemoglobin_g_per_dl")),
        albumin_g_per_dl     = float(_r("albumin_g_per_dl")),
        egfr_ml_per_min      = float(_r("egfr_ml_per_min")),
        # Wearable vitals
        daily_steps_7d_avg   = float(_r("daily_steps_7d_avg")),
        resting_hr_bpm       = float(_r("resting_hr_bpm")),
        sleep_hours_7d_avg   = float(_r("sleep_hours_7d_avg")),
        hrv_ms               = float(_r("hrv_ms")),
        ecog_ps_score        = float(_r("ecog_ps_score", 1.0)),
        # Treatment history
        treatment_cycles_completed = float(_r("treatment_cycles_completed")),
        days_since_last_dose       = float(_r("days_since_last_dose")),
        total_prior_lines          = float(_r("total_prior_lines")),
        dose_reduction_flag        = int(_r("dose_reduction_flag", 0)),
        treatment_duration_weeks   = float(_r("treatment_duration_weeks")),
        # Radiomics (Phase 5.1)
        glcm_contrast              = float(_r("glcm_contrast")),
        glcm_correlation           = float(_r("glcm_correlation")),
        shape_sphericity           = float(_r("shape_sphericity")),
        shape_surface_volume_ratio = float(_r("shape_surface_volume_ratio")),
        firstorder_entropy         = float(_r("firstorder_entropy")),
        # Longitudinal trajectory (Phase 5.3)
        sod_growth_rate_mm_per_week = float(_r("sod_growth_rate_mm_per_week")),
        pfs_trajectory_slope        = float(_r("pfs_trajectory_slope")),
        treatment_response_streak   = int(_r("treatment_response_streak", 0)),
        visit_count                 = int(_r("visit_count", longitudinal_count)),
        # Derived
        normalized       = normalized,
        imputation_mask  = imputation_mask,
        raw_values       = {k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in raw.items() if v is not None},
    )

    n_imputed = sum(imputation_mask)
    log.info(
        "patient_state: built %d-dim vector for %s  cancer_type=%s  imputed=%d/%d",
        len(FEATURE_NAMES), memory.patient_id, cancer_type,
        n_imputed, len(FEATURE_NAMES),
    )
    return vec


# ── convenience loader for phase4 JSON files ───────────────────────────────────

def load_phase4_json(patient_id: str, filename: str) -> dict[str, Any] | None:
    """Load a Phase 4 JSON (intake_form / wearable_data) for a patient.

    Looks in both layouts, in preference order:
      v2 unified: Datasets/patients/{pid}/phase4/{filename}
      v1 legacy : Datasets/phase4_patient_data/{pid}/{filename}

    Returns None if the file cannot be found in either layout.
    """
    candidates = [
        PHASE4_DATA_ROOT / patient_id / "phase4" / filename,   # v2
        PHASE4_DATA_ROOT / patient_id / filename,              # v1
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("patient_state: could not load %s: %s", path, exc)
                return None
    return None
