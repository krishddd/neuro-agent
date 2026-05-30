"""Batched Sequential Model-Based Optimisation (SMBO v3.0) engine.

Architecture
------------
Surrogate model: GaussianProcessRegressor (Matérn ν=2.5 + WhiteKernel)
  - Maps compact 5-dim drug vector → PFS score
  - Starts with NCCN warm-start anchors (×100 replication for weight amplification)
  - Updated each iteration with evaluated candidates

Candidate generation: RandomForestRegressor (fast inner-loop proxy)
  - Pre-trained on the same drug-feature space from NCCN + random exploration
  - Generates 1 000 random candidates per batch → dual-sort (EI pool + PRED pool)

Dual-sort selection per batch:
  - EI pool (5): candidates with highest Expected Improvement over current best
  - PRED pool (5): candidates with highest raw GP posterior mean (exploitation)

Active Inference acquisition:
  - acquisition = (1-w) * EI + w * (-sigma)  [collapse uncertainty]
  - When best sigma < 0.20: switch to pure EI (exploitation mode)
  - Early stop: best sigma < SMBO_SIGMA_EARLY_STOP (0.12)

Drug encoding (5 dims, normalised to [0,1]):
  [primary_drug_idx, combo_drug_idx, dose_fraction, cycle_weeks, route_idx]

Full candidate feature = [patient_vec (20) | drug_vec (5)] = 25 dims
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from ..config import (
    DRUG_ENCODING_PATH,
    DRUG_TOXICITY_PROFILE_PATH,
    NCCN_GUIDELINES_PATH,
    SMBO_ANCHOR_WEIGHT,
    SMBO_BATCH_SIZE,
    SMBO_BUDGET,
    SMBO_EI_POOL,
    SMBO_LAMBDA,
    SMBO_PRED_POOL,
    SMBO_SIGMA_EARLY_STOP,
    SMBO_TOXICITY_WEIGHT,
    SMBO_WARM_START_N,
)
from ..utils.schemas import OptimizationResult, SMBOCandidate
from .rag_penalty import batch_rag_penalties, rag_penalty

log = logging.getLogger(__name__)

# ── Drug search-space constants ────────────────────────────────────────────────
_DRUG_CLASSES: list[str] = []          # loaded lazily
_DRUG_INDEX: dict[str, int] = {}       # name → index (0-based)
_NCCN_DATA: dict[str, Any] = {}

ROUTES = ["iv", "oral", "sc"]
ROUTE_INDEX = {r: i for i, r in enumerate(ROUTES)}
N_DRUG_DIMS = 5      # compact drug encoding dimensions
N_PATIENT_DIMS = 20  # patient state vector dimensions
N_TOTAL_DIMS = N_PATIENT_DIMS + N_DRUG_DIMS


def _load_drug_classes() -> list[str]:
    global _DRUG_CLASSES, _DRUG_INDEX
    if _DRUG_CLASSES:
        return _DRUG_CLASSES
    try:
        data = json.loads(DRUG_ENCODING_PATH.read_text(encoding="utf-8"))
        _DRUG_CLASSES = data.get("drug_classes", [])
        _DRUG_INDEX = {d.lower(): i for i, d in enumerate(_DRUG_CLASSES)}
        log.info("smbo_engine: loaded %d drug classes", len(_DRUG_CLASSES))
    except Exception as exc:
        log.warning("smbo_engine: could not load drug_encoding.json (%s) — using fallback", exc)
        _DRUG_CLASSES = [
            "temozolomide", "bevacizumab", "lomustine", "carmustine",
            "procarbazine", "vincristine", "cisplatin", "carboplatin",
            "etoposide", "methotrexate", "rituximab", "nivolumab",
            "pembrolizumab", "dexamethasone", "lapatinib", "erlotinib",
            "osimertinib", "everolimus", "temsirolimus", "palbociclib",
            "olaparib", "irinotecan", "paclitaxel", "docetaxel",
            "capecitabine", "fluorouracil", "imatinib", "sunitinib",
            "sorafenib", "regorafenib",
        ]
        _DRUG_INDEX = {d: i for i, d in enumerate(_DRUG_CLASSES)}
    return _DRUG_CLASSES


def _load_nccn() -> dict[str, Any]:
    global _NCCN_DATA
    if _NCCN_DATA:
        return _NCCN_DATA
    try:
        _NCCN_DATA = json.loads(NCCN_GUIDELINES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("smbo_engine: could not load NCCN guidelines (%s)", exc)
        _NCCN_DATA = {}
    return _NCCN_DATA


# ── Toxicity profiles (Task 6) ────────────────────────────────────────────────
_TOXICITY_DB: dict[str, Any] = {}


def _load_toxicity_profiles() -> dict[str, Any]:
    global _TOXICITY_DB
    if _TOXICITY_DB:
        return _TOXICITY_DB
    try:
        _TOXICITY_DB = json.loads(
            DRUG_TOXICITY_PROFILE_PATH.read_text(encoding="utf-8")
        )
        log.info("smbo_engine: loaded toxicity profiles for %d drugs",
                 sum(1 for k in _TOXICITY_DB if not k.startswith("_")))
    except Exception as exc:
        log.warning("smbo_engine: toxicity profiles unavailable (%s) — using fallback", exc)
        _TOXICITY_DB = {}
    return _TOXICITY_DB


def _toxicity_profile(drug_name: str) -> dict[str, Any]:
    """Return the CTCAE toxicity profile for a drug, case-insensitive.

    Falls back to the ``_fallback`` entry when the drug name is unknown.
    """
    db = _load_toxicity_profiles()
    key = drug_name.lower().strip()
    if key in db and not key.startswith("_"):
        return db[key]
    # Try partial match (e.g. "temozolomide + RT" → "temozolomide")
    for k, v in db.items():
        if k.startswith("_"):
            continue
        if key in k or k in key:
            return v
    return db.get("_fallback", {})


# Patient-factor modifiers for specific toxicity dimensions.
# When a patient state field exceeds the given threshold, multiply that
# CTCAE rate by the supplied weight factor.
_PATIENT_TOX_MODIFIERS: list[tuple[str, float, str, float]] = [
    # (patient_state_field, threshold_above_which_applies, ctcae_key, multiplier)
    ("albumin_g_per_dl",  3.5,  "thrombocytopenia_g34",  1.4),   # low albumin → elevated thrombo risk
    ("hemoglobin_g_per_dl", 10.0, "anemia_g34",          1.5),   # low Hb → worse anaemia tolerance
    ("egfr_ml_per_min",   60.0,  "nephrotoxicity_g34",   1.8),   # impaired renal → nephrotox
    ("crp_mg_per_l",      10.0,  "infection_g34",        1.3),   # elevated CRP → infection-prone
    ("nlr",                5.0,  "thromboembolism_g34",  1.4),   # high NLR → thrombosis risk
]

# CTCAE severity weights for composite index.  Grade 4 events dominate.
_CTCAE_SEVERITY_WEIGHTS: dict[str, float] = {
    "thrombocytopenia_g34": 1.4,
    "neutropenia_g34":      1.3,
    "anemia_g34":           1.1,
    "nausea_vomiting_g34":  0.9,
    "fatigue_g34":          0.8,
    "hepatotoxicity_g34":   1.2,
    "nephrotoxicity_g34":   1.3,
    "hypertension_g34":     1.1,
    "hemorrhage_g34":       1.5,
    "proteinuria_g34":      0.9,
    "neurotoxicity_g34":    1.3,
    "infection_g34":        1.2,
    "thromboembolism_g34":  1.4,
    "wound_healing_g34":    1.0,
}

# AE label strings shown in SMBOCandidate.top_aes.
_AE_LABELS: dict[str, str] = {
    "thrombocytopenia_g34": "Thrombocytopenia G3/4",
    "neutropenia_g34":      "Neutropenia G3/4",
    "anemia_g34":           "Anaemia G3/4",
    "nausea_vomiting_g34":  "Nausea/Vomiting G3/4",
    "fatigue_g34":          "Fatigue G3/4",
    "hepatotoxicity_g34":   "Hepatotoxicity G3/4",
    "nephrotoxicity_g34":   "Nephrotoxicity G3/4",
    "hypertension_g34":     "Hypertension G3/4",
    "hemorrhage_g34":       "Haemorrhage G3/4",
    "proteinuria_g34":      "Proteinuria G3/4",
    "neurotoxicity_g34":    "Neurotoxicity G3/4",
    "infection_g34":        "Opportunistic Infection G3/4",
    "thromboembolism_g34":  "VTE/Thromboembolism G3/4",
    "wound_healing_g34":    "Wound Healing Impairment G3/4",
}


def predicted_toxicity(
    primary_drug: str,
    combo_drug: str,
    dose_fraction: float,
    patient_vec: np.ndarray | None = None,
    pgx_profile=None,
) -> tuple[float, list[str]]:
    """Predict composite CTCAE toxicity index and list the top adverse events.

    Returns ``(severity_index, top_ae_strings)`` where:
    - ``severity_index`` is a weighted sum (0–1 scale).
    - ``top_ae_strings`` are human-readable AE descriptors, e.g.
      ``["Thrombocytopenia G3/4 (~25%)", "Neutropenia G3/4 (~23%)"]``.

    Formula:
        base = profile_primary + 0.7 × profile_combo  (additive, capped at 1.0)
        patient-adjusted each rate by _PATIENT_TOX_MODIFIERS
        severity_index = Σ(rate_i × severity_weight_i) / Σ(severity_weight_i)
        Final score scaled by dose_fraction (higher dose → more toxicity).
    """
    p_profile = _toxicity_profile(primary_drug)
    c_profile = _toxicity_profile(combo_drug) if (
        combo_drug and combo_drug.lower() not in ("none", "")
    ) else {}

    # Build per-AE combined rates.
    combined: dict[str, float] = {}
    for ae_key in _CTCAE_SEVERITY_WEIGHTS:
        p_rate = float(p_profile.get(ae_key, 0.0))
        c_rate = float(c_profile.get(ae_key, 0.0)) * 0.7
        combined[ae_key] = min(1.0, p_rate + c_rate)

    # Patient-specific modifiers (impaired organ function escalates risk).
    if patient_vec is not None:
        pv = np.asarray(patient_vec, dtype=np.float64).flatten()
        # Map field name → index in the normalised patient state vector.
        # Indices mirror PatientStateVector field order (dims 5-10 = lab markers).
        _field_to_idx: dict[str, int] = {
            "ldh_u_per_l":       5,
            "crp_mg_per_l":      6,
            "nlr":               7,
            "hemoglobin_g_per_dl": 8,
            "albumin_g_per_dl":  9,
            "egfr_ml_per_min":  10,
        }
        for field, threshold, ae_key, multiplier in _PATIENT_TOX_MODIFIERS:
            idx = _field_to_idx.get(field)
            if idx is None or idx >= len(pv):
                continue
            # The patient vector is normalised [0,1]. We need to compare against
            # the raw threshold, so we use raw_values if available.
            # As a pragmatic fallback: for fields where a LOW value means risk
            # (albumin, Hb, eGFR) we invert the comparison (pv[idx] < norm_threshold).
            # For HIGH-value risk fields (CRP, NLR, LDH) we use pv[idx] > 0.5.
            is_low_risk = field in ("albumin_g_per_dl", "hemoglobin_g_per_dl", "egfr_ml_per_min")
            if is_low_risk:
                if pv[idx] < 0.5:    # normalised < 0.5 ≈ below normal
                    combined[ae_key] = min(1.0, combined[ae_key] * multiplier)
            else:
                if pv[idx] > 0.5:    # normalised > 0.5 ≈ elevated
                    combined[ae_key] = min(1.0, combined[ae_key] * multiplier)

    # Apply dose modifier: toxicity scales (not linearly) with dose fraction.
    # Saturates at full dose; floor at half dose.
    dose_mod = 0.6 + 0.4 * float(dose_fraction)  # [0.5,1.0] → [0.80, 1.0]
    combined = {k: v * dose_mod for k, v in combined.items()}

    # Compute weighted severity index.
    total_weight = sum(_CTCAE_SEVERITY_WEIGHTS.values())
    severity_index = sum(
        combined[ae] * _CTCAE_SEVERITY_WEIGHTS[ae]
        for ae in _CTCAE_SEVERITY_WEIGHTS
    ) / total_weight

    # Phase 5.2 / Module 5 — Pharmacogenomic adjustment.
    # Multiply severity_index by the worst-case tox multiplier across
    # primary + combo drugs (takes the maximum because tox for the
    # more-affected drug dominates the composite). Capped at 1.0.
    if pgx_profile is not None:
        try:
            from .pgx_adjuster import apply_pgx
            _p_eff, p_tox, _ = apply_pgx(primary_drug, pgx_profile)
            c_tox = 1.0
            if combo_drug and combo_drug.lower() not in ("none", ""):
                _, c_tox, _ = apply_pgx(combo_drug, pgx_profile)
            tox_mult = max(p_tox, c_tox)
            severity_index = min(1.0, severity_index * tox_mult)
        except Exception:
            pass

    # Build top-AE list (threshold ≥5% G3/4 OR rate above the base profile's index).
    top_aes: list[str] = []
    for ae_key, rate in sorted(combined.items(), key=lambda x: -x[1]):
        if rate >= 0.05 and ae_key in _AE_LABELS:
            top_aes.append(f"{_AE_LABELS[ae_key]} (~{rate*100:.0f}%)")
        if len(top_aes) >= 5:
            break

    return float(severity_index), top_aes


def _drug_idx(name: str) -> int:
    """Return index of drug name (0-based); -1 for 'none'."""
    if not name or name.lower() == "none":
        return -1
    _load_drug_classes()
    return _DRUG_INDEX.get(name.lower().strip(), 0)


def _encode_candidate(
    cand: dict,
    patient_vec: np.ndarray,
) -> np.ndarray:
    """Encode a candidate dict + patient state into a 25-dim feature vector.

    drug_vec dims:
      0: primary_drug index / (n_drugs - 1)         [0, 1]
      1: combo_drug index / n_drugs (n_drugs = none) [0, 1]
      2: dose_fraction                               [0.5, 1.0] → [0, 1]
      3: cycle_weeks / 4                             [1,4] → [0.25, 1]
      4: route index / 2                             [0, 1]
    """
    _load_drug_classes()
    n_drugs = len(_DRUG_CLASSES)

    pd_idx = _drug_idx(cand.get("primary_drug", ""))
    cd_idx = _drug_idx(cand.get("combo_drug", "none"))

    drug_vec = np.array([
        pd_idx / max(n_drugs - 1, 1),
        (cd_idx + 1) / (n_drugs + 1),    # +1 offset: "none" → 0, drugs → 1..n
        (float(cand.get("dose_fraction", 0.75)) - 0.5) / 0.5,  # [0.5,1.0]→[0,1]
        float(cand.get("cycle_weeks", 4)) / 4.0,
        ROUTE_INDEX.get(cand.get("route", "iv"), 0) / 2.0,
    ], dtype=np.float64)

    pv = np.asarray(patient_vec, dtype=np.float64).flatten()[:N_PATIENT_DIMS]
    return np.concatenate([pv, drug_vec])


# ── Candidate scoring ──────────────────────────────────────────────────────────

def _pfs_from_patient_baseline(patient_vec: np.ndarray) -> float:
    """Quick patient-level PFS estimate from RSF (no drug info yet)."""
    try:
        from .survival_models import predict_rsf_pfs
        pfs_med, _, _, _ = predict_rsf_pfs(
            np.asarray(patient_vec, dtype=np.float64).reshape(1, -1)
        )
        return float(pfs_med)
    except Exception:
        return 24.0  # fallback default


def _drug_benefit_factor(primary_drug: str, combo_drug: str,
                          dose_fraction: float, cancer_type: str) -> float:
    """Compute a multiplicative PFS benefit factor for a drug regimen.

    Uses NCCN guidelines as a reference: SOC regimens get a moderate benefit
    factor; experimental combinations get less; high-dose boosts benefit.
    Baseline factor = 1.0 (no change from patient baseline).
    """
    nccn = _load_nccn()
    ct = cancer_type.lower().replace(" ", "_").replace("-", "_")
    ct_data = nccn.get(ct, {})

    # Check if primary drug appears in first-line or second-line regimens
    benefit = 1.0
    soc_text = json.dumps(ct_data.get("standard_first_line", {})).lower()
    soc2_text = json.dumps(ct_data.get("standard_second_line", {})).lower()

    pd_lower = primary_drug.lower()
    in_first_line = pd_lower in soc_text
    in_second_line = pd_lower in soc2_text

    if in_first_line:
        benefit = 1.25   # SOC first-line → 25% PFS benefit over baseline
    elif in_second_line:
        benefit = 1.10   # SOC second-line → 10% benefit
    else:
        benefit = 0.95   # Off-label / exploratory → slight discount

    # Combo bonus
    if combo_drug and combo_drug.lower() != "none":
        cd_lower = combo_drug.lower()
        if cd_lower in soc_text or cd_lower in soc2_text:
            benefit *= 1.10  # known NCCN combo
        else:
            benefit *= 1.03  # unknown combo — marginal assumed benefit

    # Dose modifier
    benefit *= (0.7 + 0.3 * dose_fraction)   # dose_fraction ∈ [0.5, 1.0]

    return float(benefit)


# Generic-name → common abbreviations / brand variants used in NCCN summaries.
# String-contains check uses the union of forms so e.g. "TMZ" in the regimen
# text counts as a match for "temozolomide".
_NCCN_DRUG_ALIASES: dict[str, list[str]] = {
    "temozolomide":  ["temozolomide", "tmz", "temodar"],
    "bevacizumab":   ["bevacizumab", "bev", "avastin"],
    "lomustine":     ["lomustine", "ccnu", "gleostine"],
    "carmustine":    ["carmustine", "bcnu", "gliadel"],
    "procarbazine":  ["procarbazine", "pcv"],
    "regorafenib":   ["regorafenib", "stivarga"],
    "nivolumab":     ["nivolumab", "opdivo"],
    "pembrolizumab": ["pembrolizumab", "keytruda"],
    "rituximab":     ["rituximab", "rituxan"],
    "irinotecan":    ["irinotecan", "cpt-11", "camptosar"],
    "5-fluorouracil":["5-fluorouracil", "5fu", "5-fu", "fluorouracil"],
    "methotrexate":  ["methotrexate", "mtx"],
    "vincristine":   ["vincristine", "oncovin"],
    "doxorubicin":   ["doxorubicin", "adriamycin"],
    "cisplatin":     ["cisplatin", "cddp"],
    "carboplatin":   ["carboplatin", "cbdca"],
    "cyclophosphamide": ["cyclophosphamide", "cyclophos", "cytoxan"],
}


def _drug_in_text(drug: str, text: str) -> bool:
    """True if any alias for ``drug`` appears in lower-cased ``text``."""
    if not drug or not text:
        return False
    d = drug.lower().strip()
    forms = _NCCN_DRUG_ALIASES.get(d, [d])
    return any(form in text for form in forms)


def classify_nccn_alignment(
    primary_drug: str, combo_drug: str, cancer_type: str,
) -> tuple[bool, bool]:
    """Phase 5.7 / Extra B — return (off_label, novel_combo).

    ``off_label``: primary_drug not in NCCN first-line OR second-line
    regimens for the cancer type.
    ``novel_combo``: combo_drug present (not 'none') AND not found in
    NCCN regimens for the cancer type.

    Alias-aware: e.g. "temozolomide" matches NCCN regimens that say
    "TMZ" or "Temodar". Without the alias map roughly half of the SOC
    regimens would be mis-flagged as off-label.

    These flags gate the FDA FAERS adverse-event lookup so we only
    spend API budget on regimens outside the standard-of-care.
    """
    nccn = _load_nccn()
    ct = (cancer_type or "").lower().replace(" ", "_").replace("-", "_")
    ct_data = nccn.get(ct, {})
    soc1 = json.dumps(ct_data.get("standard_first_line", {})).lower()
    soc2 = json.dumps(ct_data.get("standard_second_line", {})).lower()
    soc_combined = soc1 + " " + soc2

    pd_l = (primary_drug or "").lower()
    cd_l = (combo_drug or "").lower()
    off_label = bool(pd_l) and not _drug_in_text(pd_l, soc_combined)
    novel_combo = (
        bool(cd_l) and cd_l not in ("none", "")
        and not _drug_in_text(cd_l, soc_combined)
    )
    return (off_label, novel_combo)


def score_candidate(
    candidate: dict,
    patient_vec: np.ndarray,
    pfs_baseline: float,
    rag_pen: float,
    lambda_penalty: float = SMBO_LAMBDA,
    cancer_type: str = "glioblastoma",
    tox_weight: float = SMBO_TOXICITY_WEIGHT,
    pgx_profile=None,
) -> float:
    """Multi-objective score for a single drug candidate (Task 6).

    Score = pfs_benefit
            - lambda_recist × max(0, recist_delta_correction)
            - lambda_tox    × tox_severity_index × pfs_baseline
            - rag_penalty

    ``tox_severity_index`` is a weighted composite of CTCAE G3/4 rates
    (from ``drug_toxicity_profiles.json``) modulated by patient-state fields
    (eGFR, Hb, NLR, albumin). Multiplied by ``pfs_baseline`` to express the
    toxicity penalty in the same units (weeks) as the PFS benefit term.
    """
    pd = candidate.get("primary_drug", "")
    cd = candidate.get("combo_drug", "none")
    dose = float(candidate.get("dose_fraction", 0.75))

    # PFS benefit
    benefit_factor = _drug_benefit_factor(pd, cd, dose, cancer_type)

    # Phase 5.2 / Module 5 — Pharmacogenomic efficacy multiplier on PFS benefit.
    # Uses the primary_drug's eff; combo's eff discounts at 0.3 (matches the
    # 0.7 weight combo carries in predicted_toxicity so the two objectives
    # stay comparable).
    if pgx_profile is not None:
        try:
            from .pgx_adjuster import apply_pgx
            p_eff, _, _ = apply_pgx(pd, pgx_profile)
            c_eff = 1.0
            if cd and cd.lower() not in ("none", ""):
                c_eff, _, _ = apply_pgx(cd, pgx_profile)
            benefit_factor *= p_eff * (1.0 + 0.3 * (c_eff - 1.0))
        except Exception:
            pass

    pfs_benefit = pfs_baseline * benefit_factor

    # RECIST delta proxy: lower dose → higher predicted residual tumour
    recist_correction = max(0.0, (1.0 - dose) * 30.0)

    # Task 6: toxicity objective — penalise regimens with high CTCAE severity.
    tox_index, _top_aes = predicted_toxicity(pd, cd, dose, patient_vec, pgx_profile=pgx_profile)
    # Scale to PFS units: a tox_index of 1.0 subtracts tox_weight × pfs_baseline weeks.
    tox_penalty = tox_weight * tox_index * pfs_baseline

    score = pfs_benefit - lambda_penalty * recist_correction - tox_penalty - rag_pen
    return float(score)


# ── GP surrogate for SMBO ──────────────────────────────────────────────────────

def _build_smbo_gp():
    """Build a fresh small GP for SMBO drug-space modelling."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    # noise_level_bounds raised from (1e-5, 1e5) to (1e-3, 1e3) to prevent
    # the ConvergenceWarning that fires when the optimizer hits the lower bound.
    kernel = (
        Matern(nu=2.5, length_scale_bounds=(0.01, 10.0))
        + WhiteKernel(noise_level_bounds=(1e-3, 1e3))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2,
        normalize_y=True,
        random_state=42,
    )


def _ei(mu: np.ndarray, sigma: np.ndarray, best_y: float,
        xi: float = 0.01) -> np.ndarray:
    """Expected Improvement over best_y."""
    from scipy.stats import norm  # type: ignore
    z = (mu - best_y - xi) / np.maximum(sigma, 1e-8)
    return (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def active_inference_acquisition(
    ei_scores: np.ndarray,
    gp_sigmas: np.ndarray,
    sigma_weight: float = 0.4,
    best_sigma: float = 1.0,
) -> np.ndarray:
    """Combined acquisition = (1-w)*EI + w*(-sigma).

    When best_sigma < 0.20 → switch to pure EI (exploitation mode).
    """
    if best_sigma < 0.20:
        # Pure exploitation — no longer need uncertainty reduction
        return ei_scores
    # Normalise both components to [0,1] before combining
    ei_norm = (ei_scores - ei_scores.min()) / max(ei_scores.max() - ei_scores.min(), 1e-8)
    sig_norm = (gp_sigmas - gp_sigmas.min()) / max(gp_sigmas.max() - gp_sigmas.min(), 1e-8)
    return (1.0 - sigma_weight) * ei_norm + sigma_weight * (1.0 - sig_norm)


# ── Warm-start from NCCN SOC anchors ──────────────────────────────────────────

def warm_start_from_nccn(
    cancer_type: str,
    patient_vec: np.ndarray,
    pfs_baseline: float,
    pgx_profile=None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Pull SMBO_WARM_START_N SOC regimens from NCCN for this cancer type.

    Returns (X_anchor, y_anchor, anchor_dicts) after ×SMBO_ANCHOR_WEIGHT
    replication to amplify their statistical weight in the GP.
    """
    nccn = _load_nccn()
    ct = cancer_type.lower().replace(" ", "_").replace("-", "_")
    ct_data = nccn.get(ct, {})

    # Collect regimen strings from first-line and second-line entries
    anchors: list[dict] = []

    def _parse_regimen_text(regimen_text: str) -> dict | None:
        """Extract primary drug from a regimen text string."""
        rl = regimen_text.lower()
        _load_drug_classes()
        # Find first matching drug in the regimen text
        for drug in sorted(_DRUG_CLASSES, key=len, reverse=True):  # longest first
            if drug in rl:
                # Find a sensible combo
                for combo in sorted(_DRUG_CLASSES, key=len, reverse=True):
                    if combo != drug and combo in rl:
                        return {
                            "primary_drug": drug,
                            "combo_drug": combo,
                            "dose_fraction": 1.0,
                            "cycle_weeks": 4,
                            "route": "iv",
                        }
                return {
                    "primary_drug": drug,
                    "combo_drug": "none",
                    "dose_fraction": 1.0,
                    "cycle_weeks": 4,
                    "route": "iv" if drug not in ("temozolomide", "capecitabine", "lomustine") else "oral",
                }
        return None

    for section_key in ("standard_first_line", "standard_second_line"):
        section = ct_data.get(section_key, {})
        if isinstance(section, dict):
            for sub_key, sub_val in section.items():
                if isinstance(sub_val, dict):
                    reg_text = sub_val.get("regimen", "")
                elif isinstance(sub_val, list):
                    for item in sub_val:
                        if isinstance(item, dict):
                            reg_text = item.get("regimen", "")
                            cand = _parse_regimen_text(reg_text)
                            if cand and len(anchors) < SMBO_WARM_START_N:
                                anchors.append(cand)
                    continue
                else:
                    reg_text = str(sub_val)
                cand = _parse_regimen_text(reg_text)
                if cand and len(anchors) < SMBO_WARM_START_N:
                    anchors.append(cand)
        elif isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    reg_text = item.get("regimen", "")
                    cand = _parse_regimen_text(reg_text)
                    if cand and len(anchors) < SMBO_WARM_START_N:
                        anchors.append(cand)
        if len(anchors) >= SMBO_WARM_START_N:
            break

    # Fallback if NCCN parse yielded nothing
    if not anchors:
        log.warning("smbo_engine: no NCCN anchors found for '%s' — using hardcoded TMZ fallback", ct)
        anchors = [
            {"primary_drug": "temozolomide", "combo_drug": "none", "dose_fraction": 1.0,
             "cycle_weeks": 4, "route": "oral"},
            {"primary_drug": "bevacizumab", "combo_drug": "temozolomide", "dose_fraction": 1.0,
             "cycle_weeks": 2, "route": "iv"},
            {"primary_drug": "lomustine", "combo_drug": "none", "dose_fraction": 1.0,
             "cycle_weeks": 6, "route": "oral"},
        ]

    # Deduplicate
    seen: set[tuple] = set()
    unique_anchors: list[dict] = []
    for a in anchors:
        k = (a["primary_drug"], a["combo_drug"])
        if k not in seen:
            seen.add(k)
            unique_anchors.append(a)
    anchors = unique_anchors[:SMBO_WARM_START_N]

    # Check contraindications — skip anchor if rag_penalty == 999
    safe_anchors = []
    for a in anchors:
        pen = rag_penalty(a["primary_drug"], a["combo_drug"])
        if pen < 900.0:
            safe_anchors.append(a)
        else:
            log.info("smbo_engine: warm-start anchor %s+%s contraindicated — skipped",
                     a["primary_drug"], a["combo_drug"])
    if not safe_anchors:
        log.warning("smbo_engine: all NCCN anchors contraindicated — warm-start skipped")
        return np.empty((0, N_TOTAL_DIMS)), np.empty(0), []

    anchors = safe_anchors
    rag_pens = batch_rag_penalties(anchors)
    scores = np.array([
        score_candidate(a, patient_vec, pfs_baseline, p,
                        cancer_type=cancer_type, pgx_profile=pgx_profile)
        for a, p in zip(anchors, rag_pens)
    ], dtype=np.float64)

    X_anchor = np.array([
        _encode_candidate(a, patient_vec) for a in anchors
    ], dtype=np.float64)
    y_anchor = scores

    # Replicate each anchor ×SMBO_ANCHOR_WEIGHT to amplify weight in GP
    X_rep = np.repeat(X_anchor, SMBO_ANCHOR_WEIGHT, axis=0)
    y_rep = np.repeat(y_anchor, SMBO_ANCHOR_WEIGHT)

    log.info("smbo_engine: warm-start — %d SOC anchors × %d replication = %d GP train pts",
             len(anchors), SMBO_ANCHOR_WEIGHT, len(y_rep))
    return X_rep, y_rep, anchors


# ── Random candidate generation ────────────────────────────────────────────────

def _random_candidates(
    n: int,
    patient_vec: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Sample n random candidates from the search space, return (X, cand_dicts)."""
    _load_drug_classes()
    n_drugs = len(_DRUG_CLASSES)

    cand_dicts: list[dict] = []
    X_list: list[np.ndarray] = []

    for _ in range(n):
        pd = _DRUG_CLASSES[rng.integers(0, n_drugs)]
        # combo: 30% chance "none", else a different drug
        if rng.random() < 0.30:
            cd = "none"
        else:
            cd_idx = rng.integers(0, n_drugs)
            cd = _DRUG_CLASSES[cd_idx]
            if cd == pd:
                cd = "none"  # avoid same-drug combo
        dose = float(rng.uniform(0.5, 1.0))
        cycles = int(rng.integers(1, 5))
        route = ROUTES[rng.integers(0, 3)]

        cand = {
            "primary_drug": pd,
            "combo_drug": cd,
            "dose_fraction": dose,
            "cycle_weeks": cycles,
            "route": route,
        }
        cand_dicts.append(cand)
        X_list.append(_encode_candidate(cand, patient_vec))

    return np.array(X_list, dtype=np.float64), cand_dicts


# ── Main SMBO loop ─────────────────────────────────────────────────────────────

def run_batched_smbo(
    patient_vec: np.ndarray,
    cancer_type: str = "glioblastoma",
    budget: int = SMBO_BUDGET,
    batch_size: int = SMBO_BATCH_SIZE,
    patient_id: str | None = None,
    pgx_profile=None,
) -> OptimizationResult:
    """Run the batched SMBO loop and return the optimisation result.

    Steps per iteration:
      1. RFR generates 1 000 random candidates, scored via GP posterior mean + EI
      2. Dual-sort: top-5 EI + top-5 PRED → batch of ≤10 unique candidates
      3. Evaluate all 10: score_candidate() + batch_rag_penalties()
      4. active_inference_acquisition() — adjusts next-iter acquisition weight
      5. GP updated with new (X, y) observations
      6. Early-stop check

    Returns OptimizationResult with top_3_candidates and plot paths.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(42)
    t_start = time.time()

    pfs_baseline = _pfs_from_patient_baseline(patient_vec)
    log.info("smbo_engine: starting SMBO for %s  baseline_pfs=%.1fw  budget=%d",
             cancer_type, pfs_baseline, budget)

    # ── 0. Warm-start ──────────────────────────────────────────────────────────
    X_obs, y_obs, _anchor_dicts = warm_start_from_nccn(cancer_type, patient_vec, pfs_baseline, pgx_profile=pgx_profile)
    anchor_names = [f"{a['primary_drug']}+{a['combo_drug']}" for a in _anchor_dicts]

    if X_obs.shape[0] == 0:
        # No warm-start → seed with a few random observations
        X_obs, rand_cands = _random_candidates(10, patient_vec, rng)
        rag_pens_init = batch_rag_penalties(rand_cands)
        y_obs = np.array([
            score_candidate(c, patient_vec, pfs_baseline, p, cancer_type=cancer_type, pgx_profile=pgx_profile)
            for c, p in zip(rand_cands, rag_pens_init)
        ])

    # ── Build SMBO GP (fresh, trained only on drug space) ─────────────────────
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    gp = _build_smbo_gp()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X_obs, y_obs)

    # ── Fast proxy RFR for 1000-candidate pre-scoring ─────────────────────────
    rfr_proxy = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2)
    rfr_proxy.fit(X_obs, y_obs)

    # Tracking
    all_evaluated: list[tuple[dict, float, np.ndarray]] = []  # (cand, score, X_vec)
    best_score = float(y_obs.max())
    best_sigma = 1.0
    early_stopped = False

    convergence_scores: list[float] = [best_score]

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    for iteration in range(budget):
        # 1. Generate 1 000 random candidates
        X_rand, rand_cands = _random_candidates(1000, patient_vec, rng)

        # 2. Pre-filter via RFR (fast pass — avoids GP on 1000 pts)
        rfr_preds = rfr_proxy.predict(X_rand)
        top_rfr_idx = np.argsort(rfr_preds)[-200:][::-1]  # top 200 by RFR
        X_top = X_rand[top_rfr_idx]
        top_cands = [rand_cands[i] for i in top_rfr_idx]

        # 3. GP posterior on top 200
        gp_mu, gp_sigma = gp.predict(X_top, return_std=True)

        # 4. Compute EI
        ei_scores = _ei(gp_mu, gp_sigma, best_score)
        best_sigma = float(gp_sigma.max())

        # 5. Active inference acquisition
        acq = active_inference_acquisition(ei_scores, gp_sigma, best_sigma=best_sigma)

        # 6. Dual-sort: EI pool + PRED pool
        ei_idx = np.argsort(acq)[-SMBO_EI_POOL:][::-1]
        pred_idx = np.argsort(gp_mu)[-SMBO_PRED_POOL:][::-1]
        batch_idx = list(dict.fromkeys(ei_idx.tolist() + pred_idx.tolist()))[:batch_size]

        batch_cands = [top_cands[i] for i in batch_idx]
        batch_X = X_top[batch_idx]

        # 7. Evaluate batch: full score_candidate + RAG penalty
        rag_pens = batch_rag_penalties(batch_cands)
        new_scores = np.array([
            score_candidate(c, patient_vec, pfs_baseline, p, cancer_type=cancer_type, pgx_profile=pgx_profile)
            for c, p in zip(batch_cands, rag_pens)
        ])

        # 8. Update observations
        X_obs = np.vstack([X_obs, batch_X])
        y_obs = np.concatenate([y_obs, new_scores])

        # 9. Update GP and RFR proxy (suppress harmless ConvergenceWarnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X_obs, y_obs)
        rfr_proxy.fit(X_obs, y_obs)

        # 10. Track best
        iter_best = float(new_scores.max())
        if iter_best > best_score:
            best_score = iter_best
        convergence_scores.append(best_score)

        for cand, sc, xv in zip(batch_cands, new_scores.tolist(), batch_X):
            all_evaluated.append((cand, sc, xv))

        # 11. Early stop
        if best_sigma < SMBO_SIGMA_EARLY_STOP:
            log.info("smbo_engine: early stop at iter %d — sigma=%.4f < %.4f",
                     iteration + 1, best_sigma, SMBO_SIGMA_EARLY_STOP)
            early_stopped = True
            break

        # Convergence: improvement < 0.001 for last 5 iters.
        #
        # P001-RUN-FIX: previously this fired on no-improvement ALONE,
        # which meant a low-information patient (e.g. dominant warm-start
        # anchor, no informative real outcomes yet) early-stopped at
        # iter 5 with σ≈1.2 — way too uncertain to be called convergence.
        # Now require BOTH no-improvement AND a sigma below 4×
        # SMBO_SIGMA_EARLY_STOP. Also require a minimum of 15 iters so
        # the warm-start prior doesn't short-circuit exploration.
        _MIN_ITERS = 15
        _SIGMA_NO_IMPROVE_BOUND = SMBO_SIGMA_EARLY_STOP * 4.0
        if (
            len(convergence_scores) >= 6
            and iteration + 1 >= _MIN_ITERS
        ):
            recent_improvement = convergence_scores[-1] - convergence_scores[-6]
            if (
                recent_improvement < 0.001
                and best_sigma < _SIGMA_NO_IMPROVE_BOUND
            ):
                log.info(
                    "smbo_engine: early stop at iter %d — no improvement "
                    "(σ=%.4f < %.4f, Δscore_5=%.5f)",
                    iteration + 1, best_sigma, _SIGMA_NO_IMPROVE_BOUND,
                    recent_improvement,
                )
                early_stopped = True
                break

    elapsed = time.time() - t_start
    log.info("smbo_engine: finished %d iters in %.1fs  best_score=%.2f  sigma=%.4f",
             len(convergence_scores) - 1, elapsed, best_score, best_sigma)

    # ── Build top-3 candidates ─────────────────────────────────────────────────
    # Sort all evaluated candidates by score
    all_evaluated.sort(key=lambda t: t[1], reverse=True)

    # Deduplicate by (primary_drug, combo_drug); skip invalid primaries
    seen_pairs: set[tuple[str, str]] = set()
    top_unique: list[tuple[dict, float]] = []
    for cand, sc, _ in all_evaluated:
        pd = cand.get("primary_drug", "")
        if not pd or pd.lower() in ("none", ""):
            continue  # primary_drug must be a real drug
        key = (pd, cand.get("combo_drug", "none"))
        if key not in seen_pairs:
            seen_pairs.add(key)
            top_unique.append((cand, sc))
        if len(top_unique) >= 10:
            break

    # Recompute GP-predicted PFS and sigma for top candidates
    top_candidates: list[SMBOCandidate] = []
    for rank, (cand, sc) in enumerate(top_unique[:3], start=1):
        xv = _encode_candidate(cand, patient_vec).reshape(1, -1)
        gp_mu_c, gp_sigma_c = gp.predict(xv, return_std=True)
        rag_pen = rag_penalty(cand["primary_drug"], cand["combo_drug"])

        # Determine pool: was this primarily an EI or PRED selection?
        pool = "PRED" if rank <= SMBO_PRED_POOL else "EI"

        # Task 6: toxicity scores for reporting (not re-penalised — already in score).
        tox_score, top_aes = predicted_toxicity(
            cand["primary_drug"], cand["combo_drug"],
            cand["dose_fraction"], patient_vec, pgx_profile=pgx_profile,
        )

        # Phase 5.2 / Module 5 — collect PGx audit notes for primary + combo
        # and capture the primary-drug eff multiplier for predicted_pfs_weeks.
        pgx_notes_c: list[str] = []
        pgx_adjusted_c = False
        pgx_pfs_eff_mult = 1.0
        if pgx_profile is not None:
            try:
                from .pgx_adjuster import apply_pgx
                p_eff, _, n1 = apply_pgx(cand["primary_drug"], pgx_profile)
                pgx_pfs_eff_mult = p_eff
                pgx_notes_c.extend(n1)
                cd_name = cand.get("combo_drug", "none")
                if cd_name and cd_name.lower() not in ("none", ""):
                    _, _, n2 = apply_pgx(cd_name, pgx_profile)
                    pgx_notes_c.extend(n2)
                pgx_adjusted_c = bool(pgx_notes_c)
            except Exception:
                pass

        top_candidates.append(SMBOCandidate(
            rank=rank,
            primary_drug=cand["primary_drug"],
            combo_drug=cand["combo_drug"],
            dose_fraction=round(cand["dose_fraction"], 3),
            cycle_weeks=cand["cycle_weeks"],
            route=cand["route"],
            predicted_pfs_weeks=round(
                pfs_baseline * _drug_benefit_factor(
                    cand["primary_drug"], cand["combo_drug"],
                    cand["dose_fraction"], cancer_type
                ) * pgx_pfs_eff_mult, 1
            ),
            recist_delta_pred=round((1.0 - cand["dose_fraction"]) * 30.0 - 15.0, 1),
            rag_penalty=round(rag_pen, 3),
            ei_score=round(float(np.ravel(
                _ei(gp_mu_c, gp_sigma_c, best_score)
            )[0]), 4),
            pred_score=round(float(np.ravel(gp_mu_c)[0]), 3),
            pool=pool,
            predicted_toxicity_score=round(tox_score, 4),
            top_aes=top_aes,
            pgx_adjusted=pgx_adjusted_c,
            pgx_notes=pgx_notes_c,
            # Phase 5.7 / Extra B — NCCN-alignment flags for FAERS gating.
            **dict(zip(
                ("off_label", "novel_combo"),
                classify_nccn_alignment(
                    cand["primary_drug"], cand.get("combo_drug", "none"), cancer_type,
                ),
            )),
        ))

    final_best = top_candidates[0] if top_candidates else None

    # ── Generate plots ─────────────────────────────────────────────────────────
    conv_path: str | None = None
    landscape_path: str | None = None

    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        from ..config import OUTPUTS_DIR, patient_out_dir

        # Plots dir — write directly into outputs/<pid>/plots/ when patient_id
        # is provided; otherwise fall back to a shared staging dir and let the
        # caller relocate.
        if patient_id:
            plots_dir = patient_out_dir(patient_id, "plots")
        else:
            plots_dir = OUTPUTS_DIR / "_smbo_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

        # Convergence plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(convergence_scores, marker="o", ms=3, lw=1.5, color="#2196F3")
        ax.set_xlabel("SMBO Iteration")
        ax.set_ylabel("Best Score (PFS weeks)")
        ax.set_title("SMBO Convergence — Best Score per Iteration")
        ax.grid(alpha=0.3)
        conv_path = str(plots_dir / "bo_convergence.png")
        fig.savefig(conv_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Landscape: drug index vs dose_fraction heatmap (top 5 drugs × dose levels)
        _load_drug_classes()
        n_drugs = len(_DRUG_CLASSES)
        dose_levels = np.linspace(0.5, 1.0, 10)
        drug_subset = _DRUG_CLASSES[:min(15, n_drugs)]
        Z = np.zeros((len(drug_subset), len(dose_levels)))
        for di, drug in enumerate(drug_subset):
            for dj, dose in enumerate(dose_levels):
                xv = _encode_candidate(
                    {"primary_drug": drug, "combo_drug": "none",
                     "dose_fraction": dose, "cycle_weeks": 4, "route": "iv"},
                    patient_vec,
                ).reshape(1, -1)
                Z[di, dj] = float(gp.predict(xv)[0])

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        im = ax2.imshow(Z, aspect="auto", origin="lower",
                        extent=[dose_levels[0], dose_levels[-1], 0, len(drug_subset)],
                        cmap="viridis")
        ax2.set_yticks(range(len(drug_subset)))
        ax2.set_yticklabels(drug_subset, fontsize=8)
        ax2.set_xlabel("Dose Fraction")
        ax2.set_title("GP Landscape — Drug × Dose (solo regimens)")
        plt.colorbar(im, ax=ax2, label="GP Score")
        landscape_path = str(plots_dir / "bo_landscape.png")
        fig2.savefig(landscape_path, dpi=100, bbox_inches="tight")
        plt.close(fig2)

        log.info("smbo_engine: plots saved to %s", plots_dir)

    except ImportError:
        log.warning("smbo_engine: matplotlib not available — plots skipped")
    except Exception as exc:
        log.warning("smbo_engine: plot generation failed: %s", exc)

    return OptimizationResult(
        triggered=True,
        n_iterations=len(convergence_scores) - 1,
        top_3_candidates=top_candidates,
        convergence_plot_path=conv_path,
        landscape_plot_path=landscape_path,
        final_best=final_best,
        sigma_at_convergence=round(best_sigma, 4),
        early_stopped=early_stopped,
        warm_start_anchors=anchor_names,
    )
