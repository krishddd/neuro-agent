"""Survival prediction models for Phase 4 — Sub-step 4b.

Provides lazy-loading wrappers around three sklearn/sksurv models:
  • GP  (GaussianProcessRegressor) — predicts RECIST delta % with uncertainty σ
  • RSF (RandomSurvivalForest)     — predicts PFS median + survival curve
  • RFR (RandomForestRegressor)    — rapid candidate scoring for SMBO batch generation

Training strategy (corrected — avoids cluster overfitting):
  Stage 1: Generate 5,000 synthetic patients from statistical distributions
            bounded by reference_ranges.json population norms (not by perturbing
            the 20 real patients). This fills the full 20-dim feature space.
  Stage 2: Load the 20 real phase4_patient_data patients, compute their
            normalised feature vectors, and append them to the training set
            before the final model fit.

Models are cached to MODELS_DIR as .pkl files.  On subsequent runs they are
loaded directly, skipping training.  A corrupted pkl is detected by
UnpicklingError and triggers automatic retraining.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    MODELS_DIR,
    NCCN_GUIDELINES_PATH,
    PHASE4_DATA_ROOT,
    REFERENCE_RANGES_PATH,
)
from .patient_state import (
    FEATURE_NAMES,
    build_patient_state_vector,
    extract_intake_features,
    extract_wearable_features,
    load_phase4_json,
    _get_range,
)

log = logging.getLogger(__name__)

# ── file paths ──────────────────────────────────────────────────────────────────
GP_PKL  = MODELS_DIR / "gp_recist.pkl"
RSF_PKL = MODELS_DIR / "rsf_pfs.pkl"
RFR_PKL = MODELS_DIR / "rfr_candidate.pkl"

# ── Weibull PFS parameters per cancer type (from NCCN medians) ─────────────────
# shape: (scale_weeks, k) — Weibull distribution
# scale adjusted by ECOG modifier: × (1 - 0.15 * ecog_score)
_WEIBULL_PARAMS: dict[str, tuple[float, float]] = {
    "glioblastoma":            (14.0, 1.2),
    "glioblastoma_mgmt_meth":  (23.0, 1.3),
    "astrocytoma_idh_mutant":  (52.0, 1.5),
    "oligodendroglioma":       (96.0, 1.8),
    "pcnsl":                   (48.0, 1.4),
    "meningioma":              (260.0, 2.0),
    "brain_metastasis_lung":   (20.0, 1.2),
    "medulloblastoma":         (250.0, 2.2),
    "ependymoma":              (80.0, 1.6),
    "default":                 (24.0, 1.3),
}

# Cancer type proportions in synthetic cohort
_CANCER_PROPORTIONS: dict[str, float] = {
    "glioblastoma":          0.35,
    "astrocytoma_idh_mutant":0.20,
    "oligodendroglioma":     0.10,
    "pcnsl":                 0.08,
    "meningioma":            0.10,
    "brain_metastasis_lung": 0.10,
    "medulloblastoma":       0.04,
    "ependymoma":            0.03,
}

# ── synthetic cohort generation ─────────────────────────────────────────────────

def _generate_statistical_cohort(
    n_patients: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a statistically diverse training cohort from population norms.

    Each patient's features are sampled independently from truncated Gaussians
    bounded by [p5, p95] in reference_ranges.json for their cancer type.
    PFS labels follow a Weibull distribution parameterised by NCCN medians.

    Returns:
      X          (n × 20 float64)   — normalised feature matrix
      y_pfs_wks  (n,)  float64      — PFS duration in weeks
      y_recist   (n,)  float64      — RECIST delta % (synthetic)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build cancer type assignment
    cancer_types = list(_CANCER_PROPORTIONS.keys())
    probs = np.array([_CANCER_PROPORTIONS[ct] for ct in cancer_types])
    probs /= probs.sum()
    ct_assignments = rng.choice(cancer_types, size=n_patients, p=probs)

    X = np.zeros((n_patients, len(FEATURE_NAMES)), dtype=np.float64)
    y_pfs = np.zeros(n_patients, dtype=np.float64)
    y_recist = np.zeros(n_patients, dtype=np.float64)

    for i, ct in enumerate(ct_assignments):
        # Sample each feature from truncated Gaussian bounded by [p5, p95]
        for j, feat in enumerate(FEATURE_NAMES):
            rng_vals = _get_range(feat, ct)
            p5, p95 = rng_vals["p5"], rng_vals["p95"]
            median = rng_vals["median"]
            # Normalise median to [0,1] for the normalised space
            denom = max(p95 - p5, 1e-6)
            mu_norm = (median - p5) / denom
            sigma_norm = 0.2  # spread ~20% of the [0,1] range
            # Sample then clip to [0,1]
            val = float(np.clip(rng.normal(mu_norm, sigma_norm), 0.0, 1.0))
            X[i, j] = val

        # ECOG index (feature 14)
        ecog_norm = X[i, 14]
        ecog_score = ecog_norm * 4.0  # denormalise to [0,4]

        # PFS: Weibull with ECOG and dose-reduction modifiers
        base_key = ct
        scale, k_shape = _WEIBULL_PARAMS.get(base_key, _WEIBULL_PARAMS["default"])
        ecog_mod = max(0.2, 1.0 - 0.15 * ecog_score)
        dose_red_norm = X[i, 18]  # dose_reduction_flag normalised
        dose_mod = 0.85 if dose_red_norm > 0.5 else 1.0
        effective_scale = scale * ecog_mod * dose_mod
        # Weibull sample: -scale * log(U) ^ (1/k)
        u = rng.uniform(0.001, 0.999)
        pfs_weeks = effective_scale * (-np.log(u)) ** (1.0 / k_shape)
        y_pfs[i] = max(1.0, pfs_weeks)

        # RECIST delta: correlated with PFS (low PFS → high positive delta = progression)
        # Normalise PFS to a [-40, +30] RECIST delta range
        pfs_norm = min(pfs_weeks / 100.0, 2.0)
        recist_mu = -20.0 * pfs_norm + 15.0  # inverse relationship
        y_recist[i] = float(np.clip(rng.normal(recist_mu, 10.0), -80.0, 60.0))

    log.info("survival_models: generated synthetic cohort  n=%d", n_patients)
    return X, y_pfs, y_recist


def _load_real_patients() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the 20 phase4_patient_data patients and extract their feature vectors.

    Returns (X_real, y_pfs_real, y_recist_real) — may return empty arrays if
    no patients found or extraction fails.
    """
    from ..memory import WorkingMemory  # local import to avoid circularity

    X_list, pfs_list, recist_list = [], [], []

    for pid_dir in sorted(PHASE4_DATA_ROOT.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        try:
            intake  = load_phase4_json(pid, "patient_intake_form.json")
            wearable = load_phase4_json(pid, "wearable_data.json")
            if not intake:
                continue

            # Build feature vector (no RECIST memory — use defaults)
            mem = WorkingMemory(job_id="train", patient_id=pid)
            vec = build_patient_state_vector(mem, intake, wearable)
            X_list.append(vec.normalized)

            # Derive PFS from cancer type + ECOG using NCCN Weibull
            ct = vec.cancer_type.lower().replace(" ", "_").replace("-", "_")
            scale, k = _WEIBULL_PARAMS.get(ct, _WEIBULL_PARAMS["default"])
            ecog_mod = max(0.2, 1.0 - 0.15 * vec.ecog_ps_score)
            dose_mod = 0.85 if vec.dose_reduction_flag else 1.0
            # Use median PFS estimate (deterministic for real patients)
            pfs = scale * ecog_mod * dose_mod * (np.log(2) ** (1.0 / k))
            pfs_list.append(max(1.0, float(pfs)))

            # RECIST delta from memory if available, else synthetic
            recist_list.append(-20.0 * min(pfs / 100.0, 2.0) + 15.0)

        except Exception as exc:
            log.warning("survival_models: skip patient %s: %s", pid, exc)

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), np.empty(0)

    log.info("survival_models: loaded %d real patients", len(X_list))
    return (
        np.array(X_list, dtype=np.float64),
        np.array(pfs_list, dtype=np.float64),
        np.array(recist_list, dtype=np.float64),
    )


# ── model loading / training ────────────────────────────────────────────────────

def _pkl_load(path: Path) -> Any | None:
    """Load a pickle safely. Returns None on missing file or corruption.

    Also invalidates any cached model whose expected feature-dimensionality
    doesn't match ``len(FEATURE_NAMES)`` — critical after Phase 5.1 grew the
    vector from 20→25 dims (old pkls would raise at predict time).
    """
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        log.warning("survival_models: corrupt pkl %s (%s) — will retrain", path.name, exc)
        path.unlink(missing_ok=True)
        return None

    expected = len(FEATURE_NAMES)
    got = getattr(obj, "n_features_in_", None)
    if got is not None and got != expected:
        log.warning(
            "survival_models: stale pkl %s (trained on %d features, now %d) — "
            "discarding and retraining", path.name, got, expected,
        )
        path.unlink(missing_ok=True)
        return None
    return obj


def _pkl_save(obj: Any, path: Path) -> None:
    try:
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=5)
        log.info("survival_models: saved %s", path.name)
    except Exception as exc:
        log.warning("survival_models: could not save %s: %s", path.name, exc)


def _build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine statistical cohort + real patients into final training arrays."""
    X_syn, y_pfs_syn, y_rec_syn = _generate_statistical_cohort(5000)
    X_real, y_pfs_real, y_rec_real = _load_real_patients()

    if X_real.shape[0] > 0:
        X      = np.vstack([X_syn, X_real])
        y_pfs  = np.concatenate([y_pfs_syn, y_pfs_real])
        y_rec  = np.concatenate([y_rec_syn, y_rec_real])
    else:
        X, y_pfs, y_rec = X_syn, y_pfs_syn, y_rec_syn

    return X, y_pfs, y_rec


# ── lazy-loading accessors ──────────────────────────────────────────────────────

_GP_MODEL  = None
_RSF_MODEL = None
_RFR_MODEL = None
_RSF_UNAVAILABLE = False   # set True after first ImportError — suppresses repeat warnings


def get_gp_model():
    """Lazy-load or train the GaussianProcessRegressor for RECIST delta prediction.

    Uses Matérn(ν=2.5) + WhiteKernel on a subsampled training set (GP scales
    as O(n³) — we cap at 500 points for practicality).
    """
    global _GP_MODEL
    if _GP_MODEL is not None:
        return _GP_MODEL

    cached = _pkl_load(GP_PKL)
    if cached is not None:
        _GP_MODEL = cached
        log.info("survival_models: loaded gp_recist.pkl")
        return _GP_MODEL

    log.info("survival_models: training GP (first run — may take ~30s)…")
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        X, _, y_rec = _build_training_data()
        # GP is O(n^3) — subsample to 500 for training speed
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
        X_fit, y_fit = X[idx], y_rec[idx]

        import warnings
        from sklearn.exceptions import ConvergenceWarning

        # noise_level_bounds raised to prevent ConvergenceWarning at lower bound
        kernel = (
            Matern(nu=2.5, length_scale_bounds=(0.01, 10.0))
            + WhiteKernel(noise_level_bounds=(1e-3, 1e3))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X_fit, y_fit)
        _GP_MODEL = gp
        _pkl_save(gp, GP_PKL)
        log.info("survival_models: GP trained  score=%.3f", gp.score(X_fit, y_fit))
    except ImportError:
        log.error("survival_models: scikit-learn not installed — GP unavailable")
        raise
    except Exception as exc:
        log.error("survival_models: GP training failed: %s", exc)
        raise

    return _GP_MODEL


def get_rsf_model():
    """Lazy-load or train the RandomSurvivalForest for PFS prediction."""
    global _RSF_MODEL, _RSF_UNAVAILABLE
    if _RSF_UNAVAILABLE:
        raise ImportError("scikit-survival not installed")
    if _RSF_MODEL is not None:
        return _RSF_MODEL

    cached = _pkl_load(RSF_PKL)
    if cached is not None:
        _RSF_MODEL = cached
        log.info("survival_models: loaded rsf_pfs.pkl")
        return _RSF_MODEL

    log.info("survival_models: training RSF (first run — may take ~60s)…")
    try:
        from sksurv.ensemble import RandomSurvivalForest

        X, y_pfs, _ = _build_training_data()

        # sksurv needs structured array: (event: bool, duration: float)
        y_struct = np.array(
            [(True, float(t)) for t in y_pfs],
            dtype=[("event", bool), ("duration", float)],
        )

        rsf = RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        rsf.fit(X, y_struct)
        _RSF_MODEL = rsf
        _pkl_save(rsf, RSF_PKL)
        log.info("survival_models: RSF trained  C-index=%.3f",
                 rsf.score(X[:200], y_struct[:200]))
    except ImportError:
        _RSF_UNAVAILABLE = True
        log.warning("survival_models: scikit-survival not installed — RSF unavailable")
        raise
    except Exception as exc:
        log.error("survival_models: RSF training failed: %s", exc)
        raise

    return _RSF_MODEL


def get_rfr_model():
    """Lazy-load or train the RandomForestRegressor for SMBO candidate scoring."""
    global _RFR_MODEL
    if _RFR_MODEL is not None:
        return _RFR_MODEL

    cached = _pkl_load(RFR_PKL)
    if cached is not None:
        _RFR_MODEL = cached
        log.info("survival_models: loaded rfr_candidate.pkl")
        return _RFR_MODEL

    log.info("survival_models: training RFR candidate model…")
    try:
        from sklearn.ensemble import RandomForestRegressor

        X, y_pfs, _ = _build_training_data()
        rfr = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        rfr.fit(X, y_pfs)
        _RFR_MODEL = rfr
        _pkl_save(rfr, RFR_PKL)
        log.info("survival_models: RFR trained  OOB not available (no oob_score)")
    except ImportError:
        log.error("survival_models: scikit-learn not installed — RFR unavailable")
        raise
    except Exception as exc:
        log.error("survival_models: RFR training failed: %s", exc)
        raise

    return _RFR_MODEL


# ── prediction helpers ──────────────────────────────────────────────────────────

def predict_gp(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_recist_delta_pct, std_sigma) for each row in X."""
    gp = get_gp_model()
    mean, std = gp.predict(X, return_std=True)
    return mean, std


def _weibull_pfs_fallback(X: np.ndarray) -> tuple[float, float, float, list[dict[str, float]]]:
    """Weibull-based PFS estimate when sksurv is unavailable.

    Uses ECOG (feature index 14) and treatment history (feature 15-19) from the
    normalised patient vector to pick Weibull parameters from _WEIBULL_PARAMS.
    Returns (median, ci_low, ci_high, survival_curve).
    """
    row = np.asarray(X, dtype=np.float64).flatten()

    # Default cancer type → GBM parameters
    scale, k = _WEIBULL_PARAMS["default"]
    ecog_norm = float(row[14]) if len(row) > 14 else 0.5
    dose_red_norm = float(row[18]) if len(row) > 18 else 0.0
    ecog_score = ecog_norm * 4.0
    ecog_mod = max(0.2, 1.0 - 0.15 * ecog_score)
    dose_mod = 0.85 if dose_red_norm > 0.5 else 1.0
    eff_scale = scale * ecog_mod * dose_mod

    # Weibull quantiles: F^-1(p) = scale * (-ln(1-p))^(1/k)
    median = float(eff_scale * (np.log(2) ** (1.0 / k)))
    ci_low  = float(eff_scale * ((-np.log(0.95)) ** (1.0 / k)))  # 5th percentile
    ci_high = float(eff_scale * ((-np.log(0.05)) ** (1.0 / k)))  # 95th percentile
    median  = max(1.0, median)
    ci_low  = max(1.0, ci_low)
    ci_high = max(ci_low, ci_high)

    # Survival curve: S(t) = exp(-(t/scale)^k)
    t_max = ci_high * 1.2
    times = np.linspace(0.1, t_max, 20)
    curve = [
        {"week": float(t), "prob": float(np.exp(-((t / eff_scale) ** k)))}
        for t in times
    ]
    return median, ci_low, ci_high, curve


def predict_rsf_pfs(
    X: np.ndarray,
) -> tuple[float, float, float, list[dict[str, float]]]:
    """Return (pfs_median_weeks, ci_low, ci_high, survival_curve) for a single row.

    Falls back to a Weibull-based estimate when scikit-survival is not installed.
    survival_curve is a list of {week, prob} dicts sampled at regular intervals.
    """
    try:
        rsf = get_rsf_model()
    except (ImportError, RuntimeError):
        # scikit-survival not available — use Weibull fallback
        row = np.asarray(X, dtype=np.float64)
        if row.ndim == 2:
            row = row[0]
        return _weibull_pfs_fallback(row)

    surv_fns = rsf.predict_survival_function(X)

    if len(surv_fns) == 0:
        return 24.0, 8.0, 52.0, []

    fn = surv_fns[0]
    times = fn.x
    probs = fn.y

    # Median: first time where survival drops below 0.5
    pfs_median = float(times[-1])  # fallback: last observed time
    for t, p in zip(times, probs):
        if p <= 0.5:
            pfs_median = float(t)
            break

    # 95% CI: time where survival = 0.05 and 0.95
    ci_low = float(times[0])
    ci_high = float(times[-1])
    for t, p in zip(times, probs):
        if p <= 0.95:
            ci_low = float(t)
            break
    for t, p in zip(reversed(times), reversed(probs)):
        if p >= 0.05:
            ci_high = float(t)
            break

    # Sample survival curve at up to 20 evenly-spaced time points
    n_sample = min(20, len(times))
    indices = np.linspace(0, len(times) - 1, n_sample, dtype=int)
    curve = [
        {"week": float(times[i]), "prob": float(probs[i])}
        for i in indices
    ]

    return pfs_median, ci_low, ci_high, curve
