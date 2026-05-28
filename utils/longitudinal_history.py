"""Phase 5.3 / Module 2 — Longitudinal patient history (append-only JSONL).

Each successful pipeline run finalises one ``LongitudinalVisit`` line into
``outputs/<pid>/history/longitudinal.jsonl``. The file is append-only: it
survives ``WorkingMemory.finalize()`` legacy cleanup (see ``HITL_PROTECTED``
in memory.py) so trajectory across multiple visits is preserved.

Trajectory features (computed by ``compute_trajectory_features``):

    sod_growth_rate_mm_per_week
        Δ sum-of-diameters between the most recent two visits divided by
        the elapsed weeks. Positive = tumour growing.

    pfs_trajectory_slope
        Linear-regression slope of pfs_median_weeks across the last
        ``min(visit_count, 4)`` visits. Negative = predicted PFS shrinking
        visit-over-visit (worsening response).

    treatment_response_streak
        Number of consecutive trailing visits with recist_response in
        ``{"CR", "PR", "SD"}``.

    visit_count
        Number of prior visits already recorded for this patient. Does
        *not* include the current run (which has not been appended yet).

On the first visit (``visit_count=0``) all four features are population-
median imputed by the caller and flagged in ``imputation_mask``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schemas import LongitudinalHistory, LongitudinalVisit, PatientStateVector

log = logging.getLogger(__name__)

# Filename + subdir layout per the plan spec.
HISTORY_SUBDIR = "history"
HISTORY_FILENAME = "longitudinal.jsonl"

_RECIST_GOOD = {"CR", "PR", "SD"}


def _patient_history_path(patient_out_dir: Path, patient_id: str) -> Path:
    """Resolve the JSONL path under ``outputs/<pid>/history/``.

    ``patient_out_dir`` is the per-patient output root
    (``WorkingMemory.out_dir``). Created lazily on first write.
    """
    return Path(patient_out_dir) / HISTORY_SUBDIR / HISTORY_FILENAME


def load_history(patient_out_dir: Path, patient_id: str) -> LongitudinalHistory:
    """Load all prior visits for a patient. Returns empty history on first run."""
    path = _patient_history_path(patient_out_dir, patient_id)
    visits: list[LongitudinalVisit] = []
    if not path.exists():
        return LongitudinalHistory(patient_id=patient_id, visits=visits)

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                visits.append(LongitudinalVisit.model_validate(obj))
            except (json.JSONDecodeError, Exception) as exc:
                # Tolerate corrupt lines — the file is forward-extensible
                # and we'd rather lose one visit than crash the pipeline.
                log.warning(
                    "longitudinal_history: skipping malformed line in %s: %s",
                    path, exc,
                )
    except OSError as exc:
        log.warning("longitudinal_history: read failed for %s: %s", path, exc)

    # Sort by visit_date so trajectory regression is well-defined even if
    # appends were ever out of order (shouldn't happen, but defensive).
    visits.sort(key=lambda v: v.visit_date)
    return LongitudinalHistory(patient_id=patient_id, visits=visits)


def append_visit(
    patient_out_dir: Path,
    patient_id: str,
    *,
    visit_id: str,
    sum_of_diameters_mm: float = 0.0,
    pfs_median_weeks: float | None = None,
    recist_response: str | None = None,
    cancer_type: str | None = None,
    normalized: list[float] | None = None,
    visit_date: str | None = None,
) -> Path:
    """Append one visit to the JSONL log. Creates parent dir on demand.

    Returns the file path written to. Never raises on disk errors —
    logs and returns the path so the caller can decide whether to fail.
    """
    path = _patient_history_path(patient_out_dir, patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    visit = LongitudinalVisit(
        visit_id=visit_id,
        visit_date=visit_date or datetime.now(timezone.utc).isoformat(),
        sum_of_diameters_mm=float(sum_of_diameters_mm or 0.0),
        pfs_median_weeks=pfs_median_weeks,
        recist_response=recist_response,
        cancer_type=cancer_type,
        normalized=list(normalized or []),
    )

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(visit.model_dump_json() + "\n")
    except OSError as exc:
        log.error("longitudinal_history: append failed for %s: %s", path, exc)
    return path


# ── trajectory feature computation ─────────────────────────────────────────────

def _parse_iso_to_epoch(s: str) -> float | None:
    if not s:
        return None
    try:
        # Accept both "Z" suffix and explicit offset.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary least-squares slope. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def compute_trajectory_features(
    history: LongitudinalHistory,
    current_sod_mm: float,
) -> dict[str, float]:
    """Derive the 4 trajectory dims from a chronologically ordered history.

    ``current_sod_mm`` is the *current* run's sum-of-diameters — used to
    compute growth rate against the most recent prior visit.

    On first visit (no prior history) returns all zeros; the caller treats
    these as missing and lets normalisation impute to population median.
    """
    out: dict[str, float] = {
        "sod_growth_rate_mm_per_week": 0.0,
        "pfs_trajectory_slope":        0.0,
        "treatment_response_streak":   0.0,
        "visit_count":                 float(history.visit_count),
    }

    visits = history.visits
    if not visits:
        return out

    # ── sod_growth_rate_mm_per_week ─────────────────────────────────────
    last = visits[-1]
    last_epoch = _parse_iso_to_epoch(last.visit_date)
    now_epoch = datetime.now(timezone.utc).timestamp()
    if last_epoch and now_epoch > last_epoch:
        weeks = (now_epoch - last_epoch) / (60 * 60 * 24 * 7)
        if weeks > 0:
            delta = float(current_sod_mm or 0.0) - float(last.sum_of_diameters_mm or 0.0)
            out["sod_growth_rate_mm_per_week"] = delta / weeks

    # ── pfs_trajectory_slope (last ≤4 visits with non-null PFS) ─────────
    pfs_pts = [(_parse_iso_to_epoch(v.visit_date), v.pfs_median_weeks)
               for v in visits[-4:]]
    pfs_pts = [(e, p) for e, p in pfs_pts if e is not None and p is not None]
    if len(pfs_pts) >= 2:
        # Express x in weeks for interpretable slope units.
        e0 = pfs_pts[0][0]
        xs = [(e - e0) / (60 * 60 * 24 * 7) for e, _ in pfs_pts]
        ys = [float(p) for _, p in pfs_pts]
        out["pfs_trajectory_slope"] = float(_linear_slope(xs, ys))

    # ── treatment_response_streak ───────────────────────────────────────
    streak = 0
    for v in reversed(visits):
        if (v.recist_response or "").upper() in _RECIST_GOOD:
            streak += 1
        else:
            break
    out["treatment_response_streak"] = float(streak)

    return out


__all__ = [
    "HISTORY_SUBDIR",
    "HISTORY_FILENAME",
    "load_history",
    "append_visit",
    "compute_trajectory_features",
]
