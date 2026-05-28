"""Task 8 — Clinical trial matching sub-step (4d.5).

Runs AFTER SHAP (4d) and BEFORE the MDT debate (4e) when the patient has
poor predicted PFS or radiologic progression. Queries ClinicalTrials.gov
via the compact field-projected client (``clinicaltrials_client``),
scores each candidate trial against the patient, and hands the top-3
matches into the MDT persona prompts.

Trigger:
    pred.pfs_median_weeks < TRIAL_MATCH_PFS_THRESHOLD_WEEKS
        OR RANO.response == "PD"
        OR RECIST.response == "PD"

Match score (deterministic, 0.0–1.0):
    +0.5  condition keyword matches patient diagnosis
    +0.2  biomarker eligibility matches patient MGMT / IDH status
    +0.2  intervention not already in patient's current meds (novel option)
    +0.1  Phase II or III
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..config import TRIAL_MATCH_PFS_THRESHOLD_WEEKS
from ..memory import WorkingMemory
from ..utils.clinicaltrials_client import search_trials
from ..utils.schemas import (
    ClinicalTrialMatch,
    PatientStateVector,
    PredictionResult,
    TrialMatchResult,
)
from . import register

log = logging.getLogger(__name__)

# Phase 4 should still emit an S17b envelope even when no trials match,
# so downstream (synthesis, report) can rely on its presence.


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_condition_query(diagnosis: str | None, cancer_type: str | None) -> str:
    """Concatenate the most specific diagnosis string available."""
    for candidate in (diagnosis, cancer_type):
        if candidate and isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "glioblastoma"


def _diagnosis_keywords(diagnosis: str) -> set[str]:
    """Lower-case 3+ char tokens from the diagnosis for match scoring."""
    toks = re.findall(r"[A-Za-z]{3,}", diagnosis or "")
    return {t.lower() for t in toks if t.lower() not in {"the", "and", "with", "grade"}}


def _novel_intervention(interventions: list[str], current_meds: list[str]) -> bool:
    """True if at least one intervention is not already in current meds."""
    if not interventions:
        return False
    cur = {m.lower() for m in current_meds}
    for iv in interventions:
        iv_l = (iv or "").lower()
        if not iv_l:
            continue
        # Any non-trivial intervention that isn't already on current meds.
        if not any(c in iv_l or iv_l in c for c in cur):
            return True
    return False


def _phase_bonus(phase: str) -> float:
    p = (phase or "").lower()
    return 0.1 if ("phase 2" in p or "phase 3" in p or "phase ii" in p or "phase iii" in p) else 0.0


def _score_one(
    trial: dict,
    diag_tokens: set[str],
    mgmt: str | None,
    idh: str | None,
    current_meds: list[str],
) -> tuple[float, str]:
    """Deterministic 0–1 score plus a short reasoning string."""
    score = 0.0
    reasons: list[str] = []

    # +0.5 condition keyword match (check title + interventions + elig summary)
    hay = " ".join([
        trial.get("title", ""),
        " ".join(trial.get("interventions", []) or []),
        trial.get("eligibility_summary", ""),
    ]).lower()
    if any(tok in hay for tok in diag_tokens):
        score += 0.5
        reasons.append("diagnosis match")

    # +0.2 biomarker eligibility match (MGMT / IDH)
    elig = (trial.get("eligibility_summary") or "").lower()
    bm_hit = False
    if mgmt and mgmt in ("methylated", "unmethylated") and "mgmt" in elig:
        if mgmt in elig:
            score += 0.1
            reasons.append(f"MGMT {mgmt} eligible")
            bm_hit = True
    if idh and idh in ("mutant", "wildtype") and ("idh" in elig):
        idh_kw = "mutant" if idh == "mutant" else ("wildtype" if idh == "wildtype" else "")
        if idh_kw and idh_kw in elig:
            score += 0.1
            reasons.append(f"IDH {idh} eligible")
            bm_hit = True
    if not bm_hit and ("mgmt" in elig or "idh" in elig):
        # Trial stratifies on a biomarker but we don't know patient status —
        # neutral, no bonus, note it.
        reasons.append("biomarker-stratified")

    # +0.2 novel intervention
    if _novel_intervention(trial.get("interventions") or [], current_meds):
        score += 0.2
        reasons.append("novel intervention")

    # +0.1 Phase II / III
    pb = _phase_bonus(trial.get("phase", ""))
    if pb > 0:
        score += pb
        reasons.append(f"{trial.get('phase')}")

    return round(min(score, 1.0), 3), "; ".join(reasons) if reasons else "no matching criteria"


def _should_trigger(memory: WorkingMemory, pred: PredictionResult | None) -> tuple[bool, str]:
    """Decide whether trial matching should run for this patient."""
    # 1. RANO progression → always trigger
    rano_raw = memory.get(WorkingMemory.RANO)
    if rano_raw:
        rr = rano_raw if isinstance(rano_raw, dict) else rano_raw.model_dump()
        if rr.get("response") == "PD":
            return True, "RANO=PD (progressive disease)"

    # 2. RECIST progression (fallback)
    rec_raw = memory.get(WorkingMemory.RECIST)
    if rec_raw:
        r = rec_raw if isinstance(rec_raw, dict) else rec_raw.model_dump()
        if r.get("response") == "PD":
            return True, "RECIST=PD (progressive disease)"

    # 3. Poor predicted PFS
    if pred and pred.pfs_median_weeks is not None \
       and pred.pfs_median_weeks < TRIAL_MATCH_PFS_THRESHOLD_WEEKS:
        return True, (f"predicted PFS {pred.pfs_median_weeks:.1f}w "
                      f"< threshold {TRIAL_MATCH_PFS_THRESHOLD_WEEKS}w")

    return False, ""


# ── Main tool ─────────────────────────────────────────────────────────────────

@register("match_clinical_trials")
def match_clinical_trials(memory: WorkingMemory, **_) -> dict:
    """Sub-step 4d.5 — query CT.gov and score top matches for the patient."""
    pred_raw = memory.get(WorkingMemory.PREDICTION)
    pred: PredictionResult | None = None
    if pred_raw:
        pred = (pred_raw if isinstance(pred_raw, PredictionResult)
                else PredictionResult.model_validate(pred_raw))

    triggered, reason = _should_trigger(memory, pred)
    if not triggered:
        skipped = TrialMatchResult(
            triggered=False,
            trigger_reason="patient not at poor-PFS / PD threshold",
            n_searched=0,
            top_matches=[],
        )
        memory.set(WorkingMemory.TRIAL_MATCHES, skipped)
        log.info("trial_match: skipped (not triggered)")
        return {"ok": True, "skipped": True}

    # Resolve the diagnosis / cancer-type query + patient signals.
    diagnosis: str | None = None
    rec_raw = memory.get(WorkingMemory.RECORD)
    if rec_raw:
        r = rec_raw if isinstance(rec_raw, dict) else rec_raw.model_dump()
        raw_diag = r.get("diagnosis")
        if isinstance(raw_diag, dict):
            diagnosis = raw_diag.get("histology") or raw_diag.get("primary") \
                or raw_diag.get("type")
        elif isinstance(raw_diag, str):
            diagnosis = raw_diag

    ps_raw = memory.get(WorkingMemory.PATIENT_STATE)
    mgmt: str | None = None
    idh:  str | None = None
    cancer_type: str | None = None
    if ps_raw:
        ps = (ps_raw if isinstance(ps_raw, PatientStateVector)
              else PatientStateVector.model_validate(ps_raw))
        mgmt        = ps.mgmt_methylation
        idh         = ps.idh_mutation
        cancer_type = ps.cancer_type

    condition = _build_condition_query(diagnosis, cancer_type)
    current_meds: list[str] = memory._store.get("pre_extracted_meds", []) or []

    # Fetch trials (uses 24h disk cache + field projection).
    studies = search_trials(condition=condition, max_results=20)
    log.info("trial_match: %d trials retrieved for condition=%r", len(studies), condition)

    diag_tokens = _diagnosis_keywords(
        f"{diagnosis or ''} {cancer_type or ''}".strip() or condition
    )

    matches: list[ClinicalTrialMatch] = []
    for s in studies:
        score, why = _score_one(s, diag_tokens, mgmt, idh, current_meds)
        if score <= 0.0:
            continue
        matches.append(ClinicalTrialMatch(
            nct_id=s.get("nct_id", ""),
            title=s.get("title", "")[:200],
            phase=s.get("phase", ""),
            status=s.get("status", ""),
            interventions=(s.get("interventions") or [])[:6],
            eligibility_summary=(s.get("eligibility_summary") or "")[:500],
            match_score=score,
            match_reasoning=why,
        ))

    matches.sort(key=lambda m: m.match_score, reverse=True)
    top = matches[:3]

    result = TrialMatchResult(
        triggered=True,
        trigger_reason=reason,
        n_searched=len(studies),
        top_matches=top,
        search_query=condition,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    memory.set(WorkingMemory.TRIAL_MATCHES, result)

    log.info("trial_match: %d/%d scored — top=%s (%.2f)",
             len(top), len(studies),
             top[0].nct_id if top else "n/a",
             top[0].match_score if top else 0.0)

    return {
        "ok": True,
        "n_searched": len(studies),
        "n_top_matches": len(top),
        "top_nct_id": top[0].nct_id if top else None,
    }
