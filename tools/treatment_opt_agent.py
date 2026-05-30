"""Phase 4 — Treatment Optimization (SMBO v3.0) tools.

Five deterministic sub-steps registered as tool callables:

    extract_patient_state   → S14_patient_state.json
    predict_recist_pfs      → S15_prediction.json
    run_smbo_optimization   → S16_optimization.json + plots
    explain_with_shap       → S17_shap.json + waterfall.png
    review_proposal_mdt     → S18_treatment_proposal.json

Clinical safety note: sub-step 4a (extract_patient_state) pre-extracts
the patient's current medications from the structured intake form and
pre-seeds the RAG penalty cache BEFORE the SMBO loop starts (4c). This
ensures proposed drugs are always checked against existing regimens even
though the full Pharma phase (S7/S8) runs later.

Google Workspace hooks fire after review_proposal_mdt when conditions
are met (MDT discussion required, REJECT, urgency ≥ 4).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import (
    CALENDAR_ENABLED,
    DRIVE_ENABLED,
    GMAIL_ENABLED,
    MODEL_THINKING,
    PROMPTS_DIR,
    RECIST_DELTA_SD_TRIGGER,
    SMBO_SIGMA_TRIGGER,
)
from ..memory import WorkingMemory
from ..utils.schemas import (
    MDTPersonaTurn,
    OptimizationResult,
    PatientStateVector,
    PredictionResult,
    ShapDriver,
    ShapResult,
    TreatmentProposal,
)
from . import register

log = logging.getLogger(__name__)

# ── MDT reviewer system prompt (loaded once) ──────────────────────────────────
_MDT_SYSTEM_PROMPT: str | None = None
_MDT_PERSONA_CACHE: dict[str, str] = {}

_MDT_PERSONAS = ("neuroradiologist", "neurosurgeon", "neurooncologist", "pharmacist")


def _load_mdt_prompt() -> str:
    global _MDT_SYSTEM_PROMPT
    if _MDT_SYSTEM_PROMPT is not None:
        return _MDT_SYSTEM_PROMPT
    p = PROMPTS_DIR / "treatment_opt_system.md"
    if p.exists():
        _MDT_SYSTEM_PROMPT = p.read_text(encoding="utf-8")
    else:
        _MDT_SYSTEM_PROMPT = (
            "You are a senior neuro-oncology MDT board reviewer. "
            "Review the proposed treatment and output a JSON TreatmentProposal "
            "with keys: decision, reason, proposed_regimen, modifications, "
            "contraindications_checked, guideline_alignment, mdt_discussion_required, "
            "rag_interaction_flags, clinical_narrative."
        )
    return _MDT_SYSTEM_PROMPT


def _load_persona_prompt(persona: str) -> str:
    """Load an MDT persona prompt file, cached in-process."""
    cached = _MDT_PERSONA_CACHE.get(persona)
    if cached is not None:
        return cached
    fname = {
        "neuroradiologist": "mdt_neuroradiologist.md",
        "neurosurgeon":     "mdt_neurosurgeon.md",
        "neurooncologist":  "mdt_neurooncologist.md",
        "pharmacist":       "mdt_pharmacist.md",
        "chair":            "mdt_chair_synthesis.md",
    }.get(persona)
    if not fname:
        raise ValueError(f"unknown MDT persona: {persona}")
    path = PROMPTS_DIR / fname
    text = path.read_text(encoding="utf-8") if path.exists() else (
        f"You are the {persona} on a tumour-board. "
        f"Return JSON with keys persona, round, statement, concerns, "
        f"agreement_with_proposal."
    )
    _MDT_PERSONA_CACHE[persona] = text
    return text


# ── Medication pre-extraction helpers ─────────────────────────────────────────

# Simple regex to pull drug names from free-text regimen strings
_DRUG_NAME_RE = re.compile(
    r'\b(temozolomide|bevacizumab|lomustine|carmustine|procarbazine|'
    r'vincristine|cisplatin|carboplatin|etoposide|methotrexate|rituximab|'
    r'nivolumab|pembrolizumab|dexamethasone|erlotinib|osimertinib|everolimus|'
    r'temsirolimus|irinotecan|paclitaxel|docetaxel|capecitabine|imatinib|'
    r'sunitinib|sorafenib|regorafenib|lapatinib|palbociclib|olaparib|'
    r'valproic acid|levetiracetam|omeprazole|metformin|ondansetron|'
    r'warfarin|apixaban|rivaroxaban|dabigatran|filgrastim|trastuzumab|'
    r'TMZ|BEV|CCNU|BCNU|MTX)\b',
    re.IGNORECASE,
)


def _pre_extract_medications_from_intake(intake_form: dict | None) -> list[str]:
    """Extract current drug names from intake_form structured data.

    Reads phase4_treatment_history.current_regimen (free-text) and
    phase4_treatment_history.prior_regimen to build a drug name list.
    Falls back to empty list when intake_form is None.
    """
    if not intake_form:
        return []

    drug_names: set[str] = set()
    hist = intake_form.get("phase4_treatment_history", {})

    # Current regimen (may be list of dicts or a single string)
    current_reg = hist.get("current_regimen", "")
    if isinstance(current_reg, list):
        text_blob = " ".join(
            str(r.get("drug", r) if isinstance(r, dict) else r)
            for r in current_reg
        )
    else:
        text_blob = str(current_reg or "")

    for m in _DRUG_NAME_RE.finditer(text_blob):
        drug_names.add(m.group(0).lower())

    # Also check prior regimens
    for reg in hist.get("prior_regimen", []) if isinstance(hist.get("prior_regimen"), list) else []:
        reg_text = str(reg.get("drugs", reg) if isinstance(reg, dict) else reg)
        for m in _DRUG_NAME_RE.finditer(reg_text):
            drug_names.add(m.group(0).lower())

    # Top-level allergies / medications list
    for allergy in intake_form.get("allergies", []):
        for m in _DRUG_NAME_RE.finditer(str(allergy)):
            drug_names.add(m.group(0).lower())

    result = sorted(drug_names)
    log.info("treatment_opt: pre-extracted %d current drug names: %s", len(result), result)
    return result


def _pre_seed_interaction_cache(current_drug_names: list[str]) -> None:
    """Pre-seed RAG penalty cache for all (current × SMBO search-space) pairs."""
    if not current_drug_names:
        return
    try:
        from ..utils.rag_penalty import preseed_penalty_cache
        from ..utils.smbo_engine import _load_drug_classes

        smbo_drugs = _load_drug_classes()
        preseed_penalty_cache(current_drug_names, smbo_drugs)
        log.info("treatment_opt: RAG cache pre-seeded for %d × %d pairs",
                 len(current_drug_names), len(smbo_drugs))
    except Exception as exc:
        log.warning("treatment_opt: penalty cache pre-seed failed: %s", exc)


# ── Sub-step 4a: extract_patient_state ────────────────────────────────────────

@register("extract_patient_state")
def extract_patient_state(memory: WorkingMemory, **_) -> dict:
    """Build 20-dim PatientStateVector from intake form, wearable data, and pipeline memory.

    Also pre-extracts current medications and pre-seeds the RAG interaction
    cache so the SMBO loop is clinically safe from the start.
    """
    pid = memory.patient_id

    # Load phase4 structured files
    from ..utils.patient_state import build_patient_state_vector, load_phase4_json

    intake_form  = load_phase4_json(pid, "patient_intake_form.json")
    wearable_data = load_phase4_json(pid, "wearable_data.json")

    if not intake_form:
        log.info("treatment_opt: no intake_form for %s — all features will be imputed", pid)

    # Build 20-dim vector
    vec = build_patient_state_vector(memory, intake_form, wearable_data)
    memory.set(WorkingMemory.PATIENT_STATE, vec)

    # --- Clinical safety: pre-extract meds BEFORE SMBO ---
    current_drugs = _pre_extract_medications_from_intake(intake_form)
    memory._store["pre_extracted_meds"] = current_drugs   # internal key, no stage file
    _pre_seed_interaction_cache(current_drugs)

    # --- LLM enrichment: plain-English wearable trend narrative (gemma4:e4b) ---
    wearable_blurb = ""
    try:
        daily = None
        if isinstance(wearable_data, dict):
            daily = wearable_data.get("daily_data") or wearable_data.get("daily") \
                or wearable_data.get("days") or wearable_data.get("data")
        if daily:
            from ..utils.llm_enrichment import wearable_narrative
            wearable_blurb = wearable_narrative(daily) or ""
    except Exception as exc:
        log.warning("treatment_opt: wearable_narrative failed: %s", exc)
    if wearable_blurb:
        memory._store["wearable_narrative"] = wearable_blurb

    # --- Phase 5.2 / Module 5: PGx metabolizer summary ---
    try:
        from ..utils.pgx_adjuster import summarize_patient_pgx
        pgx_summary = summarize_patient_pgx(getattr(vec, "pgx_profile", None))
        log.info("treatment_opt: 4a %s — %s", pid, pgx_summary)
    except Exception as exc:
        log.warning("treatment_opt: pgx summary failed: %s", exc)

    log.info("treatment_opt: 4a complete — %s  imputed=%d/%d  cancer=%s  wearable_narrative=%s",
             pid, sum(vec.imputation_mask), len(vec.imputation_mask), vec.cancer_type,
             "yes" if wearable_blurb else "no")
    return {"ok": True, "imputed_features": sum(vec.imputation_mask),
            "cancer_type": vec.cancer_type,
            "wearable_narrative": bool(wearable_blurb)}


# ── Sub-step 4b: predict_recist_pfs ───────────────────────────────────────────

@register("predict_recist_pfs")
def predict_recist_pfs(memory: WorkingMemory, **_) -> dict:
    """Predict RECIST delta (GP) and PFS (RSF/Weibull).  Sets optimization_triggered."""
    memory.require(WorkingMemory.PATIENT_STATE)

    import numpy as np

    from ..utils.survival_models import predict_gp, predict_rsf_pfs

    vec: PatientStateVector = memory.get(WorkingMemory.PATIENT_STATE)
    if isinstance(vec, dict):
        vec = PatientStateVector.model_validate(vec)

    X = np.array(vec.normalized, dtype=np.float64).reshape(1, -1)

    # GP — RECIST delta prediction with uncertainty
    try:
        gp_mean, gp_std = predict_gp(X)
        recist_delta_pred = float(gp_mean[0])
        recist_sigma      = float(gp_std[0])
    except Exception as exc:
        log.warning("treatment_opt: GP prediction failed (%s) — using defaults", exc)
        recist_delta_pred = 10.0
        recist_sigma      = 0.30

    # RSF / Weibull — PFS prediction
    try:
        pfs_median, ci_low, ci_high, survival_curve = predict_rsf_pfs(X)
    except Exception as exc:
        log.warning("treatment_opt: RSF/PFS prediction failed (%s) — using defaults", exc)
        pfs_median, ci_low, ci_high, survival_curve = 24.0, 8.0, 52.0, []

    # Determine trigger — RANO takes precedence for brain tumours (Task 4);
    # fall back to RECIST for non-brain primary sites.
    rano_raw = memory.get(WorkingMemory.RANO)
    rano_response: str | None = None
    if rano_raw:
        rr = rano_raw if isinstance(rano_raw, dict) else rano_raw.model_dump()
        rano_response = rr.get("response")

    recist_raw = memory.get(WorkingMemory.RECIST)
    recist_response = "NE"
    if recist_raw:
        r = recist_raw if isinstance(recist_raw, dict) else recist_raw.model_dump()
        recist_response = r.get("response", "NE")

    # Effective response for triggering: RANO when present, else RECIST.
    effective_response = rano_response or recist_response
    criterion_label = "RANO" if rano_response else "RECIST"

    optimization_triggered = False
    trigger_reason: str | None = None

    if effective_response == "PD":
        optimization_triggered = True
        trigger_reason = f"{criterion_label}=PD (progressive disease)"
    elif recist_sigma > SMBO_SIGMA_TRIGGER:
        optimization_triggered = True
        trigger_reason = f"GP uncertainty σ={recist_sigma:.3f} > {SMBO_SIGMA_TRIGGER}"
    elif effective_response == "SD" and recist_delta_pred > RECIST_DELTA_SD_TRIGGER:
        optimization_triggered = True
        trigger_reason = (
            f"{criterion_label}=SD with predicted delta "
            f"{recist_delta_pred:.1f}% > {RECIST_DELTA_SD_TRIGGER}%"
        )

    pred = PredictionResult(
        recist_delta_pred=round(recist_delta_pred, 2),
        recist_sigma=round(recist_sigma, 4),
        pfs_median_weeks=round(pfs_median, 1),
        pfs_ci_low=round(ci_low, 1),
        pfs_ci_high=round(ci_high, 1),
        survival_curve=survival_curve[:20],
        optimization_triggered=optimization_triggered,
        trigger_reason=trigger_reason,
    )
    memory.set(WorkingMemory.PREDICTION, pred)

    log.info("treatment_opt: 4b complete — RECIST_delta=%.1f%%  σ=%.3f  PFS=%.1fw  "
             "triggered=%s (%s)",
             recist_delta_pred, recist_sigma, pfs_median,
             optimization_triggered, trigger_reason or "n/a")
    return {"ok": True, "optimization_triggered": optimization_triggered,
            "trigger_reason": trigger_reason}


# ── Sub-step 4c: run_smbo_optimization ────────────────────────────────────────

@register("run_smbo_optimization")
def run_smbo_optimization(memory: WorkingMemory, **_) -> dict:
    """Run 60-iteration Batched SMBO.  Skips when optimization_triggered=False."""
    pred_raw = memory.get(WorkingMemory.PREDICTION)
    if pred_raw is None:
        return {"ok": False, "error": "predict_recist_pfs must run first"}

    pred = pred_raw if isinstance(pred_raw, PredictionResult) \
        else PredictionResult.model_validate(pred_raw)

    if not pred.optimization_triggered:
        log.info("treatment_opt: 4c — SMBO skipped (not triggered)")
        skipped = OptimizationResult(triggered=False, n_iterations=0)
        memory.set(WorkingMemory.OPTIMIZATION, skipped)
        return {"ok": True, "skipped": True}

    import numpy as np

    vec_raw = memory.get(WorkingMemory.PATIENT_STATE)
    vec = vec_raw if isinstance(vec_raw, PatientStateVector) \
        else PatientStateVector.model_validate(vec_raw)
    patient_vec = np.array(vec.normalized, dtype=np.float64)

    from ..utils.smbo_engine import run_batched_smbo

    cancer_type = vec.cancer_type or "glioblastoma"
    # Pass patient_id so SMBO writes plots directly into outputs/<pid>/plots/.
    # Phase 5.2 / Module 5 — pass patient PGx profile so SMBO scoring
    # multiplies CTCAE severity and PFS benefit by CYP-phenotype factors.
    result = run_batched_smbo(
        patient_vec, cancer_type=cancer_type, patient_id=memory.patient_id,
        pgx_profile=getattr(vec, "pgx_profile", None),
    )

    # No-op on v2 runs (plots already in plots/); safety net for legacy staging dir.
    _relocate_plots(result, memory.out_dir)

    memory.set(WorkingMemory.OPTIMIZATION, result)
    log.info("treatment_opt: 4c complete — %d iters  top=%s  sigma=%.4f",
             result.n_iterations,
             result.final_best.primary_drug if result.final_best else "n/a",
             result.sigma_at_convergence or 0.0)
    return {"ok": True, "n_iterations": result.n_iterations,
            "top_drug": result.final_best.primary_drug if result.final_best else None}


def _relocate_plots(result: OptimizationResult, out_dir: Path) -> None:
    """Move SMBO plot files from the staging dir to the patient output dir."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for attr in ("convergence_plot_path", "landscape_plot_path"):
        src_str = getattr(result, attr, None)
        if not src_str:
            continue
        src = Path(src_str)
        if src.exists():
            dst = plots_dir / src.name
            try:
                src.rename(dst)
                setattr(result, attr, str(dst))
            except Exception:
                import shutil
                try:
                    shutil.copy2(str(src), str(dst))
                    setattr(result, attr, str(dst))
                except Exception:
                    pass


# ── Sub-step 4d: explain_with_shap ────────────────────────────────────────────

@register("explain_with_shap")
def explain_with_shap(memory: WorkingMemory, **_) -> dict:
    """SHAP KernelExplainer on the best SMBO candidate.  Skips if not triggered."""
    pred_raw = memory.get(WorkingMemory.PREDICTION)
    pred = pred_raw if isinstance(pred_raw, PredictionResult) \
        else PredictionResult.model_validate(pred_raw) if pred_raw else None

    if pred is None or not pred.optimization_triggered:
        log.info("treatment_opt: 4d — SHAP skipped (not triggered)")
        memory.set(WorkingMemory.SHAP, ShapResult(base_value=24.0, top_5_drivers=[]))
        return {"ok": True, "skipped": True}

    import numpy as np

    opt_raw = memory.get(WorkingMemory.OPTIMIZATION)
    opt = opt_raw if isinstance(opt_raw, OptimizationResult) \
        else OptimizationResult.model_validate(opt_raw) if opt_raw else None

    if opt is None or not opt.final_best:
        memory.set(WorkingMemory.SHAP, ShapResult(base_value=24.0, top_5_drivers=[]))
        return {"ok": True, "skipped": True, "reason": "no SMBO result"}

    vec_raw = memory.get(WorkingMemory.PATIENT_STATE)
    vec = vec_raw if isinstance(vec_raw, PatientStateVector) \
        else PatientStateVector.model_validate(vec_raw)

    from ..utils.patient_state import FEATURE_NAMES
    from ..utils.survival_models import _generate_statistical_cohort, _weibull_pfs_fallback, predict_rsf_pfs

    # Build SHAP background: 200 random patients from synthetic cohort
    X_bg, _, _ = _generate_statistical_cohort(200, rng=np.random.default_rng(7))

    # Build RSF scorer wrapper (scalar PFS output)
    def _rsf_scorer(X_inp: np.ndarray) -> np.ndarray:
        scores = []
        for row in X_inp:
            try:
                pfs, _, _, _ = predict_rsf_pfs(row.reshape(1, -1))
            except Exception:
                pfs, _, _, _ = _weibull_pfs_fallback(row)
            scores.append(pfs)
        return np.array(scores)

    # Baseline PFS (population average for this cancer type)
    base_scores = _rsf_scorer(X_bg)
    base_value = float(np.mean(base_scores))

    # Feature vector for the best candidate
    patient_vec = np.array(vec.normalized, dtype=np.float64)

    try:
        import shap  # type: ignore

        # PERF FIX (P001 run analysis): the previous build passed the full
        # 200-sample background pool directly to KernelExplainer, which
        # produced 200 × 100 = 20,000 model evaluations and blew the cold-run
        # SHAP step out to ~30 minutes. shap.kmeans summarises the pool to K
        # weighted centroids, dropping that to K × nsamples = 20 × 100 model
        # calls (~2 minutes on the same hardware) with negligible quality
        # impact for ranking the top-5 drivers.
        try:
            bg_summary = shap.kmeans(X_bg, 20)
        except Exception:
            # shap.kmeans needs at least one finite row per cluster; fall
            # through to a uniform sample if clustering fails.
            bg_summary = X_bg[: min(20, len(X_bg))]
        explainer = shap.KernelExplainer(_rsf_scorer, bg_summary, silent=True)
        shap_values = explainer.shap_values(
            patient_vec.reshape(1, -1), nsamples=100, silent=True
        )
        sv = np.array(shap_values).flatten()

    except ImportError:
        log.warning("treatment_opt: shap not installed — using gradient approximation")
        # Finite-difference approximation as fallback
        eps = 0.05
        sv = np.zeros(len(patient_vec))
        base_pfs = float(_rsf_scorer(patient_vec.reshape(1, -1))[0])
        for j in range(len(patient_vec)):
            perturbed = patient_vec.copy()
            perturbed[j] = min(1.0, patient_vec[j] + eps)
            sv[j] = (float(_rsf_scorer(perturbed.reshape(1, -1))[0]) - base_pfs) / eps

    except Exception as exc:
        log.warning("treatment_opt: SHAP computation failed: %s", exc)
        sv = np.zeros(len(patient_vec))
        base_value = 24.0

    # Top 5 drivers by absolute SHAP value

    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[-5:][::-1]
    top_drivers = [
        ShapDriver(
            feature=FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}",
            shap_value=round(float(sv[i]), 3),
            direction="+" if sv[i] >= 0 else "-",
        )
        for i in top_idx
        if abs_sv[i] > 0
    ]

    # Waterfall plot
    waterfall_path: str | None = None
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        from ..config import patient_out_dir
        plots_dir = patient_out_dir(memory.patient_id, "plots")
        fig, ax = plt.subplots(figsize=(9, 5))

        feat_labels = [d.feature.replace("_", "\n") for d in top_drivers]
        shap_vals   = [d.shap_value for d in top_drivers]
        colors      = ["#4CAF50" if v >= 0 else "#F44336" for v in shap_vals]

        ax.barh(range(len(feat_labels)), shap_vals, color=colors, edgecolor="white")
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (weeks impact on PFS)")
        ax.set_title(f"SHAP Explainability — Top {len(feat_labels)} Drivers\n"
                     f"Base PFS: {base_value:.1f} weeks")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        waterfall_path = str(plots_dir / "shap_waterfall.png")
        fig.savefig(waterfall_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        log.info("treatment_opt: SHAP waterfall saved to %s", waterfall_path)
    except Exception as exc:
        log.warning("treatment_opt: SHAP waterfall plot failed: %s", exc)

    all_shap = {
        (FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"): round(float(sv[i]), 4)
        for i in range(len(sv))
    }

    # LLM narrative (qwen3:14b) — best-effort; empty string on failure.
    narrative = ""
    try:
        from ..utils.llm_enrichment import shap_narrative
        narrative = shap_narrative(
            [{"feature": d.feature,
              "shap_value": d.shap_value,
              "direction": d.direction}
             for d in top_drivers],
            base_pfs_weeks=base_value,
            cancer_type=(vec.cancer_type if vec else "glioblastoma"),
        )
        if narrative:
            log.info("treatment_opt: SHAP narrative generated (%d chars)",
                     len(narrative))
    except Exception as exc:
        log.warning("treatment_opt: SHAP narrative generation failed: %s", exc)

    shap_result = ShapResult(
        base_value=round(base_value, 2),
        top_5_drivers=top_drivers,
        waterfall_plot_path=waterfall_path,
        all_shap_values=all_shap,
        narrative=narrative,
    )
    memory.set(WorkingMemory.SHAP, shap_result)

    log.info("treatment_opt: 4d complete — base_pfs=%.1fw  top_driver=%s  waterfall=%s",
             base_value,
             top_drivers[0].feature if top_drivers else "n/a",
             waterfall_path or "none")
    return {"ok": True, "base_value": base_value,
            "top_driver": top_drivers[0].feature if top_drivers else None}


# ── Sub-step 4e: review_proposal_mdt ─────────────────────────────────────────

@register("review_proposal_mdt")
def review_proposal_mdt(memory: WorkingMemory, **_) -> dict:
    """Qwen3:14b MDT board reviewer — APPROVE / MODIFY / REJECT / SKIP."""
    pred_raw = memory.get(WorkingMemory.PREDICTION)
    pred = pred_raw if isinstance(pred_raw, PredictionResult) \
        else PredictionResult.model_validate(pred_raw) if pred_raw else None

    # If optimisation never triggered, write SKIP and return early
    if pred is None or not pred.optimization_triggered:
        skip = TreatmentProposal(
            decision="SKIP",
            reason="patient_responding_adequately",
            clinical_narrative=(
                "RECIST response indicates adequate treatment response; "
                "SMBO optimisation was not triggered."
            ),
        )
        memory.set(WorkingMemory.TREATMENT_PROPOSAL, skip)
        log.info("treatment_opt: 4e — SKIP (optimisation not triggered)")
        return {"ok": True, "decision": "SKIP"}

    # Build context for LLM
    ctx = _build_mdt_context(memory, pred)

    # ── Multi-agent MDT debate (Task 7) ───────────────────────────────────────
    try:
        transcript, proposal = _run_mdt_debate(ctx)
    except Exception as exc:
        log.warning("treatment_opt: MDT debate failed (%s) — degraded proposal", exc)
        transcript = []
        proposal = TreatmentProposal(
            decision="MODIFY",
            reason=f"mdt_debate_error: {str(exc)[:80]}",
            mdt_discussion_required=True,
            clinical_narrative="MDT debate unavailable; manual review required.",
        )

    # Attach transcript + consensus on the proposal
    if transcript:
        round2 = [t for t in transcript if t.round == 2 and t.persona in _MDT_PERSONAS]
        if round2:
            agree = sum(1 for t in round2 if t.agreement_with_proposal == "agree")
            consensus = round(agree / len(round2), 2)
        else:
            consensus = None
        object.__setattr__(proposal, "debate_transcript", transcript)
        object.__setattr__(proposal, "consensus_score", consensus)
        if consensus is not None and consensus < 0.5:
            object.__setattr__(proposal, "mdt_discussion_required", True)
        log.info(
            "treatment_opt: 4e debate complete — turns=%d  consensus=%s",
            len(transcript), consensus,
        )

    # Override: force mdt_discussion_required for safety conditions
    urgency = memory.get(WorkingMemory.URGENCY)
    urgency_score = 0
    if urgency:
        u = urgency if isinstance(urgency, dict) else urgency.model_dump()
        urgency_score = int(u.get("score", 0))
    if proposal.decision in ("MODIFY", "REJECT") or urgency_score >= 4:
        # Use object_setattr to update frozen-safe Pydantic field
        object.__setattr__(proposal, "mdt_discussion_required", True)

    # ── Second-pass qwen3:14b self-critique (audit the proposal for safety) ──
    try:
        from ..utils.llm_enrichment import mdt_self_critique
        audit = mdt_self_critique(
            proposal.model_dump(),
            patient_context=ctx if isinstance(ctx, dict) else {},
        )
        object.__setattr__(proposal, "audit_concerns",       audit["concerns"])
        object.__setattr__(proposal, "audit_recommendation", audit["recommendation"])
        object.__setattr__(proposal, "audit_rationale",      audit["rationale"])
        # If the audit flags a non-APPROVE recommendation, force MDT review.
        if audit["recommendation"] not in ("APPROVE", ""):
            object.__setattr__(proposal, "mdt_discussion_required", True)
        log.info("treatment_opt: 4e self-critique: rec=%s concerns=%d",
                 audit["recommendation"], len(audit["concerns"]))
    except Exception as exc:
        log.warning("treatment_opt: 4e self-critique failed (non-blocking): %s", exc)

    memory.set(WorkingMemory.TREATMENT_PROPOSAL, proposal)
    log.info("treatment_opt: 4e complete — decision=%s  mdt_required=%s  regimen=%s",
             proposal.decision, proposal.mdt_discussion_required, proposal.proposed_regimen)

    # Best-effort Google Workspace notifications
    _fire_phase4_notifications(memory, proposal, urgency_score)

    return {
        "ok": True,
        "decision": proposal.decision,
        "mdt_discussion_required": proposal.mdt_discussion_required,
        "proposed_regimen": proposal.proposed_regimen,
    }


def _build_mdt_context(memory: WorkingMemory, pred: PredictionResult) -> dict:
    """Assemble the compact JSON context dict sent to Qwen3 for MDT review."""
    pid = memory.patient_id
    ctx: dict[str, Any] = {"patient_id": pid}

    # Patient state (human-readable raw values, not normalized)
    vec_raw = memory.get(WorkingMemory.PATIENT_STATE)
    if vec_raw:
        vec = vec_raw if isinstance(vec_raw, PatientStateVector) \
            else PatientStateVector.model_validate(vec_raw)
        ctx["patient_state"] = {
            "cancer_type":    vec.cancer_type,
            "ecog_ps":        vec.ecog_ps_score,
            "egfr":           vec.egfr_ml_per_min,
            "ldh":            vec.ldh_u_per_l,
            "hemoglobin":     vec.hemoglobin_g_per_dl,
            "albumin":        vec.albumin_g_per_dl,
            "prior_lines":    int(vec.total_prior_lines),
            "dose_reduction": bool(vec.dose_reduction_flag),
            "cycles_done":    vec.treatment_cycles_completed,
        }

    # Current medications (pre-extracted in 4a)
    ctx["current_medications"] = memory._store.get("pre_extracted_meds", [])

    # Patient record demographics
    rec_raw = memory.get(WorkingMemory.RECORD)
    if rec_raw:
        r = rec_raw if isinstance(rec_raw, dict) else rec_raw.model_dump()
        ctx["diagnosis"]   = r.get("diagnosis")
        ctx["patient_age"] = r.get("age")
        ctx["patient_sex"] = r.get("sex")

    # RECIST context
    recist_raw = memory.get(WorkingMemory.RECIST)
    if recist_raw:
        r2 = recist_raw if isinstance(recist_raw, dict) else recist_raw.model_dump()
        ctx["recist"] = {
            "response":          r2.get("response"),
            "pct_change":        r2.get("pct_change"),
            "new_lesion":        r2.get("new_lesion_detected", False),
        }

    # Urgency
    urg_raw = memory.get(WorkingMemory.URGENCY)
    if urg_raw:
        u = urg_raw if isinstance(urg_raw, dict) else urg_raw.model_dump()
        ctx["urgency"] = {"score": u.get("score"), "level": u.get("level"),
                          "drivers": u.get("drivers", [])}

    # Survival prediction
    ctx["survival_prediction"] = {
        "recist_delta_pred": pred.recist_delta_pred,
        "recist_sigma":      pred.recist_sigma,
        "pfs_median_weeks":  pred.pfs_median_weeks,
        "pfs_ci":            f"{pred.pfs_ci_low:.1f}–{pred.pfs_ci_high:.1f}w",
        "trigger_reason":    pred.trigger_reason,
    }

    # SMBO top-3 candidates
    opt_raw = memory.get(WorkingMemory.OPTIMIZATION)
    if opt_raw:
        opt = opt_raw if isinstance(opt_raw, OptimizationResult) \
            else OptimizationResult.model_validate(opt_raw)
        ctx["smbo_top_3"] = [
            {
                "rank":         c.rank,
                "primary_drug": c.primary_drug,
                "combo_drug":   c.combo_drug,
                "dose_fraction": c.dose_fraction,
                "cycle_weeks":  c.cycle_weeks,
                "route":        c.route,
                "predicted_pfs": f"{c.predicted_pfs_weeks}w",
                "rag_penalty":  c.rag_penalty,
            }
            for c in opt.top_3_candidates
        ]

    # SHAP top-5 drivers
    shap_raw = memory.get(WorkingMemory.SHAP)
    if shap_raw:
        shap = shap_raw if isinstance(shap_raw, ShapResult) \
            else ShapResult.model_validate(shap_raw)
        ctx["shap_top_5"] = [
            {"feature": d.feature, "impact_weeks": d.shap_value, "direction": d.direction}
            for d in shap.top_5_drivers
        ]
        ctx["shap_base_pfs"] = shap.base_value

    # Wearable narrative (gemma4:e4b — plain-English trend summary)
    wn = memory._store.get("wearable_narrative")
    if wn:
        ctx["wearable_trend"] = wn

    # Task 8 — Clinical trial matches (compressed projection, ≤2 KB)
    tm_raw = memory.get(WorkingMemory.TRIAL_MATCHES)
    if tm_raw:
        tm = tm_raw if isinstance(tm_raw, dict) else tm_raw.model_dump()
        top = tm.get("top_matches", []) or []
        ctx["clinical_trials"] = {
            "triggered":       tm.get("triggered", False),
            "trigger_reason":  tm.get("trigger_reason", ""),
            "n_searched":      tm.get("n_searched", 0),
            "top_matches": [
                {
                    "nct_id":              t.get("nct_id"),
                    "title":               (t.get("title") or "")[:160],
                    "phase":               t.get("phase"),
                    "interventions":       (t.get("interventions") or [])[:4],
                    "eligibility_summary": (t.get("eligibility_summary") or "")[:500],
                    "match_score":         t.get("match_score"),
                    "match_reasoning":     t.get("match_reasoning"),
                }
                for t in top[:3]
            ],
        }

    # Phase 5.7 / Extra B — FAERS adverse-event signals (gated, ≤2 KB)
    fa_raw = memory.get(WorkingMemory.FAERS_SIGNALS)
    if fa_raw:
        fa = fa_raw if isinstance(fa_raw, dict) else fa_raw.model_dump()
        reports = fa.get("reports", []) or []
        ctx["faers_signals"] = {
            "n_candidates_queried": fa.get("n_candidates_queried", 0),
            "unavailable":          fa.get("faers_unavailable", False),
            "reports": [
                {
                    "rank":            r.get("rank"),
                    "primary_drug":    r.get("primary_drug"),
                    "combo_drug":      r.get("combo_drug"),
                    "triggered_by":    r.get("triggered_by", []),
                    "n_total_reports": r.get("n_total_reports", 0),
                    "fallback_used":   bool(r.get("fallback_used", False)),
                    "note":            r.get("note"),
                    "top_reactions": [
                        {
                            "reaction":    s.get("reaction"),
                            "n_reports":   s.get("n_reports"),
                            "serious_pct": s.get("serious_pct"),
                            "outcomes":    (s.get("outcomes") or [])[:3],
                        }
                        for s in (r.get("signals") or [])[:5]
                    ],
                }
                for r in reports[:3]
            ],
        }

    # Phase 5.5 / Module 3 — PubMed Evidence (compressed projection, ≤3 KB)
    # Each abstract is hard-truncated to 600 chars upstream so the
    # qwen3:14b context window stays comfortable.
    pm_raw = memory.get(WorkingMemory.PUBMED_EVIDENCE)
    if pm_raw:
        pm = pm_raw if isinstance(pm_raw, dict) else pm_raw.model_dump()
        results = pm.get("results", []) or []
        ctx["pubmed_literature"] = {
            "query_terms":  pm.get("query_terms", []),
            "n_results":    pm.get("n_results", 0),
            "unavailable":  pm.get("pubmed_unavailable", False),
            "results": [
                {
                    "pmid":      r.get("pmid"),
                    "title":     (r.get("title") or "")[:240],
                    "abstract":  (r.get("abstract") or "")[:600],
                    "journal":   (r.get("journal") or "")[:120],
                    "pubdate":   r.get("pubdate"),
                    "authors":   (r.get("authors") or [])[:3],
                    "publication_types": (r.get("publication_types") or [])[:3],
                }
                for r in results[:5]
            ],
        }

    return ctx


# ── MDT multi-agent debate (Task 7) ──────────────────────────────────────────

def _persona_turn_call(persona: str, round_num: int, ctx: dict,
                       prior_turns: list[MDTPersonaTurn] | None = None) -> MDTPersonaTurn:
    """Blocking single LLM call that returns one MDTPersonaTurn.

    Runs inside a thread (via asyncio.to_thread) during parallel rounds.
    """
    system = _load_persona_prompt(persona)
    user_payload: dict[str, Any] = {
        "round": round_num,
        "patient_context": ctx,
    }
    if prior_turns:
        user_payload["prior_round_transcript"] = [t.model_dump() for t in prior_turns]

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": json.dumps(user_payload, indent=2, default=str)},
    ]
    # First parse raw JSON so we can coerce the Literal field; then validate.
    try:
        from ..llm import _DEFAULT_OPTIONS, _client  # type: ignore
        resp = _client().chat(
            model=MODEL_THINKING, messages=messages,
            format="json", options=_DEFAULT_OPTIONS,
        )
        raw = json.loads(resp["message"]["content"] or "{}")
        if not isinstance(raw, dict):
            raw = {}
        agree_raw = str(raw.get("agreement_with_proposal", "modify")).strip().lower()
        if agree_raw.startswith("agree"):
            agree = "agree"
        elif agree_raw.startswith("disagree") or agree_raw in ("reject", "no"):
            agree = "disagree"
        else:
            agree = "modify"
        turn = MDTPersonaTurn(
            persona=persona,  # type: ignore[arg-type]
            round=round_num,
            statement=str(raw.get("statement") or "")[:1200],
            concerns=list(raw.get("concerns", []))[:6],
            agreement_with_proposal=agree,  # type: ignore[arg-type]
        )
        return turn
    except Exception as exc:
        log.warning("treatment_opt: persona %s round %d failed: %s",
                    persona, round_num, exc)
        return MDTPersonaTurn(
            persona=persona,
            round=round_num,
            statement=f"[LLM error — {str(exc)[:80]}]",
            concerns=["persona_call_failed"],
            agreement_with_proposal="modify",
        )


async def _debate_async(ctx: dict) -> list[MDTPersonaTurn]:
    """Run round-1 (4 parallel) then round-2 (4 parallel) persona calls."""
    import asyncio

    # Round 1 — fully parallel, no prior turns
    round1_tasks = [
        asyncio.to_thread(_persona_turn_call, p, 1, ctx, None)
        for p in _MDT_PERSONAS
    ]
    round1: list[MDTPersonaTurn] = list(await asyncio.gather(*round1_tasks))

    # Round 2 — each persona sees the OTHER three round-1 statements
    def _others(p: str) -> list[MDTPersonaTurn]:
        return [t for t in round1 if t.persona != p]

    round2_tasks = [
        asyncio.to_thread(_persona_turn_call, p, 2, ctx, _others(p))
        for p in _MDT_PERSONAS
    ]
    round2: list[MDTPersonaTurn] = list(await asyncio.gather(*round2_tasks))

    return round1 + round2


def _run_mdt_debate(ctx: dict) -> tuple[list[MDTPersonaTurn], TreatmentProposal]:
    """Run the 4-persona debate + chair synthesis.

    Returns (transcript, TreatmentProposal). Transcript includes 4+4 persona
    turns from rounds 1 and 2, plus the chair turn (round 3).
    """
    import asyncio

    # Run rounds 1+2 concurrently. If we're already inside a running event
    # loop (e.g. FastAPI async handler called us directly), asyncio.run()
    # would raise — detect explicitly and run in a dedicated thread.
    already_in_loop = False
    try:
        asyncio.get_running_loop()
        already_in_loop = True
    except RuntimeError:
        already_in_loop = False

    if not already_in_loop:
        transcript = asyncio.run(_debate_async(ctx))
    else:
        import threading

        result_box: dict[str, Any] = {}

        def _worker():
            loop = asyncio.new_event_loop()
            try:
                result_box["t"] = loop.run_until_complete(_debate_async(ctx))
            finally:
                loop.close()

        th = threading.Thread(target=_worker)
        th.start()
        th.join()
        transcript = result_box.get("t", [])

    # ── Chair synthesis (sequential) ─────────────────────────────────────────
    chair_system = _load_persona_prompt("chair")
    chair_payload = {
        "patient_context": ctx,
        "debate_transcript": [t.model_dump() for t in transcript],
    }
    messages = [
        {"role": "system", "content": chair_system},
        {"role": "user",   "content": json.dumps(chair_payload, indent=2, default=str)},
    ]

    chair_raw: dict = {}
    try:
        from ..llm import _DEFAULT_OPTIONS, _client  # type: ignore
        resp = _client().chat(
            model=MODEL_THINKING, messages=messages,
            format="json", options=_DEFAULT_OPTIONS,
        )
        chair_raw = json.loads(resp["message"]["content"] or "{}")
        if not isinstance(chair_raw, dict):
            chair_raw = {}
    except Exception as exc:
        log.warning("treatment_opt: chair synthesis call failed: %s", exc)
        chair_raw = {}

    # Coerce LLM string into the Literal["agree","modify","disagree"] set to
    # avoid a ValidationError that would nuke the whole debate result.
    agree_raw = str(chair_raw.get("agreement_with_proposal", "modify")).strip().lower()
    if agree_raw.startswith("agree"):
        agree = "agree"
    elif agree_raw.startswith("disagree") or agree_raw in ("reject", "no"):
        agree = "disagree"
    else:
        agree = "modify"

    chair_turn = MDTPersonaTurn(
        persona="chair",
        round=3,
        statement=str(chair_raw.get("statement")
                      or chair_raw.get("clinical_narrative")
                      or chair_raw.get("reason")
                      or "Chair synthesis unavailable.")[:1200],
        concerns=list(chair_raw.get("concerns", []))[:6],
        agreement_with_proposal=agree,  # type: ignore[arg-type]
    )
    transcript.append(chair_turn)

    decision = str(chair_raw.get("decision", "MODIFY")).upper().strip()
    if decision not in ("APPROVE", "MODIFY", "REJECT", "SKIP"):
        log.warning("treatment_opt: chair returned invalid decision=%r — defaulting to MODIFY",
                    decision)
        decision = "MODIFY"

    proposal = TreatmentProposal(
        decision=decision,  # type: ignore[arg-type]
        reason=str(chair_raw.get("reason") or "chair_synthesis")[:500],
        proposed_regimen=str(chair_raw.get("proposed_regimen") or "unchanged")[:300],
        modifications=list(chair_raw.get("modifications", []))[:10],
        clinical_narrative=str(chair_raw.get("clinical_narrative")
                               or chair_raw.get("statement") or "")[:2000],
        mdt_discussion_required=bool(chair_raw.get("mdt_discussion_required", False)),
    )
    return transcript, proposal


# ── Google Workspace notifications ────────────────────────────────────────────

def _fire_phase4_notifications(
    memory: WorkingMemory,
    proposal: TreatmentProposal,
    urgency_score: int,
) -> None:
    """Best-effort Gmail / Calendar / Drive hooks after MDT review."""
    notifs: dict[str, Any] = {}
    pid = memory.patient_id

    # Gmail — MDT alert
    if GMAIL_ENABLED and (proposal.mdt_discussion_required or proposal.decision == "REJECT"):
        try:
            from ..integrations.gmail_client import send_mdt_alert
            subject_prefix = "[URGENT - TREATMENT REJECTION] " if proposal.decision == "REJECT" else ""
            result = send_mdt_alert(pid, proposal, subject_prefix=subject_prefix)
            notifs["mdt_email"] = result
            log.info("treatment_opt: MDT alert email sent for %s", pid)
        except Exception as exc:
            notifs["mdt_email"] = {"ok": False, "error": str(exc)[:120]}
            log.warning("treatment_opt: MDT email failed: %s", exc)

    # Calendar — MDT meeting
    if CALENDAR_ENABLED and proposal.mdt_discussion_required:
        try:
            from ..integrations.calendar_client import create_mdt_meeting
            result = create_mdt_meeting(pid, proposal)
            notifs["mdt_calendar"] = result
            log.info("treatment_opt: MDT calendar event created for %s", pid)
        except Exception as exc:
            notifs["mdt_calendar"] = {"ok": False, "error": str(exc)[:120]}
            log.warning("treatment_opt: MDT calendar failed: %s", exc)

    # Drive — upload Phase 4 reports
    if DRIVE_ENABLED:
        try:
            from ..integrations.drive_client import upload_phase4_reports
            result = upload_phase4_reports(pid, memory.out_dir)
            notifs["drive"] = result
            log.info("treatment_opt: Phase 4 reports uploaded to Drive for %s", pid)
        except Exception as exc:
            notifs["drive"] = {"ok": False, "error": str(exc)[:120]}
            log.warning("treatment_opt: Drive upload failed: %s", exc)

    if notifs:
        # Use memory.set so the bag routes to outputs/<pid>/notifications/phase4.json
        memory.set("notifications_phase4", notifs)
