"""LLM-powered narrative enrichment helpers.

Gives qwen3:14b and gemma4:e4b productive work beyond raw structured
extraction: plain-English narratives, self-critique of MDT decisions,
and executive summaries. Every function is best-effort — if the LLM is
unavailable the pipeline still completes with a deterministic stub.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import MODEL_PRIMARY, MODEL_THINKING, MODEL_VISION
from ..llm import chat

log = logging.getLogger(__name__)


# ─────────────────────────── SHAP narrative ────────────────────────────
def shap_narrative(
    shap_top_drivers: list[dict[str, Any]],
    base_pfs_weeks: float,
    cancer_type: str,
    *,
    model: str = MODEL_THINKING,
) -> str:
    """Turn a list of SHAP drivers into a 2–3 sentence clinical narrative.

    Input shape (per driver): {"feature": str, "shap_value": float, "direction": "+"|"-"}.
    Returns a single paragraph (empty string on any LLM failure — caller should
    fall back to the raw bullet list).
    """
    if not shap_top_drivers:
        return ""
    try:
        drivers_compact = [
            {"feature": d.get("feature"),
             "impact_weeks": round(float(d.get("shap_value", 0.0)), 2),
             "direction": d.get("direction", "+")}
            for d in shap_top_drivers
        ]
        prompt = (
            "You are a senior neuro-oncology clinical data scientist. "
            f"A survival model (Random Survival Forest + Gaussian Process) predicts "
            f"progression-free survival for a {cancer_type} patient. "
            f"The population baseline PFS for this cancer type is {base_pfs_weeks:.1f} weeks. "
            "The SHAP explainer identified these top drivers of the individual prediction "
            "(positive = extends PFS, negative = reduces PFS):\n"
            f"{json.dumps(drivers_compact, indent=2)}\n\n"
            "Write a concise 2–3 sentence clinical narrative explaining what these "
            "drivers imply for the individual patient. Use plain medical English; no "
            "markdown, no bullet lists, no preamble. Do NOT invent features not in the list."
        )
        text = chat([{"role": "user", "content": prompt}], model=model,
                    options={"temperature": 0.1, "num_predict": 250})
        return (text or "").strip()
    except Exception as exc:
        log.warning("llm_enrichment.shap_narrative failed: %s (%s)",
                    type(exc).__name__, str(exc)[:120])
        return ""


# ─────────────────────────── Wearable narrative ────────────────────────
def wearable_narrative(
    wearable_daily: list[dict[str, Any]] | None,
    *,
    model: str = MODEL_VISION,  # gemma4:e4b — fast text summarisation
) -> str:
    """Summarise the trend of wearable vitals over the ingested window.

    `wearable_daily` is the raw list from phase4/wearable_data.json; a typical
    entry carries {date, steps, resting_hr, sleep_hours, hrv_ms}. Returns an
    empty string if the LLM call fails or data is empty/malformed.
    """
    if not wearable_daily or len(wearable_daily) < 3:
        return ""
    try:
        # Keep prompt compact — first 30 days maximum is plenty for trend spotting.
        sample = wearable_daily[:30]
        prompt = (
            "You are summarising a neuro-oncology patient's wearable data. "
            "Report ONE short paragraph (≤60 words) describing the trend of "
            "daily steps, resting HR, sleep hours and HRV over the window — "
            "specifically flag any decline or concerning pattern. Plain English, "
            "no markdown, no bullets.\n\n"
            f"DATA (chronological):\n{json.dumps(sample, default=str)}"
        )
        text = chat([{"role": "user", "content": prompt}], model=model,
                    options={"temperature": 0.1, "num_predict": 180})
        return (text or "").strip()
    except Exception as exc:
        log.warning("llm_enrichment.wearable_narrative failed: %s (%s)",
                    type(exc).__name__, str(exc)[:120])
        return ""


# ─────────────────────────── Executive summary ─────────────────────────
def executive_summary(
    pipeline_snapshot: dict[str, Any],
    *,
    model: str = MODEL_PRIMARY,
) -> str:
    """Generate a ≤200-word plain-English executive summary for a GP / clinician.

    `pipeline_snapshot` should include top-level keys like patient_id, diagnosis,
    recist_response, urgency_level, proposed_regimen, pfs_median_weeks,
    mdt_discussion_required, and a short list of SHAP drivers.
    """
    try:
        prompt = (
            "You are a senior neuro-oncology consultant writing an executive "
            "summary for the patient's general practitioner. "
            "Write 150–200 words in plain English — no headings, no bullet lists, "
            "no markdown, no preamble. Cover (in order): current disease status "
            "(RECIST), urgency, key predicted outcome, any recommended treatment "
            "change with the clinical reasoning, and whether MDT board review is "
            "needed. End with a single action line starting with 'Action: …'.\n\n"
            f"PATIENT_SNAPSHOT_JSON:\n{json.dumps(pipeline_snapshot, default=str)}"
        )
        text = chat([{"role": "user", "content": prompt}], model=model,
                    options={"temperature": 0.1, "num_predict": 350})
        return (text or "").strip()
    except Exception as exc:
        log.warning("llm_enrichment.executive_summary failed: %s (%s)",
                    type(exc).__name__, str(exc)[:120])
        return ""


# ─────────────────────────── MDT self-critique ─────────────────────────
def mdt_self_critique(
    proposal: dict[str, Any],
    patient_context: dict[str, Any],
    *,
    model: str = MODEL_THINKING,
) -> dict[str, Any]:
    """Second-pass review of an MDT treatment proposal — catches internal
    contradictions and missing safety checks.

    Returns a dict: {
        "concerns":        list[str]   — specific issues found (empty = clean),
        "recommendation":  str         — APPROVE | MODIFY | REJECT | DEFER_MDT,
        "rationale":       str         — one-paragraph explanation,
    }
    Errors in the LLM call produce a permissive default that does NOT override
    the original decision.
    """
    default = {"concerns": [], "recommendation": "APPROVE", "rationale": ""}
    try:
        prompt = (
            "You are an independent senior neuro-oncologist auditing another "
            "oncologist's MDT treatment proposal for internal consistency and "
            "safety. You have the original proposal and the patient context. "
            "Return ONLY a JSON object with exactly these keys:\n"
            "  concerns        : array of specific short strings (empty if none),\n"
            "  recommendation  : one of APPROVE | MODIFY | REJECT | DEFER_MDT,\n"
            "  rationale       : one-paragraph plain-English explanation (≤80 words).\n\n"
            "Flag things like: contradiction between decision and RECIST response, "
            "missing contraindication check, dose not reduced despite low eGFR, "
            "regimen includes a drug with a current-medication major interaction.\n\n"
            f"PROPOSAL:\n{json.dumps(proposal, default=str)}\n\n"
            f"PATIENT_CONTEXT:\n{json.dumps(patient_context, default=str)}"
        )
        text = chat([{"role": "user", "content": prompt}], model=model,
                    options={"temperature": 0.0, "num_predict": 400,
                             "format": "json"})
        parsed = json.loads(text or "{}")
        if isinstance(parsed, dict):
            # Coerce the three expected keys; fall back to defaults per-key.
            return {
                "concerns":       list(parsed.get("concerns") or []),
                "recommendation": str(parsed.get("recommendation") or "APPROVE"),
                "rationale":      str(parsed.get("rationale") or ""),
            }
    except Exception as exc:
        log.warning("llm_enrichment.mdt_self_critique failed: %s (%s)",
                    type(exc).__name__, str(exc)[:120])
    return default
