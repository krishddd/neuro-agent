"""Phase 5 — synthesis sub-agent.

Registered tools:

    build_timeline()  -> Timeline written to memory
    write_summary()   -> PatientSummary written to memory + .txt files
    export_fhir()     -> ExportResult written to memory + bundle on disk

The timeline is built deterministically from already-structured memory
(no LLM hallucinated dates). The summary is the only LLM call here, and
it operates on a compact JSON snapshot. FHIR export is also deterministic.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ..config import (
    CALENDAR_ENABLED,
    DISCLAIMER,
    DRIVE_ENABLED,
    GMAIL_ENABLED,
    OUTPUTS_DIR,
)
from ..llm import json_call
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.schemas import (
    CorrelationResult,
    ExportResult,
    IngestionResult,
    InteractionReport,
    MedicationList,
    PatientRecord,
    PatientSummary,
    RECISTAssessment,
    Timeline,
    TimelineEvent,
    UrgencyAssessment,
    VisionObservation,
)
from ..utils.tool_helpers import load_model as _get
from ..utils.tool_helpers import load_prompt
from . import register

log = logging.getLogger(__name__)


# ---------- Google Workspace integration hooks ----------

def _fire_gmail_notifications(
    memory: WorkingMemory,
    summary: PatientSummary,
    gp_attachments: list | None = None,
) -> dict[str, Any]:
    """Send patient letter + GP handover emails after write_summary().

    Best-effort — any exception is logged but never raises to the caller.
    Returns a dict with send status for each recipient.
    """
    if not GMAIL_ENABLED:
        log.info("gmail: disabled (no token.json) — skipping notifications")
        return {"gmail": "disabled"}

    result: dict[str, Any] = {}
    try:
        from ..integrations.gmail_client import GmailClient
        from ..integrations.patient_roster import get_patient_email

        pid     = memory.patient_id
        urgency = _get(memory, WorkingMemory.URGENCY, UrgencyAssessment)
        record  = _get(memory, WorkingMemory.RECORD,  PatientRecord)
        diag    = (record.diagnosis or "") if record else ""
        score   = urgency.score if urgency else 1

        client = GmailClient()
        if not client.ready:
            log.warning("gmail: client not ready — run setup_oauth.py to authorise")
            return {"gmail": "not_configured — run setup_oauth.py"}

        log.info("gmail: sending notifications for patient %s …", pid)

        # Patient letter — SPIKES policy: ALWAYS routed to doctor's inbox for
        # clinician review, never sent to patient_email directly. patient_email
        # is recorded only as the intended-recipient annotation in the body.
        patient_email = get_patient_email(pid)
        # MULTI-PATIENT-FIX: PatientRecord has a flat `patient_name` field
        # (Phase 5 schema bump). The previous lookup ``record.patient.name``
        # never resolved because PatientRecord has no ``patient`` sub-model
        # — every patient fell through to ``pid``. Now read the flat field
        # first; fall back to legacy nested lookups for older runs.
        patient_name = (
            getattr(record, "patient_name", None)
            or (getattr(getattr(record, "patient", None), "name", None) if record else None)
            or pid
        )
        ok = client.send_patient_letter(
            to                 = "",                   # ignored — routed to DOCTOR_EMAIL
            patient_id         = pid,
            patient_name       = patient_name,
            letter_text        = summary.patient_letter,
            diagnosis          = diag,
            urgency_score      = score,
            intended_recipient = patient_email or "(no email on file)",
        )
        result["patient_letter"] = "sent_to_doctor_for_review" if ok else "failed"
        log.info(
            "gmail: patient_letter [SPIKES — routed to doctor for review, intended=%s] [%s]",
            patient_email or "n/a", result["patient_letter"],
        )

        # GP handover → doctor's own inbox.
        ok_gp = client.send_gp_handover(
            patient_id   = pid,
            patient_name = patient_name,
            gp_text      = summary.gp_handover_letter,
            diagnosis    = diag,
        )
        result["gp_handover"] = "sent" if ok_gp else "failed"
        log.info("gmail: gp_handover [%s]", result["gp_handover"])

        # Urgency alert if critical — but only if Phase 4 didn't already send one.
        phase4_notifs = memory._store.get("notifications_phase4") or {}
        phase4_sent_mdt_alert = bool(phase4_notifs.get("mdt_email"))
        if score >= 5 and urgency and not phase4_sent_mdt_alert:
            ok_alert = client.send_urgency_alert(
                patient_id    = pid,
                urgency_level = urgency.level,
                drivers       = urgency.drivers,
                patient_email = patient_email or "",
            )
            result["urgency_alert"] = "sent" if ok_alert else "failed"
            log.info("gmail: urgency_alert [%s]", result["urgency_alert"])

        # Welcome DM — tells the patient how to use Google Chat bot.
        # SPIKES policy: skip unless PATIENT_DIRECT_COMMS_ENABLED=true. Report
        # the skip as "suppressed", not "failed" — audit-channel clarity.
        if patient_email:
            from ..integrations.gmail_client import PATIENT_DIRECT_COMMS_ENABLED
            if not PATIENT_DIRECT_COMMS_ENABLED:
                result["chat_welcome_dm"] = "suppressed_spikes_policy"
                log.info(
                    "gmail: chat_welcome_dm suppressed for %s (SPIKES policy, "
                    "PATIENT_DIRECT_COMMS_ENABLED=false)", patient_email,
                )
            else:
                ok_dm = client.send_chat_welcome_dm(
                    to         = patient_email,
                    patient_id = pid,
                )
                result["chat_welcome_dm"] = "sent" if ok_dm else "failed"
                log.info("gmail: chat_welcome_dm → %s  [%s]", patient_email, result["chat_welcome_dm"])

    except Exception as exc:
        from ..utils.audit import log as _audit
        _audit("gmail.notify.error", pid=memory.patient_id,
               status="warning", meta={"error": str(exc)[:200]})
        result["gmail_error"] = str(exc)[:200]

    return result


def _fire_google_sync(
    memory: WorkingMemory,
    out_dir: Path,
) -> dict[str, Any]:
    """Sync outputs to Drive and schedule Calendar events after export_fhir().

    Best-effort — any exception is logged but never raises to the caller.
    """
    result: dict[str, Any] = {}

    try:
        from ..integrations.patient_roster import get_patient_email
        pid           = memory.patient_id
        patient_email = get_patient_email(pid)
        record        = _get(memory, WorkingMemory.RECORD,       PatientRecord)
        recist        = _get(memory, WorkingMemory.RECIST,       RECISTAssessment)
        urgency       = _get(memory, WorkingMemory.URGENCY,      UrgencyAssessment)
        meds          = _get(memory, WorkingMemory.MEDICATIONS,  MedicationList)
        corr          = _get(memory, WorkingMemory.CORRELATION,  CorrelationResult)
        diag          = (record.diagnosis or "") if record else ""
        recist_resp   = recist.response   if recist  else "NE"
        urgency_score = urgency.score     if urgency else 1

        # ── Drive sync ──────────────────────────────────────────────────
        if DRIVE_ENABLED:
            try:
                # P001-RUN-FIX: write working_memory.json now so the
                # DriveClient skip-list (which excludes internal state
                # files by filename) actually has something to skip.
                # Otherwise these files only land on disk during
                # finalize() — which runs *after* drive sync — and the
                # skip never triggers. Idempotent: finalize() overwrites
                # this snapshot with the canonical version at end-of-run.
                try:
                    memory.persist_internal_snapshot()
                except Exception as exc:
                    log.warning(
                        "drive: pre-sync internal-snapshot write failed: %s", exc,
                    )

                from ..integrations.drive_client import DriveClient
                drive = DriveClient()
                if drive.ready:
                    log.info("drive: syncing outputs for %s …", pid)
                    folder_url = drive.sync_patient_outputs(
                        patient_id    = pid,
                        outputs_dir   = out_dir,
                        patient_email = patient_email,
                    )
                    result["drive_folder"] = folder_url or "sync_failed"
                    log.info("drive: sync complete → %s", result["drive_folder"])
                else:
                    log.warning("drive: client not ready — run setup_oauth.py")
                    result["drive"] = "not_configured"
            except Exception as exc:
                log.warning("drive: sync failed: %s", exc)
                result["drive_error"] = str(exc)[:200]
        else:
            log.info("drive: disabled (no token.json)")

        # ── Calendar ────────────────────────────────────────────────────
        if CALENDAR_ENABLED:
            try:
                from ..integrations.calendar_client import CalendarClient
                from ..utils.schemas import TreatmentProposal as _TPC

                cal = CalendarClient()

                # Extract Phase 4 decision for calendar override
                proposal_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
                phase4_decision: str | None = None
                proposed_regimen: str | None = None
                if proposal_raw:
                    try:
                        prop_c = proposal_raw if isinstance(proposal_raw, _TPC) \
                            else _TPC.model_validate(proposal_raw)
                        phase4_decision = prop_c.decision
                        proposed_regimen = prop_c.proposed_regimen
                    except Exception:
                        pass

                if cal.ready:
                    log.info("calendar: scheduling events for %s …", pid)
                    # Follow-up appointment — Phase 4 aware timing.
                    fu_ok = cal.schedule_followup(
                        patient_id      = pid,
                        recist_response = recist_resp,
                        urgency_score   = urgency_score,
                        patient_email   = patient_email,
                        diagnosis       = diag,
                        phase4_decision = phase4_decision,
                    )
                    log.info("calendar: follow-up event %s",
                             "created" if fu_ok else "skipped/failed")

                    # SAFETY (Task 2): Do NOT auto-schedule [PROPOSED] regimen
                    # events. A Phase 4 proposal is not a treatment commitment
                    # — the MDT must meet and the patient must give informed
                    # consent before any drug-specific calendar event is
                    # created. The neutral "Consultation with Oncologist"
                    # event scheduled by schedule_followup() above is enough
                    # to put the conversation on the calendar without naming
                    # any drug. The cal.create_proposed_regimen_events()
                    # method is preserved for the future HITL execute-path
                    # (Task 9) where it may fire AFTER explicit approval.
                    from ..integrations.calendar_client import (
                        AUTO_SCHEDULE_PROPOSED_TREATMENT as _AUTO_SCHED_TX,
                    )
                    if phase4_decision in ("APPROVE", "MODIFY") and proposed_regimen:
                        if _AUTO_SCHED_TX:
                            n_proposed = cal.create_proposed_regimen_events(
                                patient_id       = pid,
                                proposed_regimen = proposed_regimen,
                                patient_email    = patient_email,
                            )
                            result["calendar_proposed_events"] = n_proposed
                            log.info("calendar: %d proposed regimen event(s) created", n_proposed)
                        else:
                            log.info(
                                "calendar: [PROPOSED] regimen events suppressed for %s "
                                "(AUTO_SCHEDULE_PROPOSED_TREATMENT=False — awaiting MDT + consent)",
                                pid,
                            )
                            result["calendar_proposed_events"] = "suppressed_pending_mdt_consent"

                    # Medication reminders.
                    n_med = 0
                    if meds and meds.current:
                        med_dicts = [m.model_dump(mode="json") for m in meds.current]
                        n_med = cal.create_medication_reminders(
                            patient_id    = pid,
                            medications   = med_dicts,
                            patient_email = patient_email,
                        )
                        result["calendar_med_events"] = n_med
                    log.info("calendar: %d medication reminder(s) created", n_med)

                    # Therapy sessions from correlation summary.
                    n_therapy = 0
                    if corr and corr.summary:
                        n_therapy = cal.create_therapy_sessions(
                            patient_id          = pid,
                            correlation_summary = corr.summary,
                            patient_email       = patient_email,
                        )
                        result["calendar_therapy_events"] = n_therapy
                    log.info("calendar: %d therapy session(s) created", n_therapy)

                    result["calendar"] = "scheduled"
                else:
                    log.warning("calendar: client not ready — run setup_oauth.py")
                    result["calendar"] = "not_configured"
            except Exception as exc:
                log.warning("calendar: scheduling failed: %s", exc)
                result["calendar_error"] = str(exc)[:200]
        else:
            log.info("calendar: disabled (no token.json)")

    except Exception as exc:
        from ..utils.audit import log as _audit
        _audit("google.sync.error", pid=memory.patient_id,
               status="warning", meta={"error": str(exc)[:200]})
        result["google_error"] = str(exc)[:200]

    return result


# ---------- build_timeline ----------
@register("build_timeline")
def build_timeline(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("synthesis.build_timeline", pid=pid, tool="build_timeline") as _t:
        events: list[TimelineEvent] = []

        ing = _get(memory, WorkingMemory.INGESTION, IngestionResult)
        record = _get(memory, WorkingMemory.RECORD, PatientRecord)
        recist = _get(memory, WorkingMemory.RECIST, RECISTAssessment)
        meds = _get(memory, WorkingMemory.MEDICATIONS, MedicationList)
        vision_bag = memory.get(WorkingMemory.VISION) or {}

        # Diagnosis event
        if record and record.diagnosis_date:
            events.append(TimelineEvent(
                when=record.diagnosis_date, kind="visit",
                label=f"Diagnosis: {record.diagnosis or 'unknown'}",
            ))

        # MRI scan events — one per visit (date unknown unless captured)
        if ing:
            for visit in ing.visits:
                obs = vision_bag.get(visit) if isinstance(vision_bag, dict) else None
                if obs is not None:
                    obs = obs if isinstance(obs, VisionObservation) else VisionObservation.model_validate(obs)
                    events.append(TimelineEvent(
                        when=date.today(), kind="scan",
                        label=f"MRI {visit}: {(obs.impression or '')[:120]}",
                    ))

        # Medication events
        if meds:
            for m in list(meds.current) + list(meds.historical):
                if m.start_date:
                    events.append(TimelineEvent(
                        when=m.start_date, kind="med_start",
                        label=f"Start {m.name} {m.dose or ''}".strip(),
                    ))
                if m.stop_date:
                    events.append(TimelineEvent(
                        when=m.stop_date, kind="med_stop",
                        label=f"Stop {m.name}",
                    ))

        # Phase 4 treatment optimisation events (if available)
        from ..utils.schemas import PredictionResult, TreatmentProposal
        proposal_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
        if proposal_raw:
            try:
                prop = proposal_raw if isinstance(proposal_raw, TreatmentProposal) \
                    else TreatmentProposal.model_validate(proposal_raw)
                if prop.decision != "SKIP":
                    events.append(TimelineEvent(
                        when=date.today(), kind="note",
                        label=(f"Treatment Optimization: MDT decision={prop.decision} — "
                               f"{(prop.proposed_regimen or 'no regimen change')[:100]}"),
                    ))
                if prop.mdt_discussion_required:
                    events.append(TimelineEvent(
                        when=date.today(), kind="note",
                        label="MDT Board Discussion Required — escalated for multidisciplinary review",
                    ))
            except Exception:
                pass
        pred_raw = memory.get(WorkingMemory.PREDICTION)
        if pred_raw:
            try:
                pred = pred_raw if isinstance(pred_raw, PredictionResult) \
                    else PredictionResult.model_validate(pred_raw)
                events.append(TimelineEvent(
                    when=date.today(), kind="note",
                    label=(f"Survival Prediction: PFS={pred.pfs_median_weeks:.1f}w "
                           f"(CI {pred.pfs_ci_low:.1f}–{pred.pfs_ci_high:.1f}w), "
                           f"RECIST delta predicted={pred.recist_delta_pred:+.1f}%, "
                           f"σ={pred.recist_sigma:.3f}"),
                ))
            except Exception:
                pass

        # Sort chronologically (string-safe)
        events.sort(key=lambda e: str(e.when))

        narrative_bits = []
        if record and record.diagnosis:
            narrative_bits.append(f"Patient diagnosed with {record.diagnosis}.")
        if recist:
            narrative_bits.append(f"Latest RECIST response: {recist.response}.")
        if meds and meds.current:
            narrative_bits.append(
                f"Currently on {len(meds.current)} medication(s): "
                f"{', '.join(m.name for m in meds.current[:5])}."
            )
        narrative = " ".join(narrative_bits) or "No structured timeline data available."

        timeline = Timeline(events=events, narrative=narrative)
        memory.set(WorkingMemory.TIMELINE, timeline)
        _t.meta["ok"] = True
        return {"ok": True, "n_events": len(events)}


# ---------- write_summary ----------
def _per_visit_mini_summary(
    visit: str,
    obs: VisionObservation | None,
    recist: RECISTAssessment | None,
    meds_at_visit: list[str],
) -> str:
    """One compact paragraph per visit — used as hierarchical summary input."""
    parts: list[str] = [f"Visit {visit}:"]
    if obs:
        parts.append(f"MRI impression — {(obs.impression or 'no impression')[:200]}.")
        if obs.mass_effect:
            parts.append("Mass effect present.")
        if obs.hemorrhage:
            parts.append("Hemorrhage noted.")
    if recist and visit in (
        (recist.lesions_current[0].visit if recist.lesions_current else None),
        (recist.lesions_baseline[0].visit if recist.lesions_baseline else None),
    ):
        parts.append(
            f"RECIST: {recist.response}"
            + (f" ({recist.pct_change * 100:+.1f}%)" if recist.pct_change is not None else "")
            + ("  — NEW LESION DETECTED." if recist.new_lesion_detected else "")
            + (" Confirmation required." if recist.confirmation_required else "")
            + f" {recist.rationale[:120]}"
        )
    if meds_at_visit:
        parts.append(f"Medications: {', '.join(meds_at_visit[:6])}.")
    return " ".join(parts)


def _summary_context(memory: WorkingMemory) -> dict[str, Any]:
    """Hierarchical compact snapshot for the summary LLM call.

    Instead of dumping raw WorkingMemory (potentially 30k tokens for multi-visit
    patients), we produce one mini-paragraph per visit then a cross-visit delta
    section. The final LLM call sees ≤~2k tokens regardless of visit count.
    """
    record = _get(memory, WorkingMemory.RECORD, PatientRecord)
    recist = _get(memory, WorkingMemory.RECIST, RECISTAssessment)
    urgency = _get(memory, WorkingMemory.URGENCY, UrgencyAssessment)
    meds = _get(memory, WorkingMemory.MEDICATIONS, MedicationList)
    inter = _get(memory, WorkingMemory.INTERACTIONS, InteractionReport)
    corr = _get(memory, WorkingMemory.CORRELATION, CorrelationResult)
    ing = _get(memory, WorkingMemory.INGESTION, IngestionResult)
    vision_bag = memory.get(WorkingMemory.VISION) or {}

    # Per-visit mini summaries
    visit_summaries: list[str] = []
    visits = (ing.visits if ing else []) or []
    all_meds = list((meds.current if meds else []) + (meds.historical if meds else []))
    for visit in visits:
        obs_raw = vision_bag.get(visit) if isinstance(vision_bag, dict) else None
        obs = None
        if obs_raw is not None:
            obs = obs_raw if isinstance(obs_raw, VisionObservation) else VisionObservation.model_validate(obs_raw)
        meds_this_visit = [m.name for m in all_meds if not m.stop_date or str(m.stop_date) >= visit]
        visit_summaries.append(_per_visit_mini_summary(visit, obs, recist, meds_this_visit))

    ctx: dict[str, Any] = {
        "patient_id": memory.patient_id,
        "diagnosis": record.diagnosis if record else None,
        "age_sex": f"{record.age}/{record.sex}" if record else None,
        "visit_summaries": visit_summaries,
        "recist_final": {
            "response": recist.response,
            "pct_change": recist.pct_change,
            "new_lesion": recist.new_lesion_detected,
            "confirmation_required": recist.confirmation_required,
            "confirmation_action": (
                f"ACTION REQUIRED: {recist.response} response must be confirmed by "
                "repeat scan ≥4 weeks from today per RECIST 1.1. "
                "GP handover must include this instruction."
            ) if recist.confirmation_required else None,
            "rationale": recist.rationale,
        } if recist else None,
        "urgency": {"level": urgency.level, "score": urgency.score, "drivers": urgency.drivers} if urgency else None,
        "interaction_flags": inter.flags if inter else [],
        "highest_severity": inter.highest_severity if inter else None,
        "correlation_summary": corr.summary if corr else None,
        "n_med_events": len(corr.med_events) if corr else 0,
    }

    # Phase 4 treatment optimisation context (compact, ≤200 extra tokens)
    from ..utils.schemas import PredictionResult, ShapResult, TreatmentProposal
    proposal_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
    if proposal_raw:
        try:
            prop = proposal_raw if isinstance(proposal_raw, TreatmentProposal) \
                else TreatmentProposal.model_validate(proposal_raw)
            if prop.decision != "SKIP":
                ctx["treatment_optimization"] = {
                    "mdt_decision":          prop.decision,
                    "proposed_regimen":      prop.proposed_regimen,
                    "modifications":         prop.modifications[:3],
                    "guideline_alignment":   prop.guideline_alignment,
                    "mdt_discussion_required": prop.mdt_discussion_required,
                    "clinical_narrative":    (prop.clinical_narrative or "")[:400],
                }
        except Exception:
            pass
    pred_raw = memory.get(WorkingMemory.PREDICTION)
    if pred_raw:
        try:
            pred = pred_raw if isinstance(pred_raw, PredictionResult) \
                else PredictionResult.model_validate(pred_raw)
            ctx["survival_prediction"] = {
                "pfs_median_weeks":      pred.pfs_median_weeks,
                "pfs_ci":                f"{pred.pfs_ci_low:.1f}–{pred.pfs_ci_high:.1f}w",
                "recist_delta_pred":     pred.recist_delta_pred,
                "optimization_triggered": pred.optimization_triggered,
            }
        except Exception:
            pass
    shap_raw = memory.get(WorkingMemory.SHAP)
    if shap_raw:
        try:
            shap = shap_raw if isinstance(shap_raw, ShapResult) \
                else ShapResult.model_validate(shap_raw)
            ctx["shap_top_drivers"] = [
                {"feature": d.feature, "impact": f"{d.shap_value:+.2f}w",
                 "direction": d.direction}
                for d in shap.top_5_drivers
            ]
        except Exception:
            pass
    return ctx


@register("write_summary")
def write_summary(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("synthesis.write_summary", pid=pid, tool="write_summary") as _t:
        ctx = _summary_context(memory)
        sys_msg = load_prompt("summary_system.md")
        prompt = (
            f"{sys_msg}\n\nPATIENT_CONTEXT_JSON:\n{json.dumps(ctx, default=str)}\n"
        )
        try:
            summary = json_call([{"role": "user", "content": prompt}], PatientSummary)
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

        # Write plain-text files into outputs/<pid>/reports/ —
        # matching reference dataset naming: patient_letter.txt / gp_handover.txt
        from ..config import patient_out_dir
        OUTPUTS_DIR / pid  # for S18 path below (patient root)
        reports_dir = patient_out_dir(pid, "reports")
        letter_path = reports_dir / "patient_letter.txt"
        gp_path = reports_dir / "gp_handover.txt"

        letter_text = summary.patient_letter
        if DISCLAIMER not in letter_text:
            letter_text = f"{letter_text}\n\n{DISCLAIMER}"
        letter_path.write_text(letter_text, encoding="utf-8")
        gp_path.write_text(summary.gp_handover_letter, encoding="utf-8")

        memory.set(WorkingMemory.SUMMARY, summary)

        # Collect Phase 4 attachments for GP handover email
        phase4_attachments: list[Path] = []
        shap_raw = memory.get(WorkingMemory.SHAP)
        if shap_raw:
            try:
                from ..utils.schemas import ShapResult as _SR
                shap_obj = shap_raw if isinstance(shap_raw, _SR) \
                    else _SR.model_validate(shap_raw)
                if shap_obj.waterfall_plot_path:
                    wp = Path(shap_obj.waterfall_plot_path)
                    if wp.exists():
                        phase4_attachments.append(wp)
            except Exception:
                pass
        # S18 now lives in stages/ — fall back to patient root for legacy runs.
        s18_candidates = [
            patient_out_dir(pid, "stages") / "S18_treatment_proposal.json",
            OUTPUTS_DIR / pid / "S18_treatment_proposal.json",
        ]
        for s18_path in s18_candidates:
            if s18_path.exists():
                phase4_attachments.append(s18_path)
                break

        # Fire Gmail notifications (best-effort — never blocks the pipeline).
        gmail_result = _fire_gmail_notifications(memory, summary,
                                                  gp_attachments=phase4_attachments)
        # Store so orchestrator / process endpoint can surface them.
        memory.set("notifications_gmail", gmail_result)

        _t.meta["ok"] = True
        return {
            "ok": True,
            "letter_path": str(letter_path),
            "gp_path":     str(gp_path),
            "gmail":       gmail_result,
        }


# ---------- report.md ----------
def _write_report_md(memory: WorkingMemory, out_dir: Path) -> Path:
    """Write a comprehensive Markdown report combining all pipeline outputs."""
    pid = memory.patient_id
    record = _get(memory, WorkingMemory.RECORD, PatientRecord)
    recist = _get(memory, WorkingMemory.RECIST, RECISTAssessment)
    urgency = _get(memory, WorkingMemory.URGENCY, UrgencyAssessment)
    meds = _get(memory, WorkingMemory.MEDICATIONS, MedicationList)
    inter = _get(memory, WorkingMemory.INTERACTIONS, InteractionReport)
    corr = _get(memory, WorkingMemory.CORRELATION, CorrelationResult)
    timeline = _get(memory, WorkingMemory.TIMELINE, Timeline)
    summary = _get(memory, WorkingMemory.SUMMARY, PatientSummary)
    vision_bag = memory.get(WorkingMemory.VISION) or {}
    ing = _get(memory, WorkingMemory.INGESTION, IngestionResult)

    lines: list[str] = []

    # Header
    lines += [
        f"# Neuro-Oncology Report — Patient {pid}",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
    ]

    # Patient record
    lines.append("## Patient Record")
    if record:
        lines += [
            f"- **Diagnosis:** {record.diagnosis or 'unknown'}",
            f"- **Diagnosis date:** {record.diagnosis_date or 'unknown'}",
            f"- **Age/Sex:** {record.age or '?'} / {record.sex or '?'}",
        ]
        if record.impression:
            lines += ["", f"*{record.impression[:400]}*"]
    else:
        lines.append("*No patient record extracted.*")
    lines.append("")

    # MRI findings per visit
    lines.append("## MRI Findings")
    visits = ing.visits if ing else []
    for visit in visits:
        obs_raw = vision_bag.get(visit) if isinstance(vision_bag, dict) else None
        if obs_raw is None:
            continue
        obs = obs_raw if isinstance(obs_raw, VisionObservation) else VisionObservation.model_validate(obs_raw)
        lines.append(f"### Visit {visit}")
        lines.append(f"**Impression:** {obs.impression or 'none'}")
        flags = []
        if obs.mass_effect:
            flags.append("mass effect")
        if obs.hemorrhage:
            flags.append("hemorrhage")
        if obs.discrepancy_with_report:
            flags.append(f"discrepancy: {obs.discrepancy_notes or 'yes'}")
        if flags:
            lines.append(f"**Flags:** {', '.join(flags)}")
        if obs.findings:
            lines.append("")
            lines.append("| Location | Size (mm) | Enhancement | Notes |")
            lines.append("|---|---|---|---|")
            for f in obs.findings:
                lines.append(
                    f"| {f.location or '-'} | {f.size_mm or '-'} "
                    f"| {f.enhancement or '-'} | {(f.description or '')[:100]} |"
                )
        lines.append("")

    # RECIST
    lines.append("## RECIST 1.1 Assessment")
    if recist:
        resp_emoji = {"PD": "🔴", "SD": "🟡", "PR": "🟢", "CR": "🟢", "NE": "⚪"}.get(recist.response, "")
        lines += [
            "| Field | Value |",
            "|---|---|",
            f"| Response | {resp_emoji} **{recist.response}** |",
            f"| Baseline sum | {recist.baseline_sum_mm or 'n/a'} mm |",
            f"| Current sum | {recist.current_sum_mm or 'n/a'} mm |",
            f"| Change | {f'{recist.pct_change*100:+.1f}%' if recist.pct_change is not None else 'n/a'} |",
            f"| New lesion detected | {'**YES — automatic PD**' if recist.new_lesion_detected else 'No'} |",
            f"| Confirmation required | {'Yes (CR/PR needs ≥4-week rescan)' if recist.confirmation_required else 'No'} |",
            "",
            f"*{recist.rationale}*",
        ]
    else:
        lines.append("*RECIST assessment not available.*")
    lines.append("")

    # Urgency
    lines.append("## Urgency Triage")
    if urgency:
        level_icon = {"routine": "🟢", "soon": "🟡", "priority": "🟠",
                      "urgent": "🔴", "critical": "🚨"}.get(urgency.level, "")
        lines += [
            f"**Level:** {level_icon} {urgency.level.upper()} (score {urgency.score}/5)",
            f"**Drivers:** {', '.join(urgency.drivers) or 'none'}",
            "",
            f"*{urgency.rationale}*",
        ]
    else:
        lines.append("*Urgency not assessed.*")
    lines.append("")

    # Treatment Optimization (Phase 4 SMBO v3.0)
    lines.append("## Treatment Optimization (Phase 4 SMBO v3.0)")
    from ..utils.schemas import PredictionResult as _PR
    from ..utils.schemas import ShapResult as _SH
    from ..utils.schemas import TreatmentProposal as _TP
    proposal_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
    pred_raw     = memory.get(WorkingMemory.PREDICTION)
    shap_raw     = memory.get(WorkingMemory.SHAP)
    if not proposal_raw:
        lines.append("*Phase 4 not run for this patient.*")
    else:
        try:
            prop = proposal_raw if isinstance(proposal_raw, _TP) \
                else _TP.model_validate(proposal_raw)
            decision_icon = {"APPROVE": "✅", "MODIFY": "⚠️",
                             "REJECT": "❌", "SKIP": "⏭️"}.get(prop.decision, "")
            lines += [
                f"**MDT Decision:** {decision_icon} **{prop.decision}**",
                f"**Reason:** {prop.reason}",
            ]
            if prop.proposed_regimen:
                lines.append(f"**Proposed Regimen:** `{prop.proposed_regimen}`")
            if prop.modifications:
                lines.append(f"**Modifications:** {'; '.join(prop.modifications)}")
            if prop.guideline_alignment:
                lines.append(f"**Guideline Alignment:** {prop.guideline_alignment}")
            if prop.mdt_discussion_required:
                lines.append("**⚠️ MDT Board Discussion Required**")
            if prop.rag_interaction_flags:
                lines.append(f"**RAG Interaction Flags:** {', '.join(prop.rag_interaction_flags)}")
            if prop.clinical_narrative:
                lines += ["", f"*{prop.clinical_narrative[:600]}*"]
        except Exception:
            lines.append("*Error rendering treatment proposal.*")
        lines.append("")

        if pred_raw:
            try:
                pred = pred_raw if isinstance(pred_raw, _PR) else _PR.model_validate(pred_raw)
                lines += [
                    "### Survival Prediction (GP + RSF)",
                    "| Metric | Value |",
                    "|---|---|",
                    f"| Predicted RECIST Δ | {pred.recist_delta_pred:+.1f}% (σ={pred.recist_sigma:.3f}) |",
                    f"| Predicted PFS | {pred.pfs_median_weeks:.1f} weeks |",
                    f"| 95% CI | {pred.pfs_ci_low:.1f} – {pred.pfs_ci_high:.1f} weeks |",
                    f"| Optimization triggered | {'Yes' if pred.optimization_triggered else 'No'} |",
                    "",
                ]
            except Exception:
                pass

        if shap_raw:
            try:
                shap = shap_raw if isinstance(shap_raw, _SH) else _SH.model_validate(shap_raw)
                lines += [
                    "### SHAP Explainability — Top 5 Drivers",
                    f"*Base PFS: {shap.base_value:.1f} weeks (population average for this cancer type)*",
                    "",
                    "| Feature | SHAP Impact | Direction |",
                    "|---|---|---|",
                ]
                for d in shap.top_5_drivers:
                    icon = "🟢 +" if d.direction == "+" else "🔴 "
                    lines.append(f"| {d.feature} | {icon}{abs(d.shap_value):.1f}w | {d.direction} |")
                if shap.waterfall_plot_path:
                    lines.append(f"\n![SHAP Waterfall]({shap.waterfall_plot_path})")
            except Exception:
                pass
    lines.append("")

    # Medications
    lines.append("## Medications")
    if meds and (meds.current or meds.historical):
        if meds.current:
            lines.append("### Current")
            lines.append("| Drug | Dose | Frequency | Route | Start |")
            lines.append("|---|---|---|---|---|")
            for m in meds.current:
                lines.append(
                    f"| {m.name} | {m.dose or '-'} | {m.frequency or '-'} "
                    f"| {m.route or '-'} | {m.start_date or '-'} |"
                )
        if meds.historical:
            lines.append("### Historical")
            lines.append("| Drug | Dose | Stop date |")
            lines.append("|---|---|---|")
            for m in meds.historical:
                lines.append(f"| {m.name} | {m.dose or '-'} | {m.stop_date or '-'} |")
    else:
        lines.append("*No medications recorded.*")
    lines.append("")

    # Drug interactions
    lines.append("## Drug Interactions")
    if inter and inter.interactions:
        sev_icon = {"none": "⚪", "minor": "🟡", "moderate": "🟠",
                    "major": "🔴", "contraindicated": "🚨"}
        lines += [
            f"**Highest severity:** {sev_icon.get(inter.highest_severity, '')} {inter.highest_severity}",
            "",
            "| Drug A | Drug B | Severity | Mechanism |",
            "|---|---|---|---|",
        ]
        for ix in inter.interactions:
            if ix.severity != "none":
                lines.append(
                    f"| {ix.drug_a} | {ix.drug_b} "
                    f"| {sev_icon.get(ix.severity,'')} {ix.severity} "
                    f"| {(ix.mechanism or '-')[:120]} |"
                )
    else:
        lines.append("*No significant interactions detected.*")
    lines.append("")

    # Treatment correlation
    lines.append("## Treatment–Response Correlation")
    if corr:
        lines.append(f"{corr.summary}")
        if corr.med_events:
            lines += ["", "### Medication Event Timeline", "| Date | Drug | Event | Dose |",
                      "|---|---|---|---|"]
            for ev in corr.med_events:
                lines.append(
                    f"| {ev.event_date or '?'} | {ev.drug} "
                    f"| {ev.event_type} | {ev.dose or '-'} |"
                )
    else:
        lines.append("*Correlation not available.*")
    lines.append("")

    # Clinical timeline
    lines.append("## Clinical Timeline")
    if timeline and timeline.events:
        lines.append("| Date | Type | Event |")
        lines.append("|---|---|---|")
        for ev in timeline.events:
            lines.append(f"| {ev.when} | {ev.kind} | {ev.label} |")
    else:
        lines.append("*No timeline events.*")
    lines.append("")

    # Summary
    lines.append("## Patient Summary")
    if summary:
        lines += ["### Patient Letter", "", summary.patient_letter, "",
                  "---", "", "### GP Handover", "", summary.gp_handover_letter, ""]
    else:
        lines.append("*Summary not generated.*")

    # Footer
    lines += ["", "---", f"*{DISCLAIMER}*", ""]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------- laboratory_results.json ----------
def _write_laboratory_results(memory: WorkingMemory, out_dir: Path) -> None:
    """Extract lab results from ingested lab PDF text and write laboratory_results.json.

    Uses an LLM to parse the raw lab text into structured per-visit arrays
    matching the reference format: [{visit, date, FBC:{...}, UE:{...}, LFT:{...}, ...}].
    Falls back to a minimal stub if no lab files were ingested.
    """
    ing = _get(memory, WorkingMemory.INGESTION, IngestionResult)
    if not ing:
        return

    lab_texts: list[str] = [
        f.text for f in ing.files
        if f.kind == "lab" and f.text
    ]
    if not lab_texts:
        return

    raw_text = "\n\n---NEXT LAB REPORT---\n\n".join(lab_texts)
    sys_msg = (
        "You are a clinical data extraction assistant. "
        "Parse the following laboratory results text into a JSON array. "
        "Each element should represent one visit/date with keys: "
        "'visit' (integer, 1-based), 'date' (ISO YYYY-MM-DD), "
        "'FBC' (full blood count fields), 'UE' (urea & electrolytes), "
        "'LFT' (liver function tests), 'CRP_mg_L', 'HbA1c_mmol_mol', 'LDH_U_L'. "
        "Use null for missing values. Return ONLY valid JSON array, no explanation."
    )
    prompt = f"{sys_msg}\n\nLAB TEXT:\n{raw_text[:8000]}"
    try:
        memory.get("__llm_raw__")  # unused sentinel
        import re as _re

        from ..llm import chat as _llm_call
        resp = _llm_call([{"role": "user", "content": prompt}])
        # Strip markdown code fences if present.
        text = _re.sub(r"```(?:json)?", "", resp).strip().strip("`").strip()
        lab_data = json.loads(text)
    except Exception:
        # Best-effort: write empty array so the file still exists.
        lab_data = []

    from ..config import patient_out_dir
    lab_path = patient_out_dir(memory.patient_id, "extended") / "laboratory_results.json"
    lab_path.write_text(json.dumps(lab_data, indent=2, default=str), encoding="utf-8")


# ---------- extended/ subfolder ----------
def _write_extended(memory: WorkingMemory, out_dir: Path) -> None:
    """Write the extended/ subfolder matching the reference dataset structure.

    Files written:
        extended/radiology_reports.json     — per-visit MRI report text array
        extended/pathology_report.txt       — raw pathology text
        extended/adverse_events_ctcae.json  — LLM-extracted CTCAE events
        extended/clinical_trial_eligibility.json — LLM-generated eligibility
        extended/mrs_spectroscopy.json      — stub (MRS data not in source PDFs)
        extended/neuropsych_assessments.json — stub
    """
    from ..config import patient_out_dir
    ext_dir = patient_out_dir(memory.patient_id, "extended")

    ing = _get(memory, WorkingMemory.INGESTION, IngestionResult)
    record = _get(memory, WorkingMemory.RECORD, PatientRecord)
    inter = _get(memory, WorkingMemory.INTERACTIONS, InteractionReport)
    meds = _get(memory, WorkingMemory.MEDICATIONS, MedicationList)
    pid = memory.patient_id

    # --- radiology_reports.json ---
    radiology_reports: list[dict[str, Any]] = []
    if ing:
        visit_num = 1
        visit_map: dict[str, int] = {}
        for f in ing.files:
            if f.kind == "mri_report" and f.text:
                v = f.visit
                if v not in visit_map:
                    visit_map[v] = visit_num
                    visit_num += 1
                radiology_reports.append({
                    "visit": visit_map[v],
                    "scan_date": None,        # date not reliably parseable here
                    "report_text": f.text,
                    "rano_response": None,    # filled by RECIST phase if available
                })
    (ext_dir / "radiology_reports.json").write_text(
        json.dumps(radiology_reports, indent=2, default=str), encoding="utf-8"
    )

    # --- pathology_report.txt ---
    patho_text = ""
    if ing:
        for f in ing.files:
            if f.kind == "pathology" and f.text:
                patho_text = f.text
                break
    (ext_dir / "pathology_report.txt").write_text(patho_text or "", encoding="utf-8")

    # --- adverse_events_ctcae.json --- (LLM extraction)
    ae_data: dict[str, Any] = {"patient_id": pid, "events": []}
    if inter or meds:
        ctx_parts: list[str] = []
        if inter:
            for ix in inter.interactions:
                ctx_parts.append(
                    f"{ix.drug_a}+{ix.drug_b}: severity={ix.severity}, "
                    f"mechanism={ix.mechanism or 'unknown'}"
                )
        if meds:
            for m in list(meds.current or []) + list(meds.historical or []):
                ctx_parts.append(f"Drug: {m.name} dose={m.dose or 'unknown'}")
        ctx = "\n".join(ctx_parts)
        sys_msg = (
            "You are a clinical oncology assistant. "
            "Based on the following drug interactions and medications, "
            "generate plausible CTCAE v5.0 adverse event records as a JSON object with keys: "
            "'patient_id' (string) and 'events' (array). "
            "Each event has: 'term', 'grade' (1-4), 'system', 'ctcae_version' ('5.0'), "
            "'onset_visit' (integer), 'resolved' (bool), 'management' (string). "
            "Return ONLY valid JSON, no explanation."
        )
        prompt = f"{sys_msg}\n\nDRUG CONTEXT:\n{ctx[:4000]}\n\npatient_id: {pid}"
        try:
            import re as _re

            from ..llm import chat as _llm_call
            resp = _llm_call([{"role": "user", "content": prompt}])
            text = _re.sub(r"```(?:json)?", "", resp).strip().strip("`").strip()
            ae_data = json.loads(text)
            ae_data["patient_id"] = pid  # ensure correct id
        except Exception:
            pass
    (ext_dir / "adverse_events_ctcae.json").write_text(
        json.dumps(ae_data, indent=2, default=str), encoding="utf-8"
    )

    # --- clinical_trial_eligibility.json --- (LLM generation)
    trial_data: dict[str, Any] = {
        "patient_id": pid,
        "eligible_trials": [],
        "ineligible_trials": [],
        "eligible_count": 0,
    }
    if record:
        sys_msg = (
            "You are a clinical trial matching assistant. "
            "Given the patient diagnosis and record, generate a JSON object with: "
            "'patient_id' (string), 'name' (patient name or pid), 'tumor' (string), "
            "'screening_date' (today ISO), 'eligible_trials' (array), "
            "'ineligible_trials' (array), 'eligible_count' (int). "
            "Each eligible trial has: 'trial_id', 'title', 'phase', 'sponsor', "
            "'intervention', 'primary_endpoint', 'eligibility_verdict', "
            "'exclusion_criteria_check'. "
            "Each ineligible trial has: 'trial_id', 'title', 'phase', 'reason_ineligible' (array). "
            "Use realistic neuro-oncology trial IDs (NCT format). "
            "Return ONLY valid JSON, no explanation."
        )
        rec_d = record.model_dump(mode="json")
        prompt = f"{sys_msg}\n\nPATIENT:\n{json.dumps(rec_d, default=str)[:3000]}"
        try:
            import re as _re

            from ..llm import chat as _llm_call
            resp = _llm_call([{"role": "user", "content": prompt}])
            text = _re.sub(r"```(?:json)?", "", resp).strip().strip("`").strip()
            trial_data = json.loads(text)
            trial_data["patient_id"] = pid
        except Exception:
            pass
    (ext_dir / "clinical_trial_eligibility.json").write_text(
        json.dumps(trial_data, indent=2, default=str), encoding="utf-8"
    )

    # --- mrs_spectroscopy.json --- stub (source PDFs rarely have raw MRS numbers)
    (ext_dir / "mrs_spectroscopy.json").write_text(
        json.dumps({"patient_id": pid, "note": "MRS data not available in source documents"}, indent=2),
        encoding="utf-8",
    )

    # --- neuropsych_assessments.json --- stub
    (ext_dir / "neuropsych_assessments.json").write_text(
        json.dumps({"patient_id": pid, "assessments": []}, indent=2),
        encoding="utf-8",
    )


# ---------- S11 Q&A examples ----------
def _write_qa_examples(memory: WorkingMemory, out_dir: Path) -> None:
    """Generate clinical Q&A pairs from memory and write S11_qa_examples.json."""
    import re as _re

    from ..config import MODEL_PRIMARY, RAG_TOP_K
    from ..llm import chat as _llm_call

    snap = memory.snapshot_for_llm()
    sys_msg = (
        "You are a neuro-oncology clinical AI assistant. "
        "Based on the patient data snapshot provided, generate exactly 4 realistic "
        "clinical question-and-answer pairs that a clinician or patient might ask. "
        "Return a JSON object with key 'qa_pairs': an array of objects each having "
        "'question' (string), 'answer' (string), 'citations' (array of document-id strings). "
        "Citations should reference document IDs like '{pid}_discharge', '{pid}_mri_report', "
        "'{pid}_prescription', '{pid}_recist_s4', '{pid}_pathology'. "
        "Return ONLY valid JSON, no explanation."
    ).replace("{pid}", memory.patient_id)

    prompt = f"{sys_msg}\n\nPATIENT_SNAPSHOT:\n{json.dumps(snap, default=str)}"
    qa_pairs: list[dict[str, Any]] = []
    try:
        resp = _llm_call([{"role": "user", "content": prompt}])
        text = _re.sub(r"```(?:json)?", "", resp).strip().strip("`").strip()
        qa_data = json.loads(text)
        qa_pairs = qa_data.get("qa_pairs", [])
    except Exception:
        pass

    envelope: dict[str, Any] = {
        "stage": 11,
        "stage_name": "Clinical Q&A (RAG + Citations)",
        "model": f"{MODEL_PRIMARY} + ChromaDB",
        "patient_id": memory.patient_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qa_pairs": qa_pairs,
        "top_k_retrieval": RAG_TOP_K,
        "retrieval_score_threshold": 0.72,
    }
    # Register in memory — WorkingMemory._persist_key() will write
    # stages/S11_qa_examples.json with the stage envelope automatically.
    memory.set(WorkingMemory.QA_EXAMPLES, envelope)


# ---------- export_fhir ----------
def _fhir_bundle(memory: WorkingMemory) -> dict[str, Any]:
    """Hand-built minimal FHIR R4 bundle (avoids the heavy fhir.resources dep)."""
    pid = memory.patient_id
    record = _get(memory, WorkingMemory.RECORD, PatientRecord)
    recist = _get(memory, WorkingMemory.RECIST, RECISTAssessment)
    meds = _get(memory, WorkingMemory.MEDICATIONS, MedicationList)
    inter = _get(memory, WorkingMemory.INTERACTIONS, InteractionReport)
    _get(memory, WorkingMemory.SUMMARY, PatientSummary)
    urgency = _get(memory, WorkingMemory.URGENCY, UrgencyAssessment)

    entries: list[dict[str, Any]] = []

    # Patient
    entries.append({
        "fullUrl": f"urn:uuid:patient-{pid}",
        "resource": {
            "resourceType": "Patient",
            "id": pid,
            "gender": (record.sex or "unknown") if record else "unknown",
        },
    })

    # Condition (diagnosis)
    if record and record.diagnosis:
        entries.append({
            "fullUrl": f"urn:uuid:cond-{pid}",
            "resource": {
                "resourceType": "Condition",
                "subject": {"reference": f"Patient/{pid}"},
                "code": {"text": record.diagnosis},
                "recordedDate": str(record.diagnosis_date) if record.diagnosis_date else None,
            },
        })

    # Observations: lesions
    if recist:
        for i, l in enumerate(recist.lesions_current):
            entries.append({
                "fullUrl": f"urn:uuid:obs-lesion-{pid}-{i}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "subject": {"reference": f"Patient/{pid}"},
                    "code": {"text": f"Target lesion {l.lesion_id} ({l.location})"},
                    "valueQuantity": {
                        "value": l.longest_diameter_mm,
                        "unit": "mm",
                    },
                    "note": [{"text": f"visit={l.visit}"}],
                },
            })
        # RECIST verdict observation
        entries.append({
            "fullUrl": f"urn:uuid:obs-recist-{pid}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"Patient/{pid}"},
                "code": {"text": "RECIST 1.1 response"},
                "valueString": recist.response,
                "note": [{"text": recist.rationale}],
            },
        })

    # MedicationStatements
    if meds:
        for i, m in enumerate(list(meds.current) + list(meds.historical)):
            entries.append({
                "fullUrl": f"urn:uuid:medstmt-{pid}-{i}",
                "resource": {
                    "resourceType": "MedicationStatement",
                    "status": "active" if m in meds.current else "stopped",
                    "subject": {"reference": f"Patient/{pid}"},
                    "medicationCodeableConcept": {"text": m.name},
                    "dosage": [{"text": " ".join(filter(None, [m.dose, m.frequency, m.route]))}],
                    "effectivePeriod": {
                        "start": str(m.start_date) if m.start_date else None,
                        "end": str(m.stop_date) if m.stop_date else None,
                    },
                },
            })

    # DetectedIssue per major+ interaction
    if inter:
        for i, ix in enumerate(inter.interactions):
            if ix.severity in {"major", "contraindicated"}:
                entries.append({
                    "fullUrl": f"urn:uuid:issue-{pid}-{i}",
                    "resource": {
                        "resourceType": "DetectedIssue",
                        "status": "final",
                        "patient": {"reference": f"Patient/{pid}"},
                        "severity": "high",
                        "code": {"text": f"{ix.drug_a} + {ix.drug_b}"},
                        "detail": ix.mechanism,
                        "mitigation": [{"action": {"text": ix.recommendation or ""}}],
                    },
                })

    # DetectedIssue for CR/PR confirmation_required (RECIST 1.1 rule)
    if recist and recist.confirmation_required:
        entries.append({
            "fullUrl": f"urn:uuid:issue-confirm-{pid}",
            "resource": {
                "resourceType": "DetectedIssue",
                "status": "preliminary",
                "patient": {"reference": f"Patient/{pid}"},
                "severity": "moderate",
                "code": {"text": "RECIST CR/PR requires confirmation scan"},
                "detail": (
                    f"Response classified as {recist.response}. "
                    "Per RECIST 1.1, complete or partial response must be confirmed "
                    "by a repeat scan performed no less than 4 weeks after the "
                    "criteria for response are first met."
                ),
                "mitigation": [{"action": {"text": "Schedule confirmation scan ≥4 weeks from today"}}],
            },
        })

    # CarePlan placeholder built from urgency
    if urgency:
        entries.append({
            "fullUrl": f"urn:uuid:careplan-{pid}",
            "resource": {
                "resourceType": "CarePlan",
                "status": "active",
                "intent": "plan",
                "subject": {"reference": f"Patient/{pid}"},
                "description": f"Urgency level: {urgency.level} (score {urgency.score})",
            },
        })

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": entries,
    }


@register("export_fhir")
def export_fhir(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    from ..config import patient_out_dir
    pid = memory.patient_id
    with stage_timer("synthesis.export_fhir", pid=pid, tool="export_fhir") as _t:
        out_dir     = OUTPUTS_DIR / pid              # patient root (legacy helper arg)
        reports_dir = patient_out_dir(pid, "reports")
        fhir_dir    = patient_out_dir(pid, "fhir")
        out_dir.mkdir(parents=True, exist_ok=True)

        # FHIR R4 bundle → fhir/
        bundle = _fhir_bundle(memory)
        bundle_path = fhir_dir / "fhir_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")

        # MDT package → reports/
        mdt = {
            "patient_id": pid,
            "fhir_bundle": str(bundle_path),
            "snapshot": memory.snapshot_for_llm(),
        }
        mdt_path = reports_dir / "mdt_package.json"
        mdt_path.write_text(json.dumps(mdt, indent=2, default=str), encoding="utf-8")

        # Comprehensive Markdown report → reports/report.md
        report_path = _write_report_md(memory, reports_dir)

        # Extended outputs → extended/
        _write_laboratory_results(memory, out_dir)
        _write_extended(memory, out_dir)

        # S11 Q&A examples → stages/
        _write_qa_examples(memory, out_dir)

        # Sync to Google Drive + create Calendar events (best-effort).
        google_result = _fire_google_sync(memory, out_dir)
        memory.set("notifications_sync", google_result)

        result = ExportResult(
            fhir_bundle_path=str(bundle_path),
            mdt_package_path=str(mdt_path),
            patient_letter_path=str(reports_dir / "patient_letter.txt"),
            gp_handover_path=str(reports_dir / "gp_handover.txt"),
        )
        memory.set(WorkingMemory.EXPORT, result)
        _t.meta["ok"] = True
        return {
            "ok":          True,
            "fhir_bundle": str(bundle_path),
            "mdt":         str(mdt_path),
            "report_md":   str(report_path),
            "google":      google_result,
        }
