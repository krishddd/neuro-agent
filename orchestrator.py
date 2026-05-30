"""Guarded phase-DAG orchestrator.

The orchestrator is a hard-coded sequence of phases. Within each phase the
LLM is given ONLY that phase's tools and runs a short tool-call loop with
a step budget. Inter-phase dependencies are enforced by Python, never by
the model.

Phase 1 (ingest) is fully deterministic — no LLM. Subsequent phases use
qwen3:14b for tool-call reasoning; gemma4:e4b is used only for image analysis.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import (
    MODEL_PRIMARY,
    PHASE_STEP_BUDGET,
    PROMPTS_DIR,
)
from .llm import tool_call
from .memory import WorkingMemory
from .tool_schemas import PHASE_TOOLS
from .tools import dispatch  # registers all sub-agent tools via tools/__init__.py
from .utils.approval import (
    ApprovalRecord,
    archive_markers,
    require_approval,
    write_pending,
)
from .utils.audit import log as audit_log
from .utils.audit import stage_timer

log = logging.getLogger(__name__)


# ---------- phase definition ----------
@dataclass
class Phase:
    name: str
    required_keys: list[str] = field(default_factory=list)
    deterministic: Callable[[WorkingMemory], None] | None = None


def _ingest_phase(memory: WorkingMemory) -> None:
    """Phase 1 — ingest files then immediately index text into RAG.

    Indexing here means chat is available as soon as ingestion finishes,
    even if later phases (RECIST, pharma) haven't run yet.
    """
    dispatch("ingest_patient_files", memory, {"patient_id": memory.patient_id})
    # Best-effort RAG index — errors are logged but don't block the pipeline.
    try:
        dispatch("index_rag", memory, {})
    except Exception as _e:
        audit_log("ingest.index_rag.error", pid=memory.patient_id,
                  status="warning", meta={"error_type": type(_e).__name__})


def _synthesis_phase(memory: WorkingMemory) -> None:
    """Phase 6 — deterministic synthesis: always call all three tools in order.

    Made deterministic (like ingest) because the LLM tool loop was skipping
    write_summary, which prevented Gmail/Calendar notifications from firing.
    All three synthesis tools just process already-computed memory data — there
    is no reason for the LLM to decide the call order.
    """
    for tool_name in ("build_timeline", "write_summary", "export_fhir"):
        try:
            result = dispatch(tool_name, memory, {})
            if not result.get("ok", True):
                log.warning("  [WARN]  synthesis.%s returned ok=False: %s",
                            tool_name, result.get("error", "")[:120])
        except Exception as exc:
            log.warning("  [WARN]  synthesis.%s raised %s: %s",
                        tool_name, type(exc).__name__, str(exc)[:120])


def _prerun_mri_per_visit(memory: WorkingMemory) -> list[str]:
    """Deterministically run analyze_scan(visit) for every visit ingested.

    Guarantees every visit's MRI is actually analysed — the LLM-driven phase
    loop that follows can then call extract_patient_record + compare_scans
    (which need ALL visit observations already in memory).
    """
    from .utils.schemas import IngestionResult
    ing_raw = memory.get(WorkingMemory.INGESTION)
    if ing_raw is None:
        return []
    ing = ing_raw if isinstance(ing_raw, IngestionResult) else IngestionResult.model_validate(ing_raw)
    visits = sorted(ing.visits or [])
    analysed: list[str] = []
    for v in visits:
        try:
            res = dispatch("analyze_scan", memory, {"visit": v})
            if res.get("ok"):
                analysed.append(v)
        except Exception as exc:
            log.warning("  [WARN]  mri.analyze_scan(%s) raised %s: %s",
                        v, type(exc).__name__, str(exc)[:120])
        # Task 5: submit volumetric segmentation — non-blocking on CPU,
        # inline on GPU. Always runs; silently skips if no NIfTI found.
        try:
            dispatch("segment_volumetric", memory, {"visit": v})
        except Exception as exc:
            log.warning("  [WARN]  mri.segment_volumetric(%s) raised %s: %s",
                        v, type(exc).__name__, str(exc)[:120])
        # Phase 5.1 — radiomics on the mask segment_volumetric just produced.
        # Graceful-degrades to radiomics_unavailable=true if pyradiomics or
        # the mask isn't present, so the rest of the pipeline is unaffected.
        try:
            dispatch("extract_radiomics", memory, {"visit": v})
        except Exception as exc:
            log.warning("  [WARN]  mri.extract_radiomics(%s) raised %s: %s",
                        v, type(exc).__name__, str(exc)[:120])
    # If we have ≥2 visits, also run compare_scans deterministically once.
    if len(analysed) >= 2:
        try:
            dispatch("compare_scans", memory,
                     {"baseline_visit": analysed[0], "current_visit": analysed[-1]})
        except Exception as exc:
            log.warning("  [WARN]  mri.compare_scans raised %s: %s",
                        type(exc).__name__, str(exc)[:120])
    return analysed


def _prerun_recist_per_visit(memory: WorkingMemory) -> list[str]:
    """Deterministically run measure_lesions(visit) for every visit.

    The LLM-driven phase loop then calls classify_response (sums across
    all visits already measured) + score_urgency + index_rag.
    """
    from .utils.schemas import IngestionResult
    ing_raw = memory.get(WorkingMemory.INGESTION)
    if ing_raw is None:
        return []
    ing = ing_raw if isinstance(ing_raw, IngestionResult) else IngestionResult.model_validate(ing_raw)
    visits = sorted(ing.visits or [])
    measured: list[str] = []
    for v in visits:
        try:
            res = dispatch("measure_lesions", memory, {"visit": v})
            if res.get("ok"):
                measured.append(v)
        except Exception as exc:
            log.warning("  [WARN]  recist.measure_lesions(%s) raised %s: %s",
                        v, type(exc).__name__, str(exc)[:120])
    return measured


def _treatment_opt_phase(memory: WorkingMemory) -> None:
    """Phase 4 — deterministic Treatment Optimization (SMBO v3.0).

    Sub-steps 4a and 4b always run.
    Sub-steps 4c–4e run only when optimization_triggered=True from 4b.
    A SKIP TreatmentProposal is written when not triggered, so S18 always exists.
    """
    # 4a: patient state vectorization + medication pre-seeding (always)
    try:
        dispatch("extract_patient_state", memory, {})
    except Exception as exc:
        log.warning("  [WARN]  treatment_opt.extract_patient_state: %s", exc)

    # ── Task 3: Critical biomarker hard-stop (MGMT / IDH for GBM-class) ──────
    # If 4a flagged biomarker_hard_stop=True, we MUST NOT run 4b–4e and must
    # NOT impute the missing markers. Write a SKIP proposal that escalates
    # to MDT for molecular testing.
    try:
        from .utils.schemas import PatientStateVector, TreatmentProposal
        ps_raw = memory.get(WorkingMemory.PATIENT_STATE)
        if ps_raw:
            ps = (ps_raw if isinstance(ps_raw, PatientStateVector)
                  else PatientStateVector.model_validate(ps_raw))
            if ps.biomarker_hard_stop:
                reason = ps.biomarker_hard_stop_reason or "missing_critical_biomarkers"
                log.warning("  treatment_opt: HARD STOP — %s", reason)
                proposal = TreatmentProposal(
                    decision="SKIP",
                    reason=reason,
                    mdt_discussion_required=True,
                    clinical_narrative=(
                        "Treatment optimisation halted: critical molecular "
                        "biomarkers (MGMT promoter methylation and/or IDH "
                        "mutation) are missing. These biomarkers are decision-"
                        "driving for high-grade glioma management and cannot "
                        "be imputed. Order pathology re-review with IHC and "
                        "MGMT methylation PCR, then re-run Phase 4."
                    ),
                )
                memory.set(WorkingMemory.TREATMENT_PROPOSAL, proposal)
                # Best-effort doctor notification using SPIKES-routed mailer.
                try:
                    from .integrations.gmail_client import DOCTOR_EMAIL, GmailClient
                    gc = GmailClient()
                    if gc.ready and DOCTOR_EMAIL:
                        gc.send_mdt_alert(
                            patient_id=memory.patient_id,
                            proposal=proposal,
                            subject_prefix="[BIOMARKER HARD-STOP] ",
                        )
                except Exception as exc:
                    log.warning("  treatment_opt: hard-stop email failed: %s", exc)
                return
    except Exception as exc:
        log.warning("  [WARN]  treatment_opt biomarker hard-stop check failed: %s", exc)

    # 4b: GP + RSF prediction + trigger flag (always)
    try:
        dispatch("predict_recist_pfs", memory, {})
    except Exception as exc:
        log.warning("  [WARN]  treatment_opt.predict_recist_pfs: %s", exc)

    # Check trigger flag — if False, write SKIP and exit early
    pred_raw = memory.get(WorkingMemory.PREDICTION)
    triggered = False
    if pred_raw:
        p = pred_raw if isinstance(pred_raw, dict) else pred_raw.model_dump()
        triggered = bool(p.get("optimization_triggered", False))

    if not triggered:
        from .utils.schemas import TreatmentProposal
        skip_proposal = TreatmentProposal(
            decision="SKIP",
            reason="patient_responding_adequately",
            clinical_narrative=(
                "RECIST response indicates adequate treatment response; "
                "SMBO optimisation was not triggered."
            ),
        )
        memory.set(WorkingMemory.TREATMENT_PROPOSAL, skip_proposal)
        log.info("  treatment_opt: SKIP — optimization not triggered")
        return

    # 4c–4e: conditional on trigger. Task 8 inserts match_clinical_trials
    # as sub-step 4d.5 — between SHAP explainability and the MDT debate, so
    # persona prompts can reference the top-3 trial candidates.
    for tool_name in (
        "run_smbo_optimization",
        "explain_with_shap",
        # Phase 5.5 / Module 3 — sub-step 4d.4 PubMed evidence (between SHAP
        # and trial matching) so the MDT Neuro-Oncologist persona can cite PMIDs.
        "retrieve_pubmed_evidence",
        "match_clinical_trials",
        # Phase 5.7 / Extra B — sub-step 4d.6 FAERS adverse-event check
        # (gated on candidates flagged off_label or novel_combo).
        "check_adverse_events",
        "review_proposal_mdt",
    ):
        try:
            result = dispatch(tool_name, memory, {})
            if not result.get("ok", True):
                log.warning("  [WARN]  treatment_opt.%s returned ok=False", tool_name)
        except Exception as exc:
            log.warning("  [WARN]  treatment_opt.%s raised %s: %s",
                        tool_name, type(exc).__name__, str(exc)[:120])


PHASES: list[Phase] = [
    Phase(
        name="ingest",
        required_keys=[WorkingMemory.INGESTION],
        deterministic=_ingest_phase,
    ),
    Phase(name="mri", required_keys=[WorkingMemory.VISION, WorkingMemory.RECORD]),
    Phase(name="recist", required_keys=[WorkingMemory.RECIST, WorkingMemory.URGENCY]),
    Phase(
        name="treatment_opt",
        required_keys=[WorkingMemory.TREATMENT_PROPOSAL],
        deterministic=_treatment_opt_phase,
    ),
    # P001-RUN-FIX (#6): adding CORRELATION here forces the pharma phase
    # loop to keep running until ``correlate_treatment`` has been called.
    # Previously, the LLM declared PHASE_DONE after just MEDICATIONS +
    # INTERACTIONS landed — so S09_correlation.json was never produced.
    Phase(name="pharma", required_keys=[
        WorkingMemory.MEDICATIONS,
        WorkingMemory.INTERACTIONS,
        WorkingMemory.CORRELATION,
    ]),
    Phase(
        name="synthesis",
        required_keys=[WorkingMemory.TIMELINE, WorkingMemory.SUMMARY, WorkingMemory.EXPORT],
        deterministic=_synthesis_phase,
    ),
]


# ---------- in-memory job store ----------
JOBS: dict[str, dict[str, Any]] = {}


def _system_prompt() -> str:
    p = PROMPTS_DIR / "orchestrator_system.md"
    return p.read_text(encoding="utf-8") if p.exists() else "You are an orchestrator."


def _build_user_msg(phase: Phase, memory: WorkingMemory, hint: str | None = None) -> str:
    snap = memory.snapshot_for_llm()
    return (
        f"PHASE: {phase.name}\n"
        f"WORKING_MEMORY: {json.dumps(snap)}\n"
        f"REQUIRED_OUTPUTS: {phase.required_keys}\n"
        + (f"HINT: {hint}\n" if hint else "")
        + "Decide tool calls to advance the phase. Reply PHASE_DONE when done."
    )


def _phase_complete(phase: Phase, memory: WorkingMemory) -> bool:
    return all(memory.has(k) for k in phase.required_keys)


def run_phase(phase: Phase, memory: WorkingMemory) -> dict[str, Any]:
    """Run a single phase with the guarded tool-call loop."""
    pid = memory.patient_id
    if phase.deterministic is not None:
        with stage_timer(f"phase.{phase.name}", pid=pid, phase=phase.name) as _t:
            phase.deterministic(memory)
            _t.meta["steps"] = 1
        memory.mark_phase(phase.name, "ok", steps=1)
        return {"phase": phase.name, "steps": 1, "ok": True}

    tools = PHASE_TOOLS.get(phase.name, [])
    if not tools:
        memory.mark_phase(phase.name, "skipped", steps=0)
        return {"phase": phase.name, "steps": 0, "ok": True, "reason": "no tools"}

    # Phase-specific deterministic pre-runs to GUARANTEE per-visit coverage.
    # (Prevents the LLM from silently skipping visit2 when it decides one call
    # is "enough".)
    prerun_hint: str | None = None
    if phase.name == "mri":
        analysed = _prerun_mri_per_visit(memory)
        if analysed:
            prerun_hint = (
                f"analyze_scan has ALREADY been run for visits: {analysed}. "
                "Now call extract_patient_record to finalize this phase."
            )
    elif phase.name == "recist":
        measured = _prerun_recist_per_visit(memory)
        if measured:
            prerun_hint = (
                f"measure_lesions has ALREADY been run for visits: {measured}. "
                "Now call classify_response (uses all visits), then score_urgency, "
                "then index_rag. Do NOT call measure_lesions again."
            )

    sys = _system_prompt()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": _build_user_msg(phase, memory, hint=prerun_hint)},
    ]

    steps = 0
    failures = 0
    last_hint: str | None = None

    with stage_timer(f"phase.{phase.name}", pid=pid, phase=phase.name) as _t:
        while steps < PHASE_STEP_BUDGET:
            steps += 1
            try:
                resp = tool_call(messages, tools, model=MODEL_PRIMARY)
            except Exception as e:
                audit_log(
                    f"phase.{phase.name}.llm_error",
                    pid=pid,
                    status="error",
                    meta={"phase": phase.name, "error_type": type(e).__name__},
                )
                break

            calls = resp.get("tool_calls") or []
            content = (resp.get("content") or "").strip()

            if not calls:
                # Model decided we're done — verify and exit.
                if "PHASE_DONE" in content.upper() or _phase_complete(phase, memory):
                    break
                # Nudge once.
                last_hint = "You produced no tool calls. Call a tool or reply PHASE_DONE."
                messages.append({"role": "assistant", "content": content or "(empty)"})
                messages.append({"role": "user", "content": last_hint})
                continue

            # Append the assistant turn so the model sees its own calls.
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            })

            for c in calls:
                name = c["name"]
                args = c["arguments"] or {}
                try:
                    result = dispatch(name, memory, args)
                    ok = bool(result.get("ok", True))
                    if not ok:
                        failures += 1
                except Exception as e:
                    failures += 1
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"[:240]}

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result)[:2000],
                })

            if _phase_complete(phase, memory):
                break
            if failures >= 2:
                audit_log(
                    f"phase.{phase.name}.bailout",
                    pid=pid,
                    status="error",
                    meta={"phase": phase.name, "steps": steps},
                )
                break

        _t.meta["steps"] = steps
        _t.meta["ok"] = _phase_complete(phase, memory)

    status = "ok" if _phase_complete(phase, memory) else "incomplete"
    memory.mark_phase(phase.name, status, steps=steps)
    return {"phase": phase.name, "steps": steps, "ok": status == "ok"}


# ---------- dynamic phase skip ----------
def _should_skip_phase(phase_name: str, memory: WorkingMemory) -> str | None:
    """Return a skip reason string if this phase should be bypassed, else None.

    Called only after the ingest phase has written INGESTION to memory.
    """
    if not memory.has(WorkingMemory.INGESTION):
        return None  # Can't decide without ingestion data; let the phase try.

    try:
        from .utils.schemas import IngestionResult
        ing_raw = memory.get(WorkingMemory.INGESTION)
        ing = ing_raw if isinstance(ing_raw, IngestionResult) else IngestionResult.model_validate(ing_raw)
    except Exception:
        return None

    file_kinds = {f.kind for f in ing.files}

    if phase_name in ("mri", "recist"):
        if "mri_image" not in file_kinds and "mri_report" not in file_kinds:
            return "no MRI images or reports found for this patient"

    if phase_name == "pharma":
        if not {"prescription", "discharge"} & file_kinds:
            return "no prescription or discharge files found for this patient"

    # Treatment optimization: skip entirely for PR/CR patients (already responding)
    if phase_name == "treatment_opt":
        recist_raw = memory.get(WorkingMemory.RECIST)
        if recist_raw:
            r = recist_raw if isinstance(recist_raw, dict) else recist_raw.model_dump()
            response = r.get("response", "NE")
            if response in ("PR", "CR"):
                return f"RECIST response {response} — treatment optimisation not warranted"
        # No RECIST data yet → let the phase run (it handles missing data gracefully)

    # SAFETY GATE — when Phase 4 produced SKIP (biomarker hard-stop OR no-trigger),
    # the downstream phases (pharma, synthesis) must NOT fire. A SKIP means "do
    # not act on this proposal yet" — issuing a patient letter or calendar events
    # for a halted regimen is a clinical-safety violation. The biomarker MDT
    # alert email has already been sent by the treatment_opt phase itself.
    # Disabled under DEV_MODE so end-to-end dev runs always complete all phases.
    if phase_name in ("pharma", "synthesis"):
        from .config import DEV_MODE as _DEV_MODE
        if _DEV_MODE:
            return None
        prop_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
        if prop_raw:
            p = prop_raw if isinstance(prop_raw, dict) else prop_raw.model_dump()
            decision = str(p.get("decision", "")).upper()
            if decision == "SKIP":
                reason_tail = str(p.get("reason", ""))[:140]
                return f"treatment_opt decision=SKIP — {reason_tail}"

    return None


# ---------- top-level run ----------
# Task 9 — phase membership per HITL mode.
# Any phase whose side-effects (outbound email, calendar, Drive sync) MUST NOT
# fire before clinician approval belongs in EXECUTE_PHASES.
PREP_PHASES:    tuple[str, ...] = ("ingest", "mri", "recist", "treatment_opt")
EXECUTE_PHASES: tuple[str, ...] = ("pharma", "synthesis")


def run_patient(
    patient_id: str,
    *,
    job_id: str | None = None,
    stop_after: str | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    """Execute the guarded DAG for one patient.

    ``mode`` (Task 9 HITL gate):
        - ``prep``    runs phases 1–4 (ingest → treatment_opt) then stops and
                      writes ``PENDING_APPROVAL.json`` summarising the MDT
                      decision. No Drive / email / calendar side-effects.
        - ``execute`` runs phases 5–6 (pharma → synthesis). Requires
                      ``APPROVED.json`` on disk — otherwise raises
                      ``ApprovalRequiredError``. Archives the approval markers
                      into ``outputs/<pid>/audit/`` after synthesis.
        - ``full``    legacy single-shot run (no approval gate).

    ``stop_after`` lets the CLI run only up to a given phase (useful while
    later phases are still being implemented). Applies within the selected mode.
    """
    if mode not in ("full", "prep", "execute"):
        raise ValueError(f"invalid mode {mode!r} — expected full|prep|execute")
    job_id = job_id or uuid.uuid4().hex[:12]
    # In execute-mode, rehydrate from the prep-phase persisted memory so
    # phases 5–6 see the full Phase-4 context; otherwise start fresh.
    if mode == "execute":
        memory = WorkingMemory.load(patient_id)
        memory.job_id = job_id
    else:
        memory = WorkingMemory(job_id=job_id, patient_id=patient_id)
    # Update rather than overwrite — the API may have already set "status": "running".
    JOBS.setdefault(job_id, {}).update({
        "patient_id": patient_id,
        "phase":      None,
        "started_at": time.time(),
        "phases":     [],
        "qa_ready":   False,
    })

    # ── Task 9: HITL mode — resolve allowed phase list + approval guard ──────
    if mode == "prep":
        allowed = set(PREP_PHASES)
    elif mode == "execute":
        allowed = set(EXECUTE_PHASES)
        # Refuse to start without APPROVED.json on disk.
        approval: ApprovalRecord = require_approval(patient_id)
        JOBS[job_id]["approval"] = {
            "approver_email": approval.approver_email,
            "decision":       approval.decision,
            "approved_at":    approval.approved_at,
        }
        # If the clinician chose MODIFY + override_regimen, patch the in-memory
        # proposal BEFORE phase 5 runs so pharma/synthesis see the edit.
        if approval.decision == "MODIFY" and approval.override_regimen:
            try:
                _apply_override_regimen(memory, approval.override_regimen)
            except Exception as exc:
                log.warning("approval override regimen apply failed: %s", exc)
    else:
        allowed = {p.name for p in PHASES}

    log.info("=" * 60)
    log.info("PIPELINE START  patient=%s  job=%s  mode=%s", patient_id, job_id, mode)
    log.info("=" * 60)

    for phase in PHASES:
        if phase.name not in allowed:
            log.info("  [MODE-SKIP] %-12s  (mode=%s)", phase.name, mode)
            continue
        JOBS[job_id]["phase"] = phase.name

        # Skip phases whose tools aren't yet implemented.
        if phase.deterministic is None and not PHASE_TOOLS.get(phase.name):
            log.info("  [SKIP]  %-12s  (no tools registered)", phase.name)
            memory.mark_phase(phase.name, "skipped", steps=0)
            JOBS[job_id]["phases"].append({"phase": phase.name, "ok": True, "skipped": True})
            if stop_after and phase.name == stop_after:
                break
            continue

        # Dynamic phase skip: check whether the data warrants this phase.
        skip_reason = _should_skip_phase(phase.name, memory)
        if skip_reason:
            log.info("  [SKIP]  %-12s  %s", phase.name, skip_reason)
            memory.mark_phase(phase.name, "skipped", steps=0)
            JOBS[job_id]["phases"].append({
                "phase": phase.name, "ok": True, "skipped": True, "reason": skip_reason,
            })
            audit_log(
                f"phase.{phase.name}.skipped",
                pid=patient_id, status="skipped",
                meta={"reason": skip_reason},
            )
            if stop_after and phase.name == stop_after:
                break
            continue

        log.info("  [START] %-12s  ...", phase.name)
        result = run_phase(phase, memory)
        JOBS[job_id]["phases"].append(result)

        if result.get("ok"):
            log.info("  [DONE]  %-12s  steps=%s", phase.name, result.get("steps", "?"))
        else:
            log.warning("  [WARN]  %-12s  incomplete — continuing pipeline", phase.name)
            JOBS[job_id].setdefault("warnings", []).append(phase.name)
            audit_log(
                f"phase.{phase.name}.continued_after_failure",
                pid=patient_id, status="warning",
                meta={"phase": phase.name},
            )
        if stop_after and phase.name == stop_after:
            break

    elapsed = int(time.time() - JOBS[job_id]["started_at"])
    JOBS[job_id]["finished_at"] = time.time()
    JOBS[job_id]["qa_ready"] = memory.has(WorkingMemory.INGESTION)

    # Collect Gmail / Drive / Calendar results from synthesis phase.
    notifs: dict = {}
    notifs.update(memory.get("notifications_gmail") or {})
    notifs.update(memory.get("notifications_sync")  or {})
    if notifs:
        JOBS[job_id]["notifications"] = notifs
        log.info("  [NOTIFY] %s", notifs)

    # Surface warnings without marking the whole run as an error.
    warns = JOBS[job_id].get("warnings", [])
    if warns:
        JOBS[job_id]["status"] = "completed_with_warnings"
        log.warning("PIPELINE DONE  patient=%s  time=%ds  warnings=%s",
                    patient_id, elapsed, warns)
    else:
        log.info("=" * 60)
        log.info("PIPELINE DONE  patient=%s  time=%ds  all stages OK",
                 patient_id, elapsed)
        log.info("=" * 60)

    memory.finalize()

    # ── Task 9: HITL markers ─────────────────────────────────────────────────
    if mode == "prep":
        proposal_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
        proposal_d: dict[str, Any] = {}
        if proposal_raw is not None:
            proposal_d = (proposal_raw if isinstance(proposal_raw, dict)
                          else proposal_raw.model_dump())
        pending = write_pending(patient_id, proposal_d)
        JOBS[job_id]["pending_approval"] = pending.model_dump()
        log.info("PIPELINE PREP  patient=%s  → PENDING_APPROVAL.json  decision=%s",
                 patient_id, pending.mdt_decision)
    elif mode == "execute":
        try:
            moved = archive_markers(patient_id)
            if moved:
                JOBS[job_id]["approval_archived"] = moved
                log.info("PIPELINE EXECUTE  patient=%s  approval markers archived: %s",
                         patient_id, moved)
        except Exception as exc:
            log.warning("approval archival failed: %s", exc)

    return {
        "job_id":     job_id,
        "patient_id": patient_id,
        "mode":       mode,
        "phases":     JOBS[job_id]["phases"],
        "qa_ready":   JOBS[job_id]["qa_ready"],
        "pending_approval": JOBS[job_id].get("pending_approval"),
    }


def _apply_override_regimen(memory: WorkingMemory, override_regimen: str) -> None:
    """Patch the in-memory TreatmentProposal with a clinician-authored regimen.

    Only called when the approval decision is ``MODIFY`` and the clinician
    supplied a non-empty ``override_regimen``. The edited proposal is
    re-persisted so downstream phases (pharma, synthesis) see the change.
    """
    from .utils.schemas import TreatmentProposal
    prop_raw = memory.get(WorkingMemory.TREATMENT_PROPOSAL)
    if prop_raw is None:
        # Execute-mode rehydrated from disk but the prep run never produced
        # a TreatmentProposal — don't silently drop the clinician override.
        log.error("approval override supplied but no TreatmentProposal in memory "
                  "for %s — re-run prep mode before execute", memory.patient_id)
        raise RuntimeError(
            f"cannot apply clinician override_regimen for {memory.patient_id}: "
            f"no TreatmentProposal in memory (prep phase may not have completed)"
        )
    prop = (prop_raw if isinstance(prop_raw, TreatmentProposal)
            else TreatmentProposal.model_validate(prop_raw))
    object.__setattr__(prop, "proposed_regimen", override_regimen.strip())
    prop.modifications.append(f"clinician_override: {override_regimen.strip()}")
    memory.set(WorkingMemory.TREATMENT_PROPOSAL, prop)
