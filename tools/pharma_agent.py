"""Phase 4 — pharma sub-agent.

Registered tools:

    extract_medications()    -> MedicationList in memory
    check_interactions()     -> InteractionReport in memory (KB-first, LLM second)
    correlate_treatment()    -> CorrelationResult in memory

The interaction check is KB-first: every pair of current+historical meds is
looked up in `Datasets/raw_docs/drug_interaction_kb.json`. Pairs not in the
KB are forwarded to qwen3:14b in JSON mode for a conservative inference (and
get `source = "llm_inferred"`).
"""
from __future__ import annotations

import json
import re
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config import DRUG_KB_PATH
from ..llm import json_call, vision_json
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.schemas import (
    CorrelationResult,
    IngestedFile,
    IngestionResult,
    Interaction,
    InteractionReport,
    MedEvent,
    Medication,
    MedicationList,
    RECISTAssessment,
)
from ..utils.tool_helpers import drug_interactions_collection, get_ingestion, load_prompt
from . import register

_SEVERITY_RANK = {
    "none": 0, "minor": 1, "moderate": 2, "major": 3, "contraindicated": 4,
}
_SEVERITY_FROM_RANK = {v: k for k, v in _SEVERITY_RANK.items()}


# ---------- helpers ----------
def _med_source_files(ing: IngestionResult) -> list[IngestedFile]:
    return [f for f in ing.files if f.kind in {"prescription", "discharge"}]


# ---------- KB ----------
_KB_CACHE: dict[str, Any] | None = None


def _load_kb() -> dict[tuple[str, str], dict[str, Any]]:
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE  # type: ignore[return-value]
    path = Path(DRUG_KB_PATH)
    table: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("interactions", []):
                a = (entry.get("drug_a") or "").strip().lower()
                b = (entry.get("drug_b") or "").strip().lower()
                if not a or not b:
                    continue
                table[tuple(sorted([a, b]))] = entry  # type: ignore[index]
        except Exception:
            pass
    _KB_CACHE = table  # type: ignore[assignment]
    return table


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


# ---------- extract_medications ----------
_MED_SCHEMA_HINT = """
{
  "current": [
    {"name": "Temozolomide", "dose": "150 mg/m2", "frequency": "daily x5/28d",
     "route": "PO", "start_date": "2024-09-15", "stop_date": null,
     "indication": "GBM"}
  ],
  "historical": []
}
"""


@register("extract_medications")
def extract_medications(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("pharma.extract_medications", pid=pid, tool="extract_medications") as _t:
        ing = get_ingestion(memory)
        med_files = _med_source_files(ing)
        if not med_files:
            empty = MedicationList()
            memory.set(WorkingMemory.MEDICATIONS, empty)
            return {"ok": True, "n_current": 0, "n_historical": 0, "reason": "no prescription/discharge files"}

        text_blob = "\n\n".join(
            f"--- {Path(f.path).name} ---\n{f.text}"
            for f in med_files if f.text
        )
        image_paths = [f.image_path for f in med_files if f.image_path]

        sys_msg = load_prompt("pharma_system.md")
        prompt = (
            f"{sys_msg}\n\n"
            f"PRESCRIPTION/DISCHARGE TEXT:\n{text_blob or '(none)'}\n\n"
            f"Return ONE JSON object matching this shape:\n{_MED_SCHEMA_HINT}"
        )

        try:
            if image_paths:
                meds = vision_json(prompt, image_paths, MedicationList)
            else:
                meds = json_call([{"role": "user", "content": prompt}], MedicationList)
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

        memory.set(WorkingMemory.MEDICATIONS, meds)
        _t.meta["ok"] = True
        return {
            "ok": True,
            "n_current": len(meds.current),
            "n_historical": len(meds.historical),
            "current": [m.name for m in meds.current][:20],
        }


# ---------- check_interactions ----------
class _LLMPairVerdict(BaseModel):
    severity: str
    mechanism: str | None = None
    recommendation: str | None = None


def _llm_check_pair(a: str, b: str) -> Interaction:
    prompt = (
        "You are a clinical pharmacist. Assess the drug-drug interaction "
        f"between '{a}' and '{b}' for a neuro-oncology patient. Return ONE "
        'JSON object: {"severity": "none|minor|moderate|major|contraindicated", '
        '"mechanism": "...", "recommendation": "..."}. Be conservative.'
    )
    try:
        v = json_call([{"role": "user", "content": prompt}], _LLMPairVerdict)
        sev = v.severity if v.severity in _SEVERITY_RANK else "none"
        return Interaction(
            drug_a=a, drug_b=b, severity=sev,  # type: ignore[arg-type]
            mechanism=v.mechanism, recommendation=v.recommendation,
            source="llm_inferred",
        )
    except Exception:
        return Interaction(drug_a=a, drug_b=b, severity="none", source="llm_inferred")


_LIGHTRAG_SEVERITY_PATTERN = re.compile(
    r"\b(contraindicated|major|moderate|minor|none)\b", re.IGNORECASE,
)


def _parse_lightrag_answer(text: str) -> tuple[str, str | None]:
    """Extract (severity, mechanism) from a LightRAG hybrid answer.

    Returns ``("unknown", None)`` if no severity word is present so the
    caller can fall through to the LLM check. The mechanism is the first
    sentence after the severity word, truncated to 200 chars for
    consistency with KB entries.
    """
    if not text:
        return ("unknown", None)
    m = _LIGHTRAG_SEVERITY_PATTERN.search(text)
    if not m:
        return ("unknown", None)
    sev = m.group(1).lower()
    if sev not in _SEVERITY_RANK:
        return ("unknown", None)
    # Take ~1 sentence after the severity hit as mechanism.
    tail = text[m.end():].strip(" .;:-—\n")
    mech = tail.split(".")[0][:200] if tail else None
    return (sev, mech or None)


@register("check_interactions")
def check_interactions(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("pharma.check_interactions", pid=pid, tool="check_interactions") as _t:
        meds_obj = memory.get(WorkingMemory.MEDICATIONS)
        if meds_obj is None:
            return {"ok": False, "reason": "extract_medications must run first"}
        meds = meds_obj if isinstance(meds_obj, MedicationList) else MedicationList.model_validate(meds_obj)

        all_meds: list[Medication] = list(meds.current) + list(meds.historical)
        names = sorted({_normalize(m.name) for m in all_meds if m.name})
        if len(names) < 2:
            report = InteractionReport(interactions=[], highest_severity="none")
            memory.set(WorkingMemory.INTERACTIONS, report)
            return {"ok": True, "n": 0, "highest_severity": "none"}

        kb = _load_kb()
        # Pre-warm the ChromaDB drug_interactions collection (embeds KB once).
        try:
            _drug_col = drug_interactions_collection()
        except Exception:
            _drug_col = None

        interactions: list[Interaction] = []
        max_rank = 0
        flags: list[str] = []
        kb_hits = 0
        semantic_hits = 0
        llm_hits = 0

        for a, b in combinations(names, 2):
            key = tuple(sorted([a, b]))
            entry = kb.get(key)
            if entry is not None:
                # Primary: exact flat-KB match.
                sev = (entry.get("severity") or "none").lower()
                if sev not in _SEVERITY_RANK:
                    sev = "none"
                interactions.append(Interaction(
                    drug_a=entry.get("drug_a", a),
                    drug_b=entry.get("drug_b", b),
                    severity=sev,  # type: ignore[arg-type]
                    mechanism=entry.get("mechanism"),
                    recommendation=entry.get("clinical_note"),
                    source="kb",
                ))
                kb_hits += 1
            else:
                # Secondary: semantic search in ChromaDB drug_interactions collection.
                sem_hit = None
                if _drug_col is not None:
                    try:
                        from ..llm import embed as _embed
                        query = f"{a} + {b} drug interaction"
                        vec = _embed([query])
                        if vec:
                            res = _drug_col.query(query_embeddings=[vec[0]], n_results=1)
                            metas = (res.get("metadatas") or [[]])[0]
                            dists = (res.get("distances") or [[]])[0]
                            if metas and dists and float(dists[0]) < 0.25:
                                m = metas[0]
                                sev = (m.get("severity") or "none").lower()
                                if sev not in _SEVERITY_RANK:
                                    sev = "none"
                                sem_hit = Interaction(
                                    drug_a=m.get("drug_a", a),
                                    drug_b=m.get("drug_b", b),
                                    severity=sev,  # type: ignore[arg-type]
                                    mechanism=m.get("mechanism"),
                                    recommendation=m.get("clinical_note"),
                                    source="kb_semantic",
                                )
                                semantic_hits += 1
                    except Exception:
                        pass

                # Phase 5.4 / Module 1 — LightRAG hybrid causal-graph query.
                # Tried only when KB miss + Chroma miss; the graph captures
                # mechanistic chains (drug → inhibits → enzyme → AE) that
                # flat keyword search can't reach.
                lightrag_hit = None
                if sem_hit is None:
                    try:
                        from ..utils import lightrag_store
                        if lightrag_store.is_available():
                            answer = lightrag_store.query_hybrid(
                                memory.patient_id,
                                f"What is the clinical interaction between {a} and {b}? "
                                f"Identify severity (major/moderate/minor/none), mechanism, "
                                f"and clinical recommendation.",
                            )
                            sev_g, mech_g = _parse_lightrag_answer(answer)
                            if sev_g != "unknown":
                                lightrag_hit = Interaction(
                                    drug_a=a, drug_b=b,
                                    severity=sev_g,  # type: ignore[arg-type]
                                    mechanism=mech_g,
                                    recommendation=None,
                                    source="lightrag",
                                )
                    except Exception:
                        pass

                if sem_hit is not None:
                    interactions.append(sem_hit)
                elif lightrag_hit is not None:
                    interactions.append(lightrag_hit)
                else:
                    interactions.append(_llm_check_pair(a, b))
                    llm_hits += 1

            r = _SEVERITY_RANK[interactions[-1].severity]
            if r > max_rank:
                max_rank = r
            if r >= 3:
                flags.append(f"{interactions[-1].drug_a}+{interactions[-1].drug_b}:{interactions[-1].severity}")

        report = InteractionReport(
            interactions=interactions,
            highest_severity=_SEVERITY_FROM_RANK[max_rank],  # type: ignore[arg-type]
            flags=flags,
        )
        memory.set(WorkingMemory.INTERACTIONS, report)
        _t.meta["ok"] = True
        _t.meta["severity"] = report.highest_severity
        return {
            "ok": True,
            "n": len(interactions),
            "kb_hits": kb_hits,
            "semantic_hits": semantic_hits,
            "llm_hits": llm_hits,
            "highest_severity": report.highest_severity,
            "flags": flags[:10],
        }


# ---------- correlate_treatment ----------
def _build_med_events(all_meds: list[Medication]) -> list[MedEvent]:
    """Build a full chronological medication event timeline.

    Each start, stop, and detectable dose-change is recorded separately so that
    RECIST response can be correlated against the nearest preceding drug event,
    not just the earliest start date.
    """
    events: list[MedEvent] = []
    # Group by drug name to detect dose changes across visits.
    by_name: dict[str, list[Medication]] = {}
    for m in all_meds:
        key = _normalize(m.name)
        by_name.setdefault(key, []).append(m)

    for name, entries in by_name.items():
        entries_sorted = sorted(entries, key=lambda m: str(m.start_date or ""))
        prev_dose: str | None = None
        for m in entries_sorted:
            if m.start_date:
                if prev_dose is not None and m.dose and m.dose != prev_dose:
                    events.append(MedEvent(
                        drug=m.name, event_type="dose_change",
                        event_date=m.start_date, dose=m.dose,
                    ))
                else:
                    events.append(MedEvent(
                        drug=m.name, event_type="start",
                        event_date=m.start_date, dose=m.dose,
                    ))
            if m.stop_date:
                events.append(MedEvent(
                    drug=m.name, event_type="stop",
                    event_date=m.stop_date, dose=m.dose,
                ))
            prev_dose = m.dose

    events.sort(key=lambda e: str(e.event_date or ""))
    return events


def _nearest_event_before(events: list[MedEvent], target_date: date) -> MedEvent | None:
    """Return the event immediately preceding target_date."""
    prior = [e for e in events if e.event_date and e.event_date <= target_date]
    return prior[-1] if prior else None


@register("correlate_treatment")
def correlate_treatment(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("pharma.correlate_treatment", pid=pid, tool="correlate_treatment") as _t:
        meds_obj = memory.get(WorkingMemory.MEDICATIONS)
        recist_obj = memory.get(WorkingMemory.RECIST)
        if meds_obj is None or recist_obj is None:
            return {"ok": False, "reason": "needs medications and recist in memory"}

        meds = meds_obj if isinstance(meds_obj, MedicationList) else MedicationList.model_validate(meds_obj)
        recist = recist_obj if isinstance(recist_obj, RECISTAssessment) else RECISTAssessment.model_validate(recist_obj)

        all_meds: list[Medication] = list(meds.current) + list(meds.historical)
        med_events = _build_med_events(all_meds)

        # Earliest treatment start across all drugs.
        start_dates = [e.event_date for e in med_events if e.event_type == "start" and e.event_date]
        treatment_started = min(start_dates) if start_dates else None

        # Nearest event before today (proxy for response assessment date).
        from datetime import date as _date
        nearest = _nearest_event_before(med_events, _date.today())

        response_observed = None
        lag_days = None
        consistent = recist.response in {"PR", "CR", "SD"}

        summary_bits = [
            f"RECIST response: {recist.response}",
            f"Sum change: {(recist.pct_change or 0) * 100:.1f}%" if recist.pct_change is not None else "Sum change: n/a",
            f"Treatment started: {treatment_started}" if treatment_started else "No dated treatment start",
            f"Nearest preceding drug event: {nearest.drug} {nearest.event_type} ({nearest.event_date})" if nearest else "none",
            f"New lesion: {recist.new_lesion_detected}",
            f"Confirmation required: {recist.confirmation_required}",
        ]
        result = CorrelationResult(
            summary=" | ".join(summary_bits),
            treatment_started=treatment_started,
            response_observed=response_observed,
            lag_days=lag_days,
            consistent=consistent,
            notes=recist.rationale,
            med_events=med_events,
        )
        memory.set(WorkingMemory.CORRELATION, result)
        _t.meta["ok"] = True
        return {
            "ok": True,
            "consistent": consistent,
            "n_med_events": len(med_events),
            "treatment_started": str(treatment_started) if treatment_started else None,
            "nearest_event": (
                f"{nearest.drug} {nearest.event_type}" if nearest else None
            ),
            "response": recist.response,
            "new_lesion": recist.new_lesion_detected,
        }
