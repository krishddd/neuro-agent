"""Phase 3 — RECIST measurement, response classification, urgency triage,
RAG indexing.

Registered tools:

    measure_lesions(visit)   -> writes RECIST.lesions_<visit> to memory
    classify_response()      -> writes RECISTAssessment to memory
    score_urgency()          -> writes UrgencyAssessment to memory
    index_rag()              -> chunks all text + embeds into Chroma

`classify_response` is rule-based on top of measurements (deterministic),
`score_urgency` is rule-based + keyword matching (deterministic),
`measure_lesions` and `index_rag` use the LLM / embedding model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import logging
import re

log = logging.getLogger(__name__)

from ..config import (
    BRAIN_TUMOUR_KEYWORDS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    NEW_LESION_KEYWORDS,
    RECIST_MIN_LESION_MM,
    RECIST_PD_THRESHOLD,
    RECIST_PR_THRESHOLD,
    URGENCY_KEYWORDS_CRITICAL,
    URGENCY_KEYWORDS_HIGH,
)
from ..llm import embed, vision_json
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.schemas import (
    PatientRecord,
    RANOAssessment,
    RECISTAssessment,
    RECISTLesion,
    UrgencyAssessment,
    VisionObservation,
)
from ..utils.tool_helpers import (
    chroma_collection,
    get_ingestion,
    load_prompt,
    scan_images,
)
from . import register


# ---------- measure_lesions ----------
class _LesionList(BaseModel):
    visit: str
    lesions: list[RECISTLesion] = Field(default_factory=list)


_LESION_SCHEMA_HINT = """
{
  "visit": "v1",
  "lesions": [
    {"lesion_id": "L1", "location": "right frontal lobe",
     "longest_diameter_mm": 18.2, "visit": "v1", "target": true}
  ]
}
"""


@register("measure_lesions")
def measure_lesions(memory: WorkingMemory, visit: str = "v1", **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("recist.measure_lesions", pid=pid, tool="measure_lesions") as _t:
        ing = get_ingestion(memory)
        images = scan_images(ing, visit)
        if not images:
            return {"ok": False, "reason": f"no images for visit {visit}"}

        sys_msg = load_prompt("recist_system.md")
        prompt = (
            f"{sys_msg}\n\nVISIT: {visit}\n"
            f"Identify and measure target brain lesions on the MRI image(s). "
            f"Lesions smaller than {RECIST_MIN_LESION_MM:.0f} mm are NOT target.\n"
            f"Return ONE JSON object with this exact shape "
            f"(use visit='{visit}' for every lesion):\n{_LESION_SCHEMA_HINT}"
        )

        try:
            result = vision_json(prompt, images, _LesionList)
        except Exception as e:
            _t.meta["ok"] = False
            return {"ok": False, "error": str(e)[:200]}

        # Drop sub-threshold lesions and pin the visit field.
        kept = [
            l.model_copy(update={"visit": visit, "target": True})
            for l in result.lesions
            if l.longest_diameter_mm >= RECIST_MIN_LESION_MM
        ]

        bag = memory.get("recist_lesions") or {}
        if not isinstance(bag, dict):
            bag = {}
        bag[visit] = [l.model_dump(mode="json") for l in kept]
        memory.set("recist_lesions", bag)

        _t.meta["ok"] = True
        _t.meta["n_lesions"] = len(kept)
        return {
            "ok": True,
            "visit": visit,
            "n_lesions": len(kept),
            "sum_mm": round(sum(l.longest_diameter_mm for l in kept), 1),
        }


# ---------- classify_response ----------
def _sum_mm(lesions: list[dict[str, Any]]) -> float:
    return float(sum(l.get("longest_diameter_mm", 0.0) for l in lesions))


def _detect_new_lesion(memory: WorkingMemory) -> bool:
    """Scan all report text and vision impressions for new-lesion language.

    RECIST 1.1 rule: appearance of any new lesion = automatic PD,
    regardless of sum-of-diameters change.
    """
    ing = get_ingestion(memory)
    text_blob = "\n".join(
        (f.text or "") for f in ing.files
        if f.kind in {"mri_report", "discharge", "correlation", "timeline"}
    )
    vision_bag = memory.get(WorkingMemory.VISION) or {}
    if isinstance(vision_bag, dict):
        for v in vision_bag.values():
            obs = v if isinstance(v, VisionObservation) else VisionObservation.model_validate(v)
            text_blob += "\n" + (obs.impression or "")
            for fnd in obs.findings:
                text_blob += "\n" + (fnd.description or "")
    tl = text_blob.lower()
    return any(kw in tl for kw in NEW_LESION_KEYWORDS)


# ---------- RANO helpers (Task 4) ----------
# Approx bidirectional product when only longest_diameter is known: for a
# typical enhancing glioma the perpendicular axis is ~0.7× the longest axis
# (Ellipsoid with aspect ratio ≈ 1 : 0.7). We use this only as a fallback;
# Task 5 will replace it with `largest_axial_bidim_mm2` from the volumetric
# segmentation pipeline.
_BIDIM_APPROX_RATIO = 0.7

_STEROID_RE = re.compile(
    r"(dexamethasone|prednisone|prednisolone|methylprednisolone|decadron)"
    r"[^.\n]{0,80}?(\d+(?:\.\d+)?)\s*mg",
    re.IGNORECASE,
)
_T2_FLAIR_INC_RE = re.compile(
    r"(t2|flair)[^.\n]{0,60}?(increas|worsen|expand|new\b|progress)",
    re.IGNORECASE,
)
_T2_FLAIR_DEC_RE = re.compile(
    r"(t2|flair)[^.\n]{0,60}?(decreas|improv|reduc|smaller|resolv)",
    re.IGNORECASE,
)
_NEURO_WORSE_RE = re.compile(
    r"\b(new|worsen(?:ing|ed)?|increas(?:ing|ed)?)\s+"
    r"(seizure|headache|hemiparesis|hemiplegia|aphasia|"
    r"confusion|deficit|weakness|neurolog)",
    re.IGNORECASE,
)
_NEURO_BETTER_RE = re.compile(
    r"\b(improv(?:ed|ing)?|resolv(?:ed|ing)?)\s+"
    r"(seizure|headache|deficit|weakness|neurolog)",
    re.IGNORECASE,
)


def _is_brain_tumour(diagnosis: Any) -> bool:
    if not diagnosis:
        return False
    text = diagnosis if isinstance(diagnosis, str) else str(diagnosis)
    t = text.lower()
    return any(kw in t for kw in BRAIN_TUMOUR_KEYWORDS)


def _bidim_sum_mm2(lesions: list[dict[str, Any]]) -> float:
    """Sum of bidirectional products across all target lesions.

    Uses d² × ratio as the per-lesion ellipse approximation when no explicit
    perpendicular measurement is available.
    """
    total = 0.0
    for l in lesions:
        d = float(l.get("longest_diameter_mm", 0.0) or 0.0)
        if d <= 0:
            continue
        total += d * d * _BIDIM_APPROX_RATIO
    return total


def _collect_neuro_text(memory: WorkingMemory) -> str:
    ing = get_ingestion(memory)
    blob = "\n".join(
        (f.text or "") for f in ing.files
        if f.kind in {"mri_report", "discharge", "correlation", "timeline",
                      "pathology", "lab"}
    )
    vision_bag = memory.get(WorkingMemory.VISION) or {}
    if isinstance(vision_bag, dict):
        for v in vision_bag.values():
            obs = v if isinstance(v, VisionObservation) else VisionObservation.model_validate(v)
            blob += "\n" + (obs.impression or "")
            for fnd in obs.findings:
                blob += "\n" + (fnd.description or "")
    return blob


def _extract_steroid_dose_trend(text: str) -> tuple[float | None, str]:
    """Return (latest_dose_mg_per_day_or_None, change_label).

    Extremely simple: find all steroid dose mentions in order and compare
    first-vs-last. Returns one of: decreased, stable, increased, new, none.
    """
    matches = _STEROID_RE.findall(text)
    if not matches:
        return None, "none"
    doses = [float(m[1]) for m in matches]
    first, last = doses[0], doses[-1]
    if len(doses) == 1:
        return last, "new"
    if last < first * 0.9:
        return last, "decreased"
    if last > first * 1.1:
        return last, "increased"
    return last, "stable"


def _extract_t2_flair_change(text: str) -> str:
    if _T2_FLAIR_INC_RE.search(text):
        return "increased"
    if _T2_FLAIR_DEC_RE.search(text):
        return "decreased"
    if re.search(r"(t2|flair)", text, re.IGNORECASE):
        return "stable"
    return "unknown"


def _extract_neuro_status(text: str) -> str:
    worse = bool(_NEURO_WORSE_RE.search(text))
    better = bool(_NEURO_BETTER_RE.search(text))
    if worse and not better:
        return "worsened"
    if better and not worse:
        return "improved"
    if worse or better:
        return "stable"
    return "unknown"


def _classify_rano(
    base_prod: float,
    curr_prod: float,
    delta_pct: float | None,
    new_lesion: bool,
    t2_flair_change: str,
    steroid_change: str,
    neuro_status: str,
    has_measurable_current: bool,
) -> tuple[str, str]:
    """Apply simplified RANO 2010 rules → (response_code, rationale)."""
    # PD triggers
    if new_lesion:
        return "PD", "New enhancing lesion — automatic PD per RANO."
    if delta_pct is not None and delta_pct >= 0.25:
        return "PD", f"Bidirectional product increased {delta_pct*100:.1f}% (≥25%)."
    if t2_flair_change in {"increased", "new"}:
        return "PD", "Significant T2/FLAIR increase/new non-enhancing disease."
    if neuro_status == "worsened" and steroid_change in {"increased", "new"}:
        return "PD", "Clinical deterioration with escalating steroid dose."
    # CR — no enhancing disease + off steroids + clinically stable/improved
    if (not has_measurable_current
            and steroid_change in {"none", "decreased"}
            and neuro_status in {"stable", "improved"}):
        return "CR", "No measurable enhancing disease; off/tapering steroids; clinically stable."
    # PR — ≥50% decrease bidirectional product
    if delta_pct is not None and delta_pct <= -0.5:
        if steroid_change in {"stable", "decreased", "none"} and neuro_status != "worsened":
            return "PR", f"Bidirectional product decreased {abs(delta_pct)*100:.1f}% (≥50%)."
        return "SD", (
            f"Bidirectional product decreased {abs(delta_pct)*100:.1f}% but "
            "steroid/neuro criteria not met for PR."
        )
    # SD — default
    if delta_pct is None:
        return "NE", "Insufficient paired measurements for RANO delta."
    return "SD", f"Change of {delta_pct*100:.1f}% — neither PR nor PD thresholds met."


def _assess_rano(
    memory: WorkingMemory,
    baseline_v: str,
    current_v: str,
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
    new_lesion: bool,
) -> RANOAssessment:
    text = _collect_neuro_text(memory)
    base_prod = _bidim_sum_mm2(baseline)
    curr_prod = _bidim_sum_mm2(current)
    delta_pct: float | None
    if baseline_v == current_v or base_prod == 0.0:
        delta_pct = None
    else:
        delta_pct = (curr_prod - base_prod) / base_prod

    steroid_dose, steroid_change = _extract_steroid_dose_trend(text)
    t2_flair_change = _extract_t2_flair_change(text)
    neuro_status = _extract_neuro_status(text)
    has_measurable_current = bool(current)

    response, rationale = _classify_rano(
        base_prod, curr_prod, delta_pct,
        new_lesion, t2_flair_change, steroid_change, neuro_status,
        has_measurable_current,
    )

    return RANOAssessment(
        bidirectional_product_mm2=round(curr_prod, 1),
        baseline_product_mm2=round(base_prod, 1) if base_prod else None,
        delta_product_pct=round(delta_pct, 4) if delta_pct is not None else None,
        t2_flair_change=t2_flair_change,  # type: ignore[arg-type]
        corticosteroid_dose_mg_per_day=steroid_dose,
        corticosteroid_dose_change=steroid_change,  # type: ignore[arg-type]
        neurologic_status=neuro_status,  # type: ignore[arg-type]
        new_enhancing_lesion=new_lesion,
        nonmeasurable_disease_progression=(t2_flair_change in {"increased", "new"}),
        response=response,  # type: ignore[arg-type]
        criteria_used="RANO",
        rationale=rationale,
    )


def _record_diagnosis(memory: WorkingMemory) -> Any:
    rec = memory.get(WorkingMemory.RECORD)
    if rec is None:
        return None
    if isinstance(rec, PatientRecord):
        return rec.diagnosis
    if isinstance(rec, dict):
        return rec.get("diagnosis")
    try:
        return PatientRecord.model_validate(rec).diagnosis
    except Exception:
        return None


@register("classify_response")
def classify_response(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("recist.classify_response", pid=pid, tool="classify_response") as _t:
        bag = memory.get("recist_lesions") or {}
        if not isinstance(bag, dict) or not bag:
            return {"ok": False, "reason": "measure_lesions must run first"}

        visits = sorted(bag.keys())
        baseline_v = visits[0]
        current_v = visits[-1]
        baseline = bag[baseline_v]
        current = bag[current_v]

        base_sum = _sum_mm(baseline)
        curr_sum = _sum_mm(current)

        # RECIST 1.1 trigger 1: new lesion → automatic PD
        new_lesion = _detect_new_lesion(memory)

        if new_lesion:
            response = "PD"
            pct = (curr_sum - base_sum) / base_sum if base_sum else None
            rationale = "New lesion detected — automatic Progressive Disease per RECIST 1.1."
        elif baseline_v == current_v or base_sum == 0.0:
            response = "NE" if base_sum == 0.0 else "SD"
            pct = None
            rationale = (
                "Single visit only — no interval comparison possible."
                if baseline_v == current_v
                else "Baseline sum is zero — non-evaluable."
            )
        else:
            pct = (curr_sum - base_sum) / base_sum
            if not current:
                response = "CR"
                rationale = "No measurable target lesions on current visit."
            elif pct <= RECIST_PR_THRESHOLD:
                response = "PR"
                rationale = f"Sum of diameters decreased {abs(pct) * 100:.1f}% (>=30%)."
            elif pct >= RECIST_PD_THRESHOLD:
                response = "PD"
                rationale = f"Sum of diameters increased {pct * 100:.1f}% (>=20%)."
            else:
                response = "SD"
                rationale = f"Change of {pct * 100:.1f}% — neither PR nor PD."

        # RECIST 1.1 trigger 3: CR/PR requires confirmation scan ≥4 weeks
        confirmation_required = response in {"CR", "PR"}

        assessment = RECISTAssessment(
            baseline_sum_mm=base_sum if base_sum else None,
            current_sum_mm=curr_sum if curr_sum else None,
            pct_change=pct,
            response=response,
            lesions_baseline=[RECISTLesion.model_validate(l) for l in baseline],
            lesions_current=[RECISTLesion.model_validate(l) for l in current],
            rationale=rationale,
            new_lesion_detected=new_lesion,
            confirmation_required=confirmation_required,
        )
        memory.set(WorkingMemory.RECIST, assessment)

        # ---- RANO branch for brain tumours (Task 4) ----
        # Emit RANOAssessment alongside RECIST; downstream (Phase 4, synthesis)
        # prefers RANO when present, falling back to RECIST for non-brain sites.
        rano_payload: dict[str, Any] | None = None
        try:
            diagnosis = _record_diagnosis(memory)
            if _is_brain_tumour(diagnosis):
                rano = _assess_rano(
                    memory,
                    baseline_v=baseline_v,
                    current_v=current_v,
                    baseline=baseline,
                    current=current,
                    new_lesion=new_lesion,
                )
                memory.set(WorkingMemory.RANO, rano)
                rano_payload = {
                    "response": rano.response,
                    "bidirectional_product_mm2": rano.bidirectional_product_mm2,
                    "delta_product_pct": rano.delta_product_pct,
                    "t2_flair_change": rano.t2_flair_change,
                    "corticosteroid_dose_change": rano.corticosteroid_dose_change,
                    "neurologic_status": rano.neurologic_status,
                    "criteria_used": rano.criteria_used,
                }
                _t.meta["rano_response"] = rano.response
        except Exception as exc:  # best-effort: do NOT fail RECIST on RANO failure
            _t.meta["rano_error"] = f"{type(exc).__name__}: {exc}"[:160]

        _t.meta["ok"] = True
        result: dict[str, Any] = {
            "ok": True,
            "response": response,
            "pct_change": pct,
            "new_lesion_detected": new_lesion,
            "confirmation_required": confirmation_required,
            "baseline_sum_mm": base_sum,
            "current_sum_mm": curr_sum,
        }
        if rano_payload is not None:
            result["rano"] = rano_payload
        return result


# ---------- score_urgency ----------
def _hits(text: str, keywords: list[str]) -> list[str]:
    t = text.lower()
    return [k for k in keywords if k in t]


@register("score_urgency")
def score_urgency(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("recist.score_urgency", pid=pid, tool="score_urgency") as _t:
        ing = get_ingestion(memory)
        # Pool report text + any vision impressions for keyword scan.
        text_blob = "\n".join(
            (f.text or "") for f in ing.files
            if f.kind in {"mri_report", "discharge", "pathology", "lab",
                          "timeline", "correlation"}
        )
        vision_bag = memory.get(WorkingMemory.VISION) or {}
        if isinstance(vision_bag, dict):
            for v in vision_bag.values():
                obs = v if isinstance(v, VisionObservation) else VisionObservation.model_validate(v)
                text_blob += "\n" + (obs.impression or "")
                for fnd in obs.findings:
                    text_blob += "\n" + (fnd.description or "")

        critical = _hits(text_blob, URGENCY_KEYWORDS_CRITICAL)
        high = _hits(text_blob, URGENCY_KEYWORDS_HIGH)

        # Response-criteria escalation — prefer RANO (brain tumours) over RECIST.
        recist = memory.get(WorkingMemory.RECIST)
        recist_pd = False
        if recist is not None:
            r = recist if isinstance(recist, RECISTAssessment) else RECISTAssessment.model_validate(recist)
            recist_pd = r.response == "PD"
        rano = memory.get(WorkingMemory.RANO)
        if rano is not None:
            ro = rano if isinstance(rano, RANOAssessment) else RANOAssessment.model_validate(rano)
            if ro.response == "PD":
                recist_pd = True  # unified "progressive disease" flag for urgency

        score = 1
        drivers: list[str] = []
        if critical:
            score = 5
            drivers += [f"critical:{k}" for k in critical]
        elif recist_pd and high:
            score = 4
            drivers += ["recist:PD"] + [f"high:{k}" for k in high]
        elif recist_pd:
            score = 4
            drivers.append("recist:PD")
        elif high:
            score = 3
            drivers += [f"high:{k}" for k in high]
        else:
            score = 2

        level_map = {1: "routine", 2: "routine", 3: "soon", 4: "urgent", 5: "critical"}
        urgency = UrgencyAssessment(
            score=score,
            level=level_map[score],  # type: ignore[arg-type]
            drivers=drivers,
            rationale=(
                f"Critical keywords: {critical or 'none'}; "
                f"high keywords: {high or 'none'}; "
                f"RECIST PD: {recist_pd}."
            ),
        )
        memory.set(WorkingMemory.URGENCY, urgency)
        _t.meta["ok"] = True
        _t.meta["urgency"] = score
        return {
            "ok": True,
            "score": score,
            "level": urgency.level,
            "drivers": drivers[:8],
        }


# ---------- index_rag ----------
def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += max(1, size - overlap)
    return chunks


@register("index_rag")
def index_rag(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    pid = memory.patient_id
    with stage_timer("recist.index_rag", pid=pid, tool="index_rag") as _t:
        try:
            col = chroma_collection(pid)
        except Exception as e:
            return {"ok": False, "error": f"chroma init: {type(e).__name__}: {e}"[:200]}

        ing = get_ingestion(memory)
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        for f in ing.files:
            if not f.text:
                continue
            for idx, ch in enumerate(_chunk(f.text)):
                ids.append(f"{Path(f.path).stem}__{f.visit}__{idx}")
                docs.append(ch)
                metas.append({
                    "file": Path(f.path).name,
                    "visit": f.visit,
                    "kind": f.kind,
                    "chunk": idx,
                })

        if not docs:
            memory.set(WorkingMemory.RAG, {"n_chunks": 0, "collection": col.name})
            return {"ok": True, "n_chunks": 0}

        try:
            vectors = embed(docs)
        except Exception as e:
            return {"ok": False, "error": f"embed: {type(e).__name__}: {e}"[:200]}

        try:
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        except Exception as e:
            return {"ok": False, "error": f"chroma upsert: {type(e).__name__}: {e}"[:200]}

        memory.set(
            WorkingMemory.RAG,
            {"n_chunks": len(docs), "collection": col.name},
        )
        _t.meta["ok"] = True
        _t.meta["n_chunks"] = len(docs)

        # ── Phase 5.4 / Module 1 — LightRAG dual-write (background) ────────
        # Submit a graph-build job that runs LightRAG.insert(docs) in the
        # background. Sentinel transitions building → ready/failed; pharma
        # falls back to Chroma while building.
        lightrag_submitted = False
        try:
            from ..utils import lightrag_store, graph_worker
            if lightrag_store.is_available():
                wd = lightrag_store.working_dir_for_patient(pid)
                docs_snapshot = list(docs)
                graph_worker.submit_build(
                    pid, wd,
                    build_fn=lambda: lightrag_store.insert_chunks(pid, docs_snapshot),
                )
                lightrag_submitted = True
                log.info("recist.index_rag: %s — LightRAG background build queued (%d chunks)",
                         pid, len(docs_snapshot))
        except Exception as exc:
            log.warning("recist.index_rag: LightRAG submit failed (Chroma still upserted): %s", exc)

        return {
            "ok": True, "n_chunks": len(docs), "collection": col.name,
            "lightrag_submitted": lightrag_submitted,
        }
