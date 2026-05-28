"""Q&A chatbot sub-agent.

The chatbot is invoked by the API/CLI through `answer()`. It runs a small
tool-call loop with five tools (recall_memory, search_records,
get_recist_trend, get_current_meds, get_interactions), retrieves citations
from Chroma, validates them post-hoc, and returns a `QAAnswer`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from ..config import (
    CHAT_REQUIRE_CITATIONS,
    DISCLAIMER,
    MODEL_PRIMARY,
    RAG_TOP_K,
)
from ..llm import embed, json_call, stream as llm_stream
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.schemas import (
    InteractionReport,
    MedicationList,
    QAAnswer,
    QACitation,
    RECISTAssessment,
)
from ..utils.tool_helpers import chroma_collection, load_prompt

# Kept for streaming mode where JSON is not available.
_CITATION_RE = re.compile(
    r"\[source:\s*([^,\]]+)(?:,\s*visit\s*([^,\]]+))?(?:,\s*chunk\s*(\d+))?\s*\]",
    re.IGNORECASE,
)


# ---------- retrieval ----------
def search_chunks(pid: str, query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    try:
        col = chroma_collection(pid)
        vecs = embed([query])
        if not vecs:  # embed failed silently (empty result) — fail open.
            return []
        res = col.query(query_embeddings=[vecs[0]], n_results=top_k)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "chat_agent.search_chunks: %s (%s)", type(exc).__name__, str(exc)[:120]
        )
        return []

    out: list[dict[str, Any]] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0] if "distances" in res else [None] * len(docs)
    for d, m, dist in zip(docs, metas, dists):
        score = (1.0 - float(dist)) if dist is not None else None
        out.append({
            "text": d,
            "file": (m or {}).get("file", "unknown"),
            "visit": (m or {}).get("visit"),
            "chunk": (m or {}).get("chunk"),
            "kind": (m or {}).get("kind"),
            "score": score,
        })
    return out


# ---------- chat tools (registered for LLM tool calling) ----------
def _tool_recall(memory: WorkingMemory, key: str = "", **_: Any) -> dict[str, Any]:
    val = memory.get(key)
    if val is None:
        return {"ok": False, "reason": f"no key {key}"}
    if hasattr(val, "model_dump"):
        return {"ok": True, "data": val.model_dump(mode="json")}
    return {"ok": True, "data": val}


def _tool_search(memory: WorkingMemory, query: str = "", top_k: int = RAG_TOP_K, **_: Any) -> dict[str, Any]:
    hits = search_chunks(memory.patient_id, query, top_k=top_k)
    return {
        "ok": True,
        "n": len(hits),
        "hits": [
            {
                "file": h["file"], "visit": h["visit"], "chunk": h["chunk"],
                "score": h["score"], "snippet": (h["text"] or "")[:400],
            }
            for h in hits
        ],
    }


def _tool_recist(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    r = memory.get(WorkingMemory.RECIST)
    if r is None:
        return {"ok": False}
    r = r if isinstance(r, RECISTAssessment) else RECISTAssessment.model_validate(r)
    return {
        "ok": True,
        "response": r.response,
        "pct_change": r.pct_change,
        "baseline_sum_mm": r.baseline_sum_mm,
        "current_sum_mm": r.current_sum_mm,
        "rationale": r.rationale,
    }


def _tool_meds(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    m = memory.get(WorkingMemory.MEDICATIONS)
    if m is None:
        return {"ok": False}
    m = m if isinstance(m, MedicationList) else MedicationList.model_validate(m)
    return {
        "ok": True,
        "current": [x.model_dump(mode="json") for x in m.current],
        "historical": [x.model_dump(mode="json") for x in m.historical],
    }


def _tool_interactions(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    i = memory.get(WorkingMemory.INTERACTIONS)
    if i is None:
        return {"ok": False}
    i = i if isinstance(i, InteractionReport) else InteractionReport.model_validate(i)
    return {
        "ok": True,
        "highest_severity": i.highest_severity,
        "n": len(i.interactions),
        "flags": i.flags,
    }


_LOCAL_TOOLS: dict[str, Any] = {
    "recall_memory": _tool_recall,
    "search_records": _tool_search,
    "get_recist_trend": _tool_recist,
    "get_current_meds": _tool_meds,
    "get_interactions": _tool_interactions,
}


# ---------- core answer() ----------
def _system() -> str:
    return load_prompt("chat_system.md", default="You are a clinical Q&A assistant.")


def _initial_context(memory: WorkingMemory, question: str) -> dict[str, Any]:
    """Pre-fetch a small retrieval pass so the model has citations on turn 1."""
    hits = search_chunks(memory.patient_id, question, top_k=RAG_TOP_K)
    return {
        "memory_snapshot": memory.snapshot_for_llm(),
        "retrieved": [
            {"file": h["file"], "visit": h["visit"], "chunk": h["chunk"],
             "score": h["score"], "snippet": (h["text"] or "")[:400]}
            for h in hits
        ],
    }


def _validate_structured_sources(
    sources: list[QACitation],
    allowed_files: set[str],
) -> list[QACitation]:
    """Filter model-supplied QACitation objects to only those from retrieved files.

    This is deterministic: we compare file names directly against the set of
    files that were actually retrieved from Chroma for this question. No regex.
    """
    if not allowed_files:
        return list(sources)
    return [s for s in sources if s.file in allowed_files]


def _fallback_inline_citations(answer_text: str, allowed_files: set[str]) -> list[QACitation]:
    """Regex fallback used only when the model returned zero structured sources."""
    found: list[QACitation] = []
    for m in _CITATION_RE.finditer(answer_text):
        f = (m.group(1) or "").strip()
        v = (m.group(2) or "").strip() or None
        c = int(m.group(3)) if m.group(3) else None
        if f in allowed_files or not allowed_files:
            found.append(QACitation(file=f, visit=v, chunk=c))
    return found


def answer(
    memory: WorkingMemory,
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    require_citations: bool = CHAT_REQUIRE_CITATIONS,
) -> QAAnswer:
    pid = memory.patient_id
    with stage_timer("chat.answer", pid=pid, tool="chat") as _t:
        ctx = _initial_context(memory, question)
        allowed_files = {h["file"] for h in ctx["retrieved"]}

        sys_msg = _system()
        user_msg = (
            f"PATIENT_ID: {pid}\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT_JSON:\n{json.dumps(ctx, default=str)}\n\n"
            f"INSTRUCTIONS: In your JSON response, include every retrieved file "
            f"you used in the `sources` list as QACitation objects with the exact "
            f"filename from CONTEXT_JSON. Do NOT invent filenames.\n"
            f"DISCLAIMER: {DISCLAIMER}\n"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_msg},
            *(history or []),
            {"role": "user", "content": user_msg},
        ]

        try:
            qa = json_call(messages, QAAnswer, model=MODEL_PRIMARY)
        except Exception:
            return QAAnswer(
                answer="I couldn't generate a grounded answer for this question.",
                sources=[],
                confidence="low",
                disclaimer=DISCLAIMER,
            )

        # Primary: validate structured sources directly against retrieved file names.
        validated = _validate_structured_sources(list(qa.sources or []), allowed_files)

        # Fallback: if model returned no valid structured sources, try inline regex.
        if not validated:
            validated = _fallback_inline_citations(qa.answer, allowed_files)

        if require_citations and not validated:
            _t.meta["confidence"] = "low"
            return QAAnswer(
                answer="I don't have enough cited information to answer that for this patient.",
                sources=[],
                confidence="low",
                disclaimer=DISCLAIMER,
            )

        out = QAAnswer(
            answer=qa.answer,
            sources=validated,
            confidence=qa.confidence or "medium",
            disclaimer=DISCLAIMER,
        )
        _t.meta["confidence"] = out.confidence
        return out


# ---------- streaming variant ----------
def stream_answer(
    memory: WorkingMemory,
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Token stream for WebSocket clients.

    Streams the raw model output (no JSON enforcement). Final QAAnswer with
    citations should be obtained from `answer()` after streaming finishes.
    """
    ctx = _initial_context(memory, question)
    sys_msg = _system()
    user_msg = (
        f"PATIENT_ID: {memory.patient_id}\nQUESTION: {question}\n\n"
        f"CONTEXT_JSON:\n{json.dumps(ctx, default=str)}\n"
        f"DISCLAIMER: {DISCLAIMER}\n"
        "Reply in plain text with inline [source: file, visit v, chunk i] "
        "citations. Be concise."
    )
    messages = [
        {"role": "system", "content": sys_msg},
        *(history or []),
        {"role": "user", "content": user_msg},
    ]
    yield from llm_stream(messages, model=MODEL_PRIMARY)
