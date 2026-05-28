"""Agentic RAG penalty scorer for the SMBO drug-interaction safety gate.

Design goals:
  - Fully synchronous — safe to call from Typer CLI or FastAPI background tasks
    without async/await complexity or event-loop conflicts.
  - Pre-built O(1) cache from drug_interaction_kb.json at import time, so the
    tight 600-query SMBO loop never hits disk for known drug pairs.
  - ThreadPoolExecutor for parallel ChromaDB fallback on cache misses only.
  - combo_drug == "none" short-circuits immediately (no penalty).

Severity → penalty mapping:
    none             →  0.00
    minor            →  0.05
    moderate         →  0.30
    major            →  0.80
    contraindicated  →  999.0   (effectively bans the candidate)
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Severity → numeric penalty ─────────────────────────────────────────────────
SEVERITY_PENALTY: dict[str, float] = {
    "none":            0.00,
    "minor":           0.05,
    "moderate":        0.30,
    "major":           0.80,
    "contraindicated": 999.0,
}

# ── Brand-name → generic aliases (lowercase) ──────────────────────────────────
# Keeps the cache consistent regardless of how drugs are named in the KB vs
# the SMBO search space.
_DRUG_ALIASES: dict[str, str] = {
    # Chemotherapy
    "temodar":        "temozolomide",
    "tmz":            "temozolomide",
    "avastin":        "bevacizumab",
    "herceptin":      "trastuzumab",
    "taxol":          "paclitaxel",
    "abraxane":       "nab-paclitaxel",
    "platinol":       "cisplatin",
    "paraplatin":     "carboplatin",
    "eloxatin":       "oxaliplatin",
    "camptosar":      "irinotecan",
    "adriamycin":     "doxorubicin",
    "ifex":           "ifosfamide",
    "cytoxan":        "cyclophosphamide",
    "methotrexate":   "methotrexate",
    # Anti-epileptics / corticosteroids
    "depakote":       "valproic acid",
    "depakene":       "valproic acid",
    "valproate":      "valproic acid",
    "keppra":         "levetiracetam",
    "decadron":       "dexamethasone",
    "medrol":         "methylprednisolone",
    # Targeted / immunotherapy
    "keytruda":       "pembrolizumab",
    "opdivo":         "nivolumab",
    "yervoy":         "ipilimumab",
    "tagrisso":       "osimertinib",
    "tarceva":        "erlotinib",
    "zelboraf":       "vemurafenib",
    "gleevec":        "imatinib",
    # Anticoagulants
    "coumadin":       "warfarin",
    "xarelto":        "rivaroxaban",
    "eliquis":        "apixaban",
    "pradaxa":        "dabigatran",
    # Supportive
    "zofran":         "ondansetron",
    "neupogen":       "filgrastim",
    "aranesp":        "darbepoetin",
}


def _normalize_drug_name(name: str) -> str:
    """Lowercase, strip whitespace, resolve brand→generic aliases."""
    n = name.strip().lower()
    return _DRUG_ALIASES.get(n, n)


def _canonical_key(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Sorted canonical key so (A,B) == (B,A)."""
    a = _normalize_drug_name(drug_a)
    b = _normalize_drug_name(drug_b)
    return tuple(sorted([a, b]))  # type: ignore[return-value]


# ── Pre-built O(1) penalty cache ───────────────────────────────────────────────
_PENALTY_CACHE: dict[tuple[str, str], float] = {}


def _build_cache_from_kb() -> None:
    """Populate _PENALTY_CACHE from drug_interaction_kb.json at import time.

    This is O(n_interactions) — executed once, never repeated.
    Any (drug_a, drug_b) pair found here will never touch ChromaDB.
    """
    global _PENALTY_CACHE

    from ..config import DRUG_KB_PATH  # avoid circular at module level

    if not DRUG_KB_PATH.exists():
        log.warning("rag_penalty: KB not found at %s — cache empty", DRUG_KB_PATH)
        return

    try:
        kb = json.loads(DRUG_KB_PATH.read_text(encoding="utf-8"))
        interactions: list[dict[str, Any]] = kb.get("interactions", [])
        count = 0
        for entry in interactions:
            da = entry.get("drug_a", "")
            db = entry.get("drug_b", "")
            sev = entry.get("severity", "none").lower()
            if da and db:
                key = _canonical_key(da, db)
                penalty = SEVERITY_PENALTY.get(sev, 0.0)
                # Keep worst (highest) penalty for duplicate pairs
                if _PENALTY_CACHE.get(key, -1.0) < penalty:
                    _PENALTY_CACHE[key] = penalty
                count += 1
        log.info("rag_penalty: KB cache built — %d interactions, %d unique pairs",
                 count, len(_PENALTY_CACHE))
    except Exception as exc:
        log.warning("rag_penalty: failed to build KB cache: %s", exc)


# Build cache at module-import time (fast — 12 entries in current KB).
_build_cache_from_kb()


# ── ChromaDB semantic fallback ─────────────────────────────────────────────────

def _chroma_lookup(drug_a: str, drug_b: str) -> float:
    """Synchronous ChromaDB semantic query for cache-miss drug pairs.

    Embeds the query once and calls collection.query() with n_results=1.
    Caches result in _PENALTY_CACHE before returning.
    Distance threshold 0.25: if closest document is farther, return 0.0
    (unknown pair = no penalty).
    """
    key = _canonical_key(drug_a, drug_b)

    try:
        import chromadb  # type: ignore

        from ..config import CHROMA_DIR, CHROMA_COLLECTION_FMT  # noqa: F401

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Try to find a drug-interactions collection (built during ingest phase).
        collection_names = [c.name for c in client.list_collections()]
        target = next(
            (n for n in collection_names if "drug" in n.lower() or "interact" in n.lower()),
            collection_names[0] if collection_names else None,
        )
        if target is None:
            _PENALTY_CACHE[key] = 0.0
            return 0.0

        col = client.get_collection(target)
        query_text = f"drug interaction between {key[0]} and {key[1]}"
        results = col.query(query_texts=[query_text], n_results=1)

        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        if not distances or distances[0] > 0.25:
            # No close semantic match → no known interaction
            _PENALTY_CACHE[key] = 0.0
            return 0.0

        # Infer severity from document text heuristics
        doc_text = (documents[0] or "").lower()
        penalty = 0.0
        if "contraindicated" in doc_text:
            penalty = SEVERITY_PENALTY["contraindicated"]
        elif "major" in doc_text or "severe" in doc_text:
            penalty = SEVERITY_PENALTY["major"]
        elif "moderate" in doc_text:
            penalty = SEVERITY_PENALTY["moderate"]
        elif "minor" in doc_text or "mild" in doc_text:
            penalty = SEVERITY_PENALTY["minor"]

        _PENALTY_CACHE[key] = penalty
        return penalty

    except Exception as exc:
        log.debug("rag_penalty: chroma_lookup failed for (%s, %s): %s",
                  key[0], key[1], exc)
        _PENALTY_CACHE[key] = 0.0
        return 0.0


def _lightrag_lookup(drug_a: str, drug_b: str) -> float | None:
    """Phase 5.4 / Module 1 — Shared drug-KG lookup (best-effort).

    Consults the cross-patient LightRAG graph at
    ``chroma_db/_lightrag_shared/`` if a build has finished. Returns
    a penalty value when the graph yields a confident severity word, or
    ``None`` when LightRAG is unavailable / graph not built / answer
    inconclusive — caller falls through to Chroma in those cases.

    The shared graph is currently bootstrapped lazily; until something
    populates it, this path is a fast no-op that cleanly returns ``None``.
    """
    try:
        from .. import config
        from . import graph_worker, lightrag_store
        if not lightrag_store.is_available():
            return None
        wd = config.LIGHTRAG_SHARED_WORKING_DIR
        if not graph_worker.is_ready(wd):
            return None
        # Build a transient instance pointed at the shared dir.
        rag = lightrag_store._build_instance(wd)  # internal helper, sentinel-checked above
        if rag is None:
            return None
        from lightrag import QueryParam  # type: ignore
        answer = (rag.query(
            f"What is the clinical severity of the interaction between {drug_a} and {drug_b}?",
            param=QueryParam(mode="hybrid"),
        ) or "").lower()
        if not answer:
            return None
        if "contraindicated" in answer:
            return SEVERITY_PENALTY["contraindicated"]
        if "major" in answer or "severe" in answer:
            return SEVERITY_PENALTY["major"]
        if "moderate" in answer:
            return SEVERITY_PENALTY["moderate"]
        if "minor" in answer or "mild" in answer:
            return SEVERITY_PENALTY["minor"]
        return None  # answer present but no severity word → defer
    except Exception as exc:
        log.debug("rag_penalty: lightrag_lookup failed for (%s, %s): %s",
                  drug_a, drug_b, exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def rag_penalty(primary_drug: str, combo_drug: str) -> float:
    """Return the interaction penalty for a (primary, combo) drug pair.

    Fully synchronous and safe to call from any context (sync or ASGI).

    Steps:
      1. combo_drug == "none"  → 0.0 (no interaction possible)
      2. Check _PENALTY_CACHE  → return if hit  (O(1))
      3. _chroma_lookup()      → cache result → return
      4. No match              → 0.0 (unknown = no penalty)
    """
    if not combo_drug or combo_drug.lower() in ("none", ""):
        return 0.0

    key = _canonical_key(primary_drug, combo_drug)

    # Cache hit
    if key in _PENALTY_CACHE:
        return _PENALTY_CACHE[key]

    # Phase 5.4 / Module 1 — Try LightRAG shared graph first; fall back
    # to Chroma when it returns None (unavailable, not-yet-built, or
    # inconclusive answer).
    pen = _lightrag_lookup(primary_drug, combo_drug)
    if pen is not None:
        _PENALTY_CACHE[key] = pen
        return pen

    # ChromaDB fallback
    return _chroma_lookup(primary_drug, combo_drug)


def preseed_penalty_cache(current_drug_names: list[str], smbo_drug_list: list[str]) -> None:
    """Pre-compute (and cache) all (current_drug × smbo_drug) pairs.

    Called by `extract_patient_state()` (sub-step 4a) so the SMBO loop
    never blocks on cache misses for clinically critical pairs.

    Uses ThreadPoolExecutor(max_workers=4) for parallel ChromaDB lookups
    when pairs are absent from the KB cache.
    """
    miss_pairs: list[tuple[str, str]] = []
    for cur in current_drug_names:
        for smbo in smbo_drug_list:
            key = _canonical_key(cur, smbo)
            if key not in _PENALTY_CACHE:
                miss_pairs.append((cur, smbo))

    if not miss_pairs:
        log.info("rag_penalty: preseed — all %d pairs in cache",
                 len(current_drug_names) * len(smbo_drug_list))
        return

    log.info("rag_penalty: preseed — %d cache misses, launching parallel lookup",
             len(miss_pairs))

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_chroma_lookup, a, b): (a, b)
                for a, b in miss_pairs
            }
            for fut in as_completed(futures, timeout=30):
                pair = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    log.debug("rag_penalty: preseed lookup failed for %s: %s", pair, exc)
    except Exception as exc:
        log.warning("rag_penalty: preseed failed (falling back to sequential): %s", exc)
        for a, b in miss_pairs:
            _chroma_lookup(a, b)


def batch_rag_penalties(candidates: list[dict]) -> list[float]:
    """Score a batch of SMBO candidates synchronously.

    Each candidate dict must have keys 'primary_drug' and 'combo_drug'.
    Cache hits are O(1); misses use ThreadPoolExecutor(max_workers=4).

    Returns a list of float penalties indexed to candidates.
    """
    if not candidates:
        return []

    results: list[float | None] = [None] * len(candidates)
    miss_indices: list[int] = []

    # First pass: cache hits
    for i, cand in enumerate(candidates):
        pd = cand.get("primary_drug", "")
        cd = cand.get("combo_drug", "none")
        key = _canonical_key(pd, cd)
        if not cd or cd.lower() in ("none", ""):
            results[i] = 0.0
        elif key in _PENALTY_CACHE:
            results[i] = _PENALTY_CACHE[key]
        else:
            miss_indices.append(i)

    if not miss_indices:
        return results  # type: ignore[return-value]

    # Second pass: parallel ChromaDB for misses
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(miss_indices))) as pool:
            future_map = {
                pool.submit(
                    _chroma_lookup,
                    candidates[i]["primary_drug"],
                    candidates[i]["combo_drug"],
                ): i
                for i in miss_indices
            }
            for fut in as_completed(future_map, timeout=30):
                idx = future_map[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = 0.0
    except Exception as exc:
        log.warning("rag_penalty: batch parallel lookup failed (%s), using sequential", exc)
        for i in miss_indices:
            results[i] = rag_penalty(
                candidates[i]["primary_drug"], candidates[i]["combo_drug"]
            )

    # Ensure no None remains (safety net)
    return [r if r is not None else 0.0 for r in results]
