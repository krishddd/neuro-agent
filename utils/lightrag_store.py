"""Phase 5.4 / Module 1 — LightRAG (GraphRAG) wrapper.

Thin facade over the ``lightrag-hku`` package, configured to use the local
Ollama stack (``qwen3:14b`` + ``nomic-embed-text``) so it shares
embeddings with the existing ChromaDB index.

Graceful degrade
----------------
If the ``lightrag`` package isn't installed (or fails to initialise), the
module sets ``LIGHTRAG_AVAILABLE = False`` and every public function
returns a no-op result. ``recist_agent.index_rag`` and ``pharma_agent``
guard on this flag so the pipeline stays green Chroma-only.

Per-patient layout
------------------
    chroma_db/_lightrag/<pid>/
        graph_chunk_entity_relation.graphml   # the KG itself
        kv_store_*.json                       # LightRAG internals
        .build_status.json                    # graph_worker sentinel

A shared drug-knowledge graph lives at ``chroma_db/_lightrag_shared/``
(populated lazily; reserved for future cross-patient drug ontology).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from .. import config

log = logging.getLogger(__name__)

# ── lightrag import: fail soft ─────────────────────────────────────────────────
try:
    from lightrag import LightRAG, QueryParam  # type: ignore
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete  # type: ignore
    from lightrag.utils import EmbeddingFunc  # type: ignore
    LIGHTRAG_AVAILABLE = True
except Exception as _exc:  # ImportError or transitive init failure
    LightRAG = None         # type: ignore
    QueryParam = None       # type: ignore
    ollama_model_complete = None  # type: ignore
    ollama_embed = None     # type: ignore
    EmbeddingFunc = None    # type: ignore
    LIGHTRAG_AVAILABLE = False
    log.info("lightrag_store: lightrag-hku not available (%s) — Chroma-only mode", _exc)


# ── per-patient + shared instance cache ───────────────────────────────────
# MULTI-PATIENT-FIX: capped LRU so a long-running server processing
# hundreds of patients doesn't accumulate LightRAG instances forever.
# Each instance pins ~50–200 MB depending on graph size, so leaving them
# unbounded was a slow memory leak.
from collections import OrderedDict

_INSTANCE_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_INSTANCE_CACHE_MAX = 32


def is_available() -> bool:
    """True when both the package is importable AND LIGHTRAG_ENABLED is set."""
    return bool(LIGHTRAG_AVAILABLE and getattr(config, "LIGHTRAG_ENABLED", False))


def working_dir_for_patient(patient_id: str) -> Path:
    """Per-patient LightRAG working directory."""
    return Path(config.LIGHTRAG_WORKING_DIR) / patient_id


def _build_instance(working_dir: Path) -> Optional[Any]:
    """Construct a LightRAG instance with Ollama LLM + embedder.

    Returns None if construction fails (logged); caller treats as no-op.
    """
    if not LIGHTRAG_AVAILABLE:
        return None
    try:
        working_dir.mkdir(parents=True, exist_ok=True)
        host = config.LIGHTRAG_OLLAMA_HOST
        llm_model = config.LIGHTRAG_LLM_MODEL
        embed_model = config.LIGHTRAG_EMBED_MODEL

        # Embedding function — must match Chroma's nomic-embed-text dim (768).
        embed_fn = EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model, host=host,
            ),
        )
        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model,
            llm_model_kwargs={"host": host, "options": {"num_ctx": 8192}},
            embedding_func=embed_fn,
        )
        return rag
    except Exception as exc:
        log.error("lightrag_store: instance build failed for %s: %s",
                  working_dir, exc)
        return None


def get_instance(patient_id: str) -> Optional[Any]:
    """Return a cached per-patient LightRAG instance (or None on failure)."""
    if not is_available():
        return None
    if patient_id in _INSTANCE_CACHE:
        # Refresh recency in the LRU.
        _INSTANCE_CACHE.move_to_end(patient_id)
        return _INSTANCE_CACHE[patient_id]
    inst = _build_instance(working_dir_for_patient(patient_id))
    if inst is not None:
        _INSTANCE_CACHE[patient_id] = inst
        # Evict oldest entries when over the cap.
        while len(_INSTANCE_CACHE) > _INSTANCE_CACHE_MAX:
            old_pid, _ = _INSTANCE_CACHE.popitem(last=False)
            log.info("lightrag_store: evicted cached instance for %s (LRU cap)", old_pid)
    return inst


# ── Public ingest + query ─────────────────────────────────────────────────────

def insert_chunks(patient_id: str, chunks: Iterable[str]) -> int:
    """Synchronously insert text chunks into the patient's LightRAG graph.

    Intended to be called from inside a ``graph_worker`` background job.
    Returns the number of chunks ingested (0 if LightRAG unavailable).
    """
    rag = get_instance(patient_id)
    if rag is None:
        return 0
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        return 0
    try:
        rag.insert(chunks)
        return len(chunks)
    except Exception as exc:
        log.error("lightrag_store: insert failed for %s: %s", patient_id, exc)
        raise


def query_hybrid(patient_id: str, prompt: str, mode: str = "hybrid") -> str:
    """Run a hybrid LightRAG query (KG paths + chunk neighbours).

    Returns plain-text response; empty string when LightRAG is unavailable
    or the per-patient graph hasn't been built yet (sentinel != ready).
    """
    if not is_available():
        return ""
    # Defer the import so the sentinel module load stays cheap.
    from . import graph_worker
    if not graph_worker.is_ready(working_dir_for_patient(patient_id)):
        return ""
    rag = get_instance(patient_id)
    if rag is None or QueryParam is None:
        return ""
    try:
        return str(rag.query(prompt, param=QueryParam(mode=mode)) or "")
    except Exception as exc:
        log.warning("lightrag_store: query failed for %s: %s", patient_id, exc)
        return ""


__all__ = [
    "LIGHTRAG_AVAILABLE",
    "is_available",
    "working_dir_for_patient",
    "get_instance",
    "insert_chunks",
    "query_hybrid",
]
