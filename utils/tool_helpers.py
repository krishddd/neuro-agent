"""Shared helpers for tools/* modules.

Centralizes patterns that were duplicated across the five sub-agent files:
prompt loading (cached), ingestion access, dict-or-model coercion, and
per-visit image / report extraction from `IngestionResult`.

ChromaDB storage layout:
    chroma_db/
    ├── P001/                   ← separate DB per patient
    │   ├── chroma.sqlite3
    │   └── <hnsw_segment>/
    ├── P002/
    │   ├── chroma.sqlite3
    │   └── <hnsw_segment>/
    └── _shared/                ← drug interaction KB (shared across patients)
        ├── chroma.sqlite3
        └── <hnsw_segment>/
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from ..config import CHROMA_DIR, PROMPTS_DIR
from ..memory import WorkingMemory
from .schemas import IngestedFile, IngestionResult

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


@lru_cache(maxsize=32)
def load_prompt(name: str, default: str = "") -> str:
    try:
        return (PROMPTS_DIR / name).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def load_model(memory: WorkingMemory, key: str, schema: Type[M]) -> M | None:
    obj = memory.get(key)
    if obj is None:
        return None
    return obj if isinstance(obj, schema) else schema.model_validate(obj)


def get_ingestion(memory: WorkingMemory) -> IngestionResult:
    ing = load_model(memory, WorkingMemory.INGESTION, IngestionResult)
    if ing is None:
        raise RuntimeError("ingest must run before this tool")
    return ing


def files_for_visit(ing: IngestionResult, visit: str) -> list[IngestedFile]:
    return [f for f in ing.files if f.visit == visit]


def scan_images(ing: IngestionResult, visit: str) -> list[str]:
    return [
        f.image_path for f in ing.files
        if f.visit == visit and f.kind == "mri_image" and f.image_path
    ]


def report_text(ing: IngestionResult, visit: str) -> str:
    parts = [
        (f.text or "") for f in ing.files
        if f.visit == visit and f.kind == "mri_report" and f.text
    ]
    return "\n\n".join(parts).strip()


# ---------- Chroma — per-patient isolated databases ----------
#
# Each patient gets their own ChromaDB PersistentClient stored in:
#     chroma_db/<PATIENT_ID>/chroma.sqlite3
#
# This means:
#   - Each patient's vector data is in its own folder, clearly named
#   - No data mixing — P001's data is physically separate from P002's
#   - Easy to backup/delete a single patient's data
#   - The drug interaction KB lives in chroma_db/_shared/

_PATIENT_CLIENTS: dict[str, "chromadb.ClientAPI"] = {}  # noqa: F821 — chromadb imported lazily
_SHARED_CLIENT = None
_DRUG_KB_INDEXED = False


def _get_patient_chroma_client(pid: str):
    """Return a per-patient ChromaDB PersistentClient.

    Creates chroma_db/<PID>/ folder on first access.
    """
    global _PATIENT_CLIENTS
    pid_upper = pid.upper()
    if pid_upper in _PATIENT_CLIENTS:
        return _PATIENT_CLIENTS[pid_upper]

    import chromadb

    patient_db_dir = Path(CHROMA_DIR) / pid_upper
    patient_db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(patient_db_dir))
    _PATIENT_CLIENTS[pid_upper] = client
    log.info("chroma: opened patient DB → %s", patient_db_dir)
    return client


def _get_shared_chroma_client():
    """Return the shared ChromaDB client for drug interactions KB."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None:
        return _SHARED_CLIENT

    import chromadb

    shared_dir = Path(CHROMA_DIR) / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    _SHARED_CLIENT = chromadb.PersistentClient(path=str(shared_dir))
    log.info("chroma: opened shared DB → %s", shared_dir)
    return _SHARED_CLIENT


def chroma_collection(pid: str):
    """Return the per-patient Chroma collection from the patient's own DB.

    Storage: chroma_db/<PID>/chroma.sqlite3
    Collection name: "patient_data" (only one collection per DB, so name is simple)
    """
    client = _get_patient_chroma_client(pid)
    return client.get_or_create_collection(
        name="patient_data",
        metadata={"hnsw:space": "cosine"},
    )


def drug_interactions_collection():
    """Return (and lazily populate) the shared drug_interactions Chroma collection.

    Storage: chroma_db/_shared/chroma.sqlite3
    Collection name: "drug_interactions"

    On first call the drug_interaction_kb.json entries are embedded and
    upserted so pharma_agent can do semantic KB search.
    """
    global _DRUG_KB_INDEXED
    from ..config import DRUG_KB_PATH

    client = _get_shared_chroma_client()
    col = client.get_or_create_collection(
        name="drug_interactions",
        metadata={"hnsw:space": "cosine"},
    )

    if _DRUG_KB_INDEXED:
        return col

    import json
    from pathlib import Path as _Path
    kb_path = _Path(DRUG_KB_PATH)
    if not kb_path.exists():
        _DRUG_KB_INDEXED = True
        return col

    try:
        entries = json.loads(kb_path.read_text(encoding="utf-8")).get("interactions", [])
    except Exception:
        _DRUG_KB_INDEXED = True
        return col

    if not entries:
        _DRUG_KB_INDEXED = True
        return col

    # Only embed if the collection is empty (avoids re-embedding on restart).
    existing = col.count()
    if existing >= len(entries):
        _DRUG_KB_INDEXED = True
        return col

    from ..llm import embed as _embed
    docs, ids, metas = [], [], []
    for i, e in enumerate(entries):
        a = (e.get("drug_a") or "").strip()
        b = (e.get("drug_b") or "").strip()
        sev = (e.get("severity") or "none").lower()
        mech = (e.get("mechanism") or "")
        note = (e.get("clinical_note") or "")
        text = f"{a} + {b}: {sev}. {mech} {note}".strip()
        docs.append(text)
        ids.append(f"kb_{i}")
        metas.append({"drug_a": a, "drug_b": b, "severity": sev,
                      "mechanism": mech[:300], "clinical_note": note[:300]})

    try:
        vecs = _embed(docs)
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
        log.info("chroma: indexed %d drug KB entries into _shared/", len(docs))
    except Exception:
        pass  # best-effort; flat KB lookup remains the primary path

    _DRUG_KB_INDEXED = True
    return col
