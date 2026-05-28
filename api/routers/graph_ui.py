"""Phase 5.6 / Extra A — Visual Graph UI endpoints.

Two endpoints:

    GET /api/v1/graph/{patient_id}
        → Cytoscape-compatible JSON for the patient's LightRAG graph.
          Returns 404 when the graph hasn't been built (graph_worker
          sentinel != ``ready``) or when ``networkx`` / ``lightrag-hku``
          aren't installed.

    GET /api/v1/graph/{patient_id}/status
        → Returns the raw graph_worker sentinel JSON
          (``building`` | ``ready`` | ``failed`` | ``absent``) plus a
          short human summary so the viewer can show progress.

The viewer SPA at ``/ui/graph_viewer.html`` polls ``/status`` until
``ready`` then fetches the graph JSON.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ...config import LIGHTRAG_WORKING_DIR

log = logging.getLogger(__name__)

router = APIRouter()


def _safe_pid(pid: str) -> str:
    if not pid or "/" in pid or "\\" in pid or ".." in pid or not pid.isascii():
        raise HTTPException(400, "invalid patient_id — alphanumeric + hyphens only")
    return pid.strip().upper()


@router.get("/graph/{patient_id}/status")
def get_graph_status(patient_id: str) -> dict[str, Any]:
    """Return the LightRAG build sentinel for this patient."""
    pid = _safe_pid(patient_id)
    wd = Path(LIGHTRAG_WORKING_DIR) / pid

    from ...utils import graph_worker
    status = graph_worker.read_status(wd)

    state = status.get("status", "absent")
    summary = {
        "ready":    "Graph ready — viewer will render now.",
        "building": "Graph build in progress (LLM extracting entities + relations).",
        "failed":   "Graph build failed; viewer falls back to ChromaDB.",
        "absent":   "No graph yet — run the pipeline once to populate.",
    }.get(state, state)

    return {
        "patient_id": pid,
        "state":      state,
        "summary":    summary,
        "raw":        status,
    }


@router.get("/graph/{patient_id}")
def get_graph(patient_id: str) -> dict[str, Any]:
    """Return Cytoscape elements for the patient's LightRAG graph."""
    pid = _safe_pid(patient_id)
    wd = Path(LIGHTRAG_WORKING_DIR) / pid

    from ...utils import graph_worker
    from ...utils.graph_export import GRAPHML_FILENAME, graphml_to_cytoscape

    status = graph_worker.read_status(wd)
    if status.get("status") != "ready":
        # Surface 404 with structured detail so the viewer can poll status.
        raise HTTPException(
            status_code=404,
            detail={
                "patient_id":   pid,
                "graph_status": status.get("status", "absent"),
                "message": (
                    "LightRAG graph not ready — poll /api/v1/graph/{pid}/status "
                    "until state == 'ready'."
                ),
            },
        )

    graphml_path = wd / GRAPHML_FILENAME
    elements = graphml_to_cytoscape(graphml_path)

    if elements.get("unavailable"):
        raise HTTPException(
            status_code=404,
            detail={
                "patient_id": pid,
                "reason":     elements.get("reason", "unknown"),
                "message":    "graph file present but could not be loaded",
            },
        )

    return {
        "patient_id":      pid,
        "graphml_path":    str(graphml_path),
        "n_nodes_total":   elements.get("n_nodes_total"),
        "n_edges_total":   elements.get("n_edges_total"),
        "n_nodes_emitted": elements.get("n_nodes_emitted"),
        "n_edges_emitted": elements.get("n_edges_emitted"),
        "trimmed":         elements.get("trimmed", False),
        "elements": {
            "nodes": elements["nodes"],
            "edges": elements["edges"],
        },
    }
