"""Phase 5.6 / Extra A — LightRAG GraphML → Cytoscape.js converter.

Reads LightRAG's ``graph_chunk_entity_relation.graphml`` (NetworkX-
compatible) and emits the JSON shape Cytoscape.js expects:

    {
      "nodes": [{"data": {"id": "...", "label": "...", "type": "..."}}, ...],
      "edges": [{"data": {"id": "e1", "source": "...", "target": "...",
                          "label": "...", "weight": 1.0}}, ...]
    }

Graceful degrade:
    * ``networkx`` not installed → return ``{"nodes": [], "edges": [], "unavailable": True}``
    * GraphML file missing → ``{"nodes": [], "edges": [], "unavailable": True}``
    * Parse error → logged, empty result returned

Bounded output: very large patient graphs are capped to
``MAX_NODES`` / ``MAX_EDGES`` (defaults 500 / 2000) so the browser
viewer stays responsive. The cap drops lowest-degree nodes first
to preserve the densest causal sub-graph.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GRAPHML_FILENAME = "graph_chunk_entity_relation.graphml"
MAX_NODES = 500
MAX_EDGES = 2000


def _empty_unavailable(reason: str) -> dict[str, Any]:
    return {"nodes": [], "edges": [], "unavailable": True, "reason": reason}


def graphml_to_cytoscape(
    graphml_path: Path | str,
    *,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> dict[str, Any]:
    """Parse a LightRAG GraphML and return Cytoscape elements.

    Caps node count by trimming the lowest-degree nodes when the graph
    exceeds ``max_nodes``; edges referencing trimmed nodes are dropped.
    """
    path = Path(graphml_path)
    if not path.exists():
        return _empty_unavailable(f"file_not_found: {path.name}")

    try:
        import networkx as nx  # type: ignore
    except ImportError:
        return _empty_unavailable("networkx_not_installed")

    try:
        g = nx.read_graphml(str(path))
    except Exception as exc:
        log.warning("graph_export: read_graphml failed for %s: %s", path, exc)
        return _empty_unavailable(f"parse_error: {type(exc).__name__}")

    # ── trim by degree if oversize ────────────────────────────────────────
    n_nodes_total = g.number_of_nodes()
    n_edges_total = g.number_of_edges()
    trimmed = False
    if n_nodes_total > max_nodes:
        # Keep top-degree nodes; build a subgraph view.
        degrees = sorted(g.degree, key=lambda kv: kv[1], reverse=True)
        keep = {node for node, _ in degrees[:max_nodes]}
        g = g.subgraph(keep).copy()
        trimmed = True

    # ── nodes ─────────────────────────────────────────────────────────────
    nodes: list[dict[str, Any]] = []
    for node_id, attrs in g.nodes(data=True):
        # LightRAG entity attributes vary — try common keys, fall through.
        label = (
            attrs.get("entity_name")
            or attrs.get("name")
            or attrs.get("label")
            or str(node_id)
        )
        ntype = (
            attrs.get("entity_type")
            or attrs.get("type")
            or "entity"
        )
        nodes.append({"data": {
            "id": str(node_id),
            "label": str(label)[:80],
            "type": str(ntype)[:40],
            "description": str(attrs.get("description", ""))[:240],
        }})

    # ── edges ─────────────────────────────────────────────────────────────
    edges: list[dict[str, Any]] = []
    edge_id = 0
    for u, v, attrs in g.edges(data=True):
        if len(edges) >= max_edges:
            break
        rel = (
            attrs.get("description")
            or attrs.get("relation")
            or attrs.get("label")
            or ""
        )
        try:
            weight = float(attrs.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        edges.append({"data": {
            "id": f"e{edge_id}",
            "source": str(u),
            "target": str(v),
            "label": str(rel)[:120],
            "weight": weight,
        }})
        edge_id += 1

    return {
        "nodes":           nodes,
        "edges":           edges,
        "unavailable":     False,
        "n_nodes_total":   n_nodes_total,
        "n_edges_total":   n_edges_total,
        "n_nodes_emitted": len(nodes),
        "n_edges_emitted": len(edges),
        "trimmed":         trimmed,
    }


__all__ = ["graphml_to_cytoscape", "GRAPHML_FILENAME", "MAX_NODES", "MAX_EDGES"]
