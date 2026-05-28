"""FastAPI gateway for the Neuro-Oncology Unified Care Agent.

Run with:
    uvicorn neuro_agent.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import DISCLAIMER
from ..llm import ping
from .routers import approval, chat, google_chat, graph_ui, process, qa, smart_fhir

app = FastAPI(
    title="Neuro-Oncology Unified Care Agent",
    version="0.2.0",
    description=(
        "Guarded multi-agent pipeline for neuro-oncology MRI + RECIST + "
        "drug-interaction analysis. " + DISCLAIMER
    ),
)

# CORS — configurable via env var. Defaults to allow all for dev.
_origins = os.environ.get("NEURO_CORS_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _origins.split(",") if o.strip()] if _origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# Emit the DICOM decompression-capability banner once at server boot so
# the operator knows up-front whether compressed DICOMs (JPEG / JPEG2000
# / JPEG-LS / RLE) will decode. Without it, "Cannot extract pixel data"
# warnings during ingest were the only signal — easy to miss.
try:
    from ..utils.dicom_anon import log_dicom_dependency_status
    log_dicom_dependency_status()
except Exception:
    pass


@app.get("/healthz")
def healthz() -> dict:
    """Liveness + dependency check.

    Includes a ``dicom`` field summarising compressed-DICOM handler
    availability so a quick GET tells the operator whether to expect
    pixel-data failures during ingest.
    """
    try:
        from ..utils.dicom_anon import dicom_handler_summary
        dicom_status = dicom_handler_summary()
    except Exception as exc:
        dicom_status = f"check_failed: {exc}"
    return {"ok": True, "ollama": ping(), "dicom": dicom_status}


@app.get("/")
def root() -> dict:
    return {
        "name": "neuro-oncology-agent",
        "version": "0.2.0",
        "disclaimer": DISCLAIMER,
        "endpoints": {
            "GET  /healthz":
                "Ollama reachability check",
            "POST /api/v1/run":
                "Upload patient ZIP → run full pipeline → returns complete results. Fields: patient_id, file (zip)",
            "GET  /api/v1/patients":
                "List all processed patients with summary info",
            "GET  /api/v1/patients/{patient_id}":
                "Patient detail — outputs, stages completed, notifications sent",
            "POST /api/v1/qa":
                "Doctor single-turn Q&A. Body: {patient_id, question}",
            "POST /api/v1/chat":
                "Doctor multi-turn chat. Body: {patient_id, message, session_id?}. Returns session_id",
            "WS   /api/v1/chat/stream/{session_id}": (
                "WebSocket streaming chat. "
                "Create session via POST /api/v1/chat first, then open WS with returned session_id. "
                "Send: {\"message\": \"...\"}. "
                "Receive: {\"event\":\"start\"} | {\"event\":\"token\",\"delta\":\"...\"} | {\"event\":\"end\"} | {\"event\":\"error\"}"
            ),
            "POST /api/v1/google-chat":
                "Patient Google Chat webhook (called by Google). Body: Google Chat event JSON",
            "GET  /api/v1/google-chat/test/{patient_id}":
                "Dev test — simulates a patient message. Requires GOOGLE_CHAT_SKIP_AUTH=1",
            "POST /api/v1/calendar/clear":
                "Delete ALL agent-created Google Calendar events (dev/cleanup tool)",
            "POST /api/v1/calendar/clear/{patient_id}":
                "Delete calendar events for one patient only",
            "GET  /api/v1/graph/{patient_id}":
                "Cytoscape-compatible JSON for the patient's LightRAG knowledge graph (Phase 5.6)",
            "GET  /api/v1/graph/{patient_id}/status":
                "LightRAG build sentinel: building | ready | failed | absent",
            "GET  /ui/graph_viewer.html?pid={patient_id}":
                "Browser-side knowledge-graph viewer (Cytoscape.js, no build step)",
            "GET  /smart/launch":
                "EHR-initiated SMART-on-FHIR launch (Phase 5.8). Params: iss, launch[, pid]",
            "GET  /smart/authorize":
                "Standalone SMART launch (no EHR context). Params: [iss, pid]",
            "GET  /smart/callback":
                "OAuth2 redirect target — exchanges code for tokens, persists per-patient",
            "POST /api/v1/smart/ingest/{fhir_patient_id}?local_pid=...":
                "Pull FHIR resources and write Datasets/patients/<pid>/ — pipeline runs unchanged after",
        },
        "quick_start": {
            "1_run_pipeline": "POST /api/v1/run  form: patient_id=P001  file=<patient.zip>",
            "2_list_patients": "GET  /api/v1/patients",
            "3_ask_question":  "POST /api/v1/qa   body={\"patient_id\":\"P001\", \"question\":\"What is the diagnosis?\"}",
            "4_chat_session":  "POST /api/v1/chat body={\"patient_id\":\"P001\", \"message\":\"Summarise the treatment plan\"}",
        },
    }


app.include_router(process.router,      prefix="/api/v1", tags=["pipeline"])
app.include_router(approval.router,     prefix="/api/v1", tags=["hitl-approval"])
app.include_router(qa.router,           prefix="/api/v1", tags=["doctor-qa"])
app.include_router(chat.router,         prefix="/api/v1", tags=["doctor-chat"])
app.include_router(google_chat.router,  prefix="/api/v1", tags=["patient-chat"])
app.include_router(graph_ui.router,     prefix="/api/v1", tags=["graph-ui"])
# Phase 5.8 / Extra C — SMART-on-FHIR launch endpoints. Mounted at root
# (/smart/*) and under /api/v1/ for ingest, matching SMART convention.
app.include_router(smart_fhir.router,                      tags=["smart-fhir"])
app.include_router(smart_fhir.router,    prefix="/api/v1", tags=["smart-fhir-ingest"])

# Phase 5.6 / Extra A — Static viewer for the LightRAG knowledge graph.
# Visit http://localhost:8000/ui/graph_viewer.html?pid=P001 after a run.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")
