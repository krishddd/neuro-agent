"""Multi-turn chat endpoint + WebSocket streaming — doctor Q&A."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, field_validator

from ... import chat_sessions
from ...config import OUTPUTS_DIR
from ...memory import WorkingMemory
from ...tools.chat_agent import answer, stream_answer
from ...utils.audit import log_access

log = logging.getLogger(__name__)
router = APIRouter()

_MAX_MSG_LEN = 2000
_MAX_PID_LEN = 20


class ChatRequest(BaseModel):
    patient_id: str
    message:    str
    session_id: Optional[str] = None

    @field_validator("patient_id")
    @classmethod
    def _clean_pid(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or len(v) > _MAX_PID_LEN:
            raise ValueError("patient_id invalid")
        return v

    @field_validator("message")
    @classmethod
    def _clean_msg(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty")
        if len(v) > _MAX_MSG_LEN:
            raise ValueError(f"message too long (max {_MAX_MSG_LEN} chars)")
        return v


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    patient_dir = OUTPUTS_DIR / req.patient_id
    if not patient_dir.exists():
        raise HTTPException(
            404,
            f"Patient '{req.patient_id}' not found. "
            "Run the pipeline first: POST /api/v1/run",
        )

    memory = WorkingMemory.load(req.patient_id)
    if not memory.has(WorkingMemory.INGESTION):
        raise HTTPException(
            409,
            f"Patient '{req.patient_id}' pipeline output is incomplete.",
        )

    # Check for evicted session.
    if req.session_id and not chat_sessions.get(req.session_id):
        if chat_sessions.was_evicted(req.session_id):
            raise HTTPException(
                410,
                detail={
                    "error": "session_expired",
                    "message": "Session expired. Please start a new conversation.",
                },
            )

    try:
        sess = chat_sessions.get_or_create(req.session_id, req.patient_id)
    except ValueError as e:
        raise HTTPException(409, str(e))

    log.info("chat: patient=%s  session=%s  msg=%s…",
             req.patient_id, sess.session_id, req.message[:60])

    log_access(
        pid=req.patient_id, role="doctor", caller_name="swagger_ui",
        action="chat_message", endpoint="/api/v1/chat",
        meta={"session_id": sess.session_id},
    )

    history   = sess.messages()
    qa_answer = answer(memory, req.message, history=history)
    sess.append("user", req.message)
    sess.append("assistant", qa_answer.answer)

    out = qa_answer.model_dump(mode="json")
    out["session_id"] = sess.session_id
    return out


@router.websocket("/chat/stream/{session_id}")
async def chat_stream(ws: WebSocket, session_id: str) -> None:
    await ws.accept()

    sess = chat_sessions.get(session_id)
    if not sess:
        msg = ("session_expired" if chat_sessions.was_evicted(session_id)
               else "unknown_session")
        await ws.send_json({"error": msg,
                            "message": "Start a new session via POST /api/v1/chat"})
        await ws.close()
        return

    patient_dir = OUTPUTS_DIR / sess.patient_id
    if not patient_dir.exists():
        await ws.send_json({"error": "patient_not_found"})
        await ws.close()
        return

    memory = WorkingMemory.load(sess.patient_id)
    if not memory.has(WorkingMemory.INGESTION):
        await ws.send_json({"error": "patient_not_processed"})
        await ws.close()
        return

    try:
        while True:
            data     = await ws.receive_json()
            question = (data or {}).get("message", "").strip()
            if not question:
                await ws.send_json({"error": "empty_message"})
                continue
            if len(question) > _MAX_MSG_LEN:
                await ws.send_json({"error": "message_too_long",
                                    "max": _MAX_MSG_LEN})
                continue

            await ws.send_json({"event": "start", "session_id": session_id})
            buf: list[str] = []
            try:
                for chunk in stream_answer(memory, question,
                                           history=sess.messages()):
                    buf.append(chunk)
                    await ws.send_json({"event": "token", "delta": chunk})
            except Exception as e:
                await ws.send_json({"event": "error", "error": str(e)[:200]})
                continue

            full = "".join(buf)
            sess.append("user", question)
            sess.append("assistant", full)
            await ws.send_json({"event": "end", "session_id": session_id})

    except WebSocketDisconnect:
        return
