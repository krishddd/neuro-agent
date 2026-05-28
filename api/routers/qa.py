"""Stateless single-turn Q&A endpoint — doctor queries patient data."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from ...config import OUTPUTS_DIR
from ...memory import WorkingMemory
from ...tools.chat_agent import answer
from ...utils.audit import log_access

log = logging.getLogger(__name__)
router = APIRouter()

_MAX_PID_LEN = 20
_MAX_Q_LEN   = 2000


class QARequest(BaseModel):
    patient_id: str
    question:   str

    @field_validator("patient_id")
    @classmethod
    def _clean_pid(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("patient_id cannot be empty")
        if len(v) > _MAX_PID_LEN:
            raise ValueError(f"patient_id too long (max {_MAX_PID_LEN} chars)")
        return v

    @field_validator("question")
    @classmethod
    def _clean_q(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty")
        if len(v) > _MAX_Q_LEN:
            raise ValueError(f"question too long (max {_MAX_Q_LEN} chars)")
        return v


@router.post("/qa")
def qa(req: QARequest) -> dict:
    # Verify the patient has been processed (outputs folder must exist).
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
            f"Patient '{req.patient_id}' pipeline output is incomplete. "
            "Re-run the pipeline: POST /api/v1/run",
        )

    log.info("qa: patient=%s  q=%s…", req.patient_id, req.question[:60])

    log_access(
        pid=req.patient_id, role="doctor", caller_name="swagger_ui",
        action="qa_query", endpoint="/api/v1/qa",
    )

    qa_answer = answer(memory, req.question)
    return qa_answer.model_dump(mode="json")
