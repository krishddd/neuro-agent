"""FastAPI router — Google Chat Bot webhook endpoint.

Google Chat sends a POST to this endpoint whenever a patient messages the bot.
The endpoint verifies the Google JWT, routes to chat_bot.handle_message(),
and returns the reply. Patient can ONLY access their own data (enforced by
email → patient_id mapping in patient_roster).

Endpoint: POST /api/v1/google-chat
Dev:      set GOOGLE_CHAT_SKIP_AUTH=1 to bypass JWT verification locally
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()

_GOOGLE_CHAT_ISSUER = "chat@system.gserviceaccount.com"
_SKIP_AUTH = os.environ.get("GOOGLE_CHAT_SKIP_AUTH", "").strip().lower() in (
    "1", "true", "yes", "on"
)


def _verify_google_jwt(auth_header: str) -> bool:
    """Verify the Bearer JWT from Google Chat."""
    if _SKIP_AUTH:
        return True
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token
        info = id_token.verify_oauth2_token(
            token, g_requests.Request(), audience=None,
        )
        issuer = info.get("iss", "")
        if issuer != _GOOGLE_CHAT_ISSUER:
            log.warning("google_chat: unexpected JWT issuer: %s", issuer)
            return False
        return True
    except ImportError:
        log.error(
            "google_chat: google-auth not installed — JWT cannot be verified. "
            "Install with: pip install google-auth  "
            "Set GOOGLE_CHAT_SKIP_AUTH=1 only for local dev."
        )
        return False
    except Exception as exc:
        log.warning("google_chat: JWT verification failed: %s", exc)
        return False


@router.post("/google-chat")
async def google_chat_webhook(request: Request) -> JSONResponse:
    """Receive Google Chat events and return bot replies.

    Patient isolation: sender email → patient_id via roster.
    Patient can ONLY see their own data.
    """
    auth = request.headers.get("Authorization", "")
    if not _verify_google_jwt(auth):
        raise HTTPException(status_code=401, detail="Unauthorized — invalid Google Chat token")

    try:
        event: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    sender_email = (event.get("user") or {}).get("email", "unknown")
    log.info("google_chat: event type=%s from=%s", event.get("type", "?"), sender_email)

    # Audit log the access.
    try:
        from ...integrations.patient_roster import get_patient_id
        from ...utils.audit import log_access
        pid = get_patient_id(sender_email)
        if pid:
            log_access(
                pid=pid, role="patient", caller_name=sender_email,
                action="google_chat_message", endpoint="/api/v1/google-chat",
                meta={"channel": "google_chat"},
            )
    except Exception:
        pass

    try:
        from ...integrations.chat_bot import handle_message
        reply = handle_message(event)
    except Exception as exc:
        log.error("google_chat: handle_message raised: %s", exc, exc_info=True)
        reply = {"text": "Something went wrong. Please try again or contact the clinic."}

    return JSONResponse(content=reply or {})


@router.get("/google-chat/test/{patient_id}")
async def test_bot_locally(patient_id: str, q: str = "What is my diagnosis?") -> JSONResponse:
    """Dev-only: simulate a Chat message. Requires GOOGLE_CHAT_SKIP_AUTH=1.

    NOTE: Does NOT return patient email in response (security fix).
    """
    if not _SKIP_AUTH:
        raise HTTPException(
            status_code=403,
            detail="Test endpoint only available when GOOGLE_CHAT_SKIP_AUTH=1",
        )

    from ...integrations.patient_roster import PATIENT_EMAILS
    email = PATIENT_EMAILS.get(patient_id.upper())
    if not email:
        raise HTTPException(404, f"Patient {patient_id} not in roster.")

    fake_event: dict[str, Any] = {
        "type": "MESSAGE",
        "user": {"email": email, "displayName": f"Patient {patient_id}"},
        "message": {
            "text": q,
            "thread": {"name": f"spaces/DEV/threads/{patient_id}"},
        },
        "space": {"name": "spaces/DEV", "type": "DM"},
    }

    from ...integrations.chat_bot import handle_message
    reply = handle_message(fake_event)

    # Return reply WITHOUT email (was leaking PII before).
    return JSONResponse(content={
        "patient_id": patient_id,
        "question":   q,
        "bot_reply":  reply.get("text", "(no reply)"),
    })
