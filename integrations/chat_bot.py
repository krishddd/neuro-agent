"""Google Chat Bot — patient Q&A backed by ChromaDB RAG + qwen3:14b.

Message flow
────────────
Patient sends message in Google Chat
    → Google POSTs JSON event to /api/v1/google-chat (our FastAPI endpoint)
    → handle_message(event)
        1. Identify sender email → patient ID via patient_roster
        2. Urgency keyword check — route 999 alert immediately if critical
        3. search_chunks(patient_id, question) → top-6 ChromaDB hits
        4. Build prompt with retrieved context
        5. llm.chat(messages, model=MODEL_PRIMARY)  — qwen3:14b for text reasoning
        6. Return {"text": reply} → Google Chat displays it to the patient

Multi-turn support
──────────────────
Conversation history is kept per Google Chat thread/space in an in-process
dict (_SESSION_HISTORY).  Threads are identified by their Google Chat thread
name (e.g. "spaces/XXXX/threads/YYYY").  The last MAX_HISTORY turns are kept
so the model has context without blowing the context window.

Urgency guard
─────────────
If the message matches any URGENCY_KEYWORDS the bot:
    • Returns an immediate 999/emergency message to the patient
    • Fires a Gmail urgency alert to the doctor
    • Does NOT call the LLM (speed matters in emergencies)
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_HISTORY = 10   # turns kept per session (1 turn = user + assistant)

URGENCY_KEYWORDS: list[str] = [
    "severe headache", "sudden headache", "worst headache of my life",
    "can't see", "cannot see", "vision loss", "blurred vision", "double vision",
    "seizure", "convulsion", "fitting", "fit",
    "unconscious", "collapsed", "unresponsive", "not breathing",
    "emergency", "ambulance", "999", "112", "911",
    "can't breathe", "chest pain",
    "weakness in arm", "weakness in leg", "arm is weak", "leg is weak",
    "can't speak", "can't talk", "speech problem", "slurred speech",
    "sudden confusion", "very confused", "very dizzy",
]

EMERGENCY_MSG = (
    "⚠️  *Please seek emergency help immediately.*\n\n"
    "If this is a medical emergency, call *999* (UK) or *112* (EU) right now.\n\n"
    "I have automatically notified your care team.\n\n"
    "_Do not wait — go to A&E or call an ambulance._"
)

WELCOME_DM = (
    "👋 Hello! I'm your Neuro-Oncology Care Assistant.\n\n"
    "I can answer questions about your medical records, scans, medications, "
    "and treatment plan — using only the information in your clinical notes.\n\n"
    "Examples you can ask me:\n"
    "  • What did my last MRI show?\n"
    "  • What medications am I currently on?\n"
    "  • What does RECIST mean for my results?\n"
    "  • When is my next scan?\n\n"
    "⚠️  For urgent symptoms, always call 999 / your clinic directly.\n"
    "_Powered by Neuro-Oncology AI — clinical assistant._"
)

WELCOME_SPACE = (
    "👋 Neuro-Oncology Care Assistant joined this space.\n"
    "Patients: send me a direct message with your question about your records."
)

NO_PATIENT_RECORD = (
    "I'm sorry — I couldn't find your patient record linked to this account.\n\n"
    "Please contact the clinic to ensure your email address "
    "is registered with the system."
)

BOT_SYSTEM_PROMPT = """\
You are a compassionate, accurate neuro-oncology clinical assistant chatbot.
You answer patient questions using ONLY the medical context provided below.
Model: Neuro-Oncology AI (text reasoning).

Rules:
1. Use ONLY information from the CONTEXT — never invent facts.
2. Speak in plain, patient-friendly language (avoid jargon).
3. For medication changes: always say "please discuss with your care team first."
4. Never predict prognosis or diagnosis beyond what is in the records.
5. Keep responses under 3 short paragraphs.
6. Cite the source (e.g. "According to your MRI report from visit 2…").
7. If the context doesn't contain the answer, say so and suggest contacting the clinic.
8. Always end with: "If you have any urgent symptoms, please call 999 immediately."
"""

# ── In-process session history ────────────────────────────────────────────────
# Maps (patient_id, session_key) → list of {role, content} message dicts.
# Keyed by patient_id+session so one session can never see another patient's data.
_SESSION_HISTORY: dict[str, list[dict[str, Any]]] = {}

# Urgency alert throttle: patient_id → last alert timestamp (monotonic).
_LAST_ALERT: dict[str, float] = {}
_ALERT_THROTTLE_S = 3600   # max 1 urgency alert per patient per hour

# Max message length accepted from patients.
_MAX_MSG = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_urgent(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in URGENCY_KEYWORDS)


def _alert_doctor(patient_id: str, sender_email: str, message: str) -> None:
    """Fire Gmail urgency alert — throttled to max 1 per patient per hour."""
    import time
    now  = time.monotonic()
    last = _LAST_ALERT.get(patient_id, 0.0)
    if now - last < _ALERT_THROTTLE_S:
        log.info("chat_bot: urgency alert throttled for %s (last sent %.0fs ago)",
                 patient_id, now - last)
        return
    _LAST_ALERT[patient_id] = now
    try:
        from .gmail_client import GmailClient
        client = GmailClient()
        if client.ready:
            client.send_urgency_alert(
                patient_id    = patient_id,
                urgency_level = "critical",
                drivers       = [f"Patient message: {message[:300]}"],
                patient_email = sender_email,
            )
            log.warning("chat_bot: urgency alert fired for %s → doctor", patient_id)
    except Exception as exc:
        log.warning("chat_bot: urgency alert failed: %s", exc)


def _answer_question(
    patient_id: str,
    question: str,
    history: list[dict[str, Any]],
) -> str:
    """RAG retrieval from ChromaDB + Gemma4 answer — no WorkingMemory dependency."""
    from ..config import DISCLAIMER, MODEL_PRIMARY
    from ..llm import chat as llm_chat
    from ..tools.chat_agent import search_chunks

    # 1. Retrieve top-6 relevant chunks for this patient.
    chunks = search_chunks(patient_id, question, top_k=6)

    if not chunks:
        return (
            "I couldn't find relevant information in your records to answer that question.\n\n"
            "Please contact the clinic or your neuro-oncology nurse specialist directly.\n\n"
            f"_{DISCLAIMER}_"
        )

    # 2. Build context block from retrieved chunks.
    context_lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        file_ref = c.get("file", "unknown")
        visit    = c.get("visit") or "N/A"
        snippet  = (c.get("text") or "")[:600]
        context_lines.append(f"[{i}] Source: {file_ref} | Visit: {visit}\n{snippet}")
    context = "\n\n---\n\n".join(context_lines)

    # 3. Build message list: system + history + current question.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BOT_SYSTEM_PROMPT},
        *history[-(MAX_HISTORY * 2):],   # last N turns (user+assistant pairs)
        {
            "role": "user",
            "content": (
                f"PATIENT_ID: {patient_id}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"PATIENT QUESTION: {question}"
            ),
        },
    ]

    # 4. Call Gemma 4 exclusively.
    try:
        reply = llm_chat(messages, model=MODEL_PRIMARY)
    except Exception as exc:
        log.warning("chat_bot: llm failed [%s]: %s", patient_id, exc)
        return (
            "I'm having trouble processing your question right now.\n"
            "Please try again in a moment or contact the clinic directly.\n\n"
            f"_{DISCLAIMER}_"
        )

    # 5. Ensure disclaimer is appended.
    if DISCLAIMER not in reply:
        reply = reply.rstrip() + f"\n\n_{DISCLAIMER}_"

    return reply


# ── Main entry point ──────────────────────────────────────────────────────────

def handle_message(event: dict[str, Any]) -> dict[str, Any]:
    """Process one Google Chat event and return the reply dict.

    Called by the FastAPI webhook at POST /api/v1/google-chat.
    Returns {"text": "..."} for simple messages.
    Returns {} to send no reply (e.g. for removal events).
    """
    from .patient_roster import get_patient_id

    event_type = event.get("type", "")

    # ── Bot added to a space / DM ────────────────────────────────────────
    if event_type == "ADDED_TO_SPACE":
        space_type = event.get("space", {}).get("type", "")
        msg = WELCOME_DM if space_type in ("DM", "DIRECT_MESSAGE") else WELCOME_SPACE
        log.info("chat_bot: added to space [%s]", space_type)
        return {"text": msg}

    # ── Bot removed — nothing to reply ──────────────────────────────────
    if event_type == "REMOVED_FROM_SPACE":
        return {}

    # ── Only handle MESSAGE events from here on ─────────────────────────
    if event_type != "MESSAGE":
        return {}

    message     = event.get("message", {})
    text        = (message.get("text") or message.get("argumentText") or "").strip()
    # Strip @mention prefix if present (e.g. "@NeuroCareBot what is my diagnosis?")
    if text.startswith("@"):
        text = text.split(" ", 1)[-1].strip() if " " in text else text

    sender      = event.get("user") or message.get("sender") or {}
    email       = (sender.get("email") or "").lower()
    thread_name = (message.get("thread") or {}).get("name", "")
    space_name  = (event.get("space") or {}).get("name", "")
    # Use thread name as session key; fall back to space name for DMs.
    session_key = thread_name or space_name or email

    log.info("chat_bot: message from %s [session=%s]: %s", email, session_key, text[:80])

    if not text:
        return {"text": "Please type a question and I'll do my best to help."}

    # ── Map email → patient ID ───────────────────────────────────────────
    patient_id = get_patient_id(email)
    if not patient_id:
        log.warning("chat_bot: unknown sender email: %s", email)
        return {"text": NO_PATIENT_RECORD}

    # ── Message size limit ───────────────────────────────────────────────
    if len(text) > _MAX_MSG:
        text = text[:_MAX_MSG]
        log.info("chat_bot: message truncated to %d chars for %s", _MAX_MSG, patient_id)

    # ── Urgency guard ────────────────────────────────────────────────────
    if _is_urgent(text):
        log.warning("chat_bot: URGENT message from %s (%s)", patient_id, email)
        _alert_doctor(patient_id, email, text)
        return {"text": EMERGENCY_MSG}

    # ── Audit: log that this patient accessed their own records ────────
    try:
        from ..utils.audit import log_access
        log_access(
            pid=patient_id, role="patient", caller_name=email,
            action="google_chat_query", endpoint="google_chat",
            meta={"channel": "google_chat"},
        )
    except Exception:
        pass

    # ── Retrieve history scoped to this patient (patient_id + session) ───
    # Key includes patient_id to ensure session data is never shared across patients.
    scoped_key = f"{patient_id}:{session_key}"
    history    = _SESSION_HISTORY.get(scoped_key, [])
    reply      = _answer_question(patient_id, text, history)

    # ── Update session history ───────────────────────────────────────────
    updated = history + [
        {"role": "user",      "content": text},
        {"role": "assistant", "content": reply},
    ]
    # Cap at MAX_HISTORY turns, always trim on a pair boundary.
    max_entries = MAX_HISTORY * 2
    if len(updated) > max_entries:
        excess = len(updated) - max_entries
        excess += excess % 2
        updated = updated[excess:]
    _SESSION_HISTORY[scoped_key] = updated

    return {"text": reply}


def clear_session(patient_id: str, session_key: str) -> None:
    """Clear conversation history for a patient's session (e.g. on space removal)."""
    _SESSION_HISTORY.pop(f"{patient_id}:{session_key}", None)
