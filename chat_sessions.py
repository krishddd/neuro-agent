"""In-memory chat session store.

Thread-safe, LRU-evicting session store.  Each session pins one patient_id
and stores a rolling history of {role, content} turns.

Sessions idle for more than IDLE_TIMEOUT_H hours are automatically reaped.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import CHAT_HISTORY_TURNS

IDLE_TIMEOUT_H    = 24    # sessions unused for 24 h are dropped automatically
_MAX_SESSIONS     = 256
_MAX_EVICTION_LOG = 512

_lock    = threading.Lock()
_SESSIONS: dict[str, "ChatSession"] = {}
_EVICTED: set[str] = set()


@dataclass
class ChatSession:
    session_id: str
    patient_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_used:  float = field(default_factory=time.monotonic)
    history:    list[dict[str, Any]] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.last_used = time.monotonic()
        # Keep last N turns (2*N entries — user + assistant per turn).
        # Always trim on a pair boundary so we never split a turn.
        max_entries = CHAT_HISTORY_TURNS * 2
        if len(self.history) > max_entries:
            excess = len(self.history) - max_entries
            excess += excess % 2          # round up to even
            self.history = self.history[excess:]

    def messages(self) -> list[dict[str, Any]]:
        return list(self.history)


# ── internal helpers (caller must hold _lock) ─────────────────────────────────

def _evict_idle_unsafe() -> None:
    cutoff = time.monotonic() - IDLE_TIMEOUT_H * 3600
    stale  = [sid for sid, s in _SESSIONS.items() if s.last_used < cutoff]
    for sid in stale:
        _EVICTED.add(sid)
        del _SESSIONS[sid]


def _evict_if_full_unsafe() -> None:
    _evict_idle_unsafe()
    if len(_SESSIONS) <= _MAX_SESSIONS:
        return
    oldest = min(_SESSIONS.values(), key=lambda s: s.last_used)
    _EVICTED.add(oldest.session_id)
    del _SESSIONS[oldest.session_id]
    if len(_EVICTED) > _MAX_EVICTION_LOG:
        _EVICTED.clear()


# ── public API ────────────────────────────────────────────────────────────────

def was_evicted(session_id: str) -> bool:
    """Return True if this session existed but was evicted."""
    with _lock:
        return session_id in _EVICTED


def create(patient_id: str) -> ChatSession:
    with _lock:
        _evict_if_full_unsafe()
        sid  = uuid.uuid4().hex
        sess = ChatSession(session_id=sid, patient_id=patient_id)
        _SESSIONS[sid] = sess
        return sess


def get(session_id: str) -> ChatSession | None:
    with _lock:
        sess = _SESSIONS.get(session_id)
        if sess:
            sess.last_used = time.monotonic()
        return sess


def get_or_create(session_id: str | None, patient_id: str) -> ChatSession:
    with _lock:
        if session_id and session_id in _SESSIONS:
            sess = _SESSIONS[session_id]
            if sess.patient_id != patient_id:
                raise ValueError(
                    f"session '{session_id}' belongs to a different patient"
                )
            sess.last_used = time.monotonic()
            return sess
        _evict_if_full_unsafe()
        sid  = uuid.uuid4().hex
        sess = ChatSession(session_id=sid, patient_id=patient_id)
        _SESSIONS[sid] = sess
        return sess


def drop(session_id: str) -> None:
    with _lock:
        _SESSIONS.pop(session_id, None)
        _EVICTED.discard(session_id)


def active_count() -> int:
    with _lock:
        return len(_SESSIONS)
