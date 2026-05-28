"""Role-based authentication & authorization.

Two roles:
    doctor  — can access ALL patients, run pipelines, chat/qa with any patient_id
    patient — can ONLY access their own data (patient_id bound to their token)

How to use in Swagger UI (/docs):
    1. Open http://localhost:8000/docs
    2. Click the green "Authorize" 🔒 button at the top-right
    3. Paste your API key (from credentials/auth_keys.json)
    4. Click "Authorize" → done — all endpoints work now

Keys are auto-generated on first server start into credentials/auth_keys.json.
The doctor key is also printed in the server startup log.

Google Chat webhook uses its own Google JWT verification (unchanged).
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from .config import PKG

log = logging.getLogger(__name__)

AUTH_KEYS_PATH = PKG / "credentials" / "auth_keys.json"

# ── FastAPI security scheme — shows "Authorize" button in Swagger UI ────────
# auto_error=False so we can give a friendly message instead of raw 403.
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="API Key",
    description=(
        "Paste your API key here. "
        "Find it in neuro_agent/credentials/auth_keys.json "
        "(auto-generated on first server start). "
        "Doctor key starts with doc_  ·  Patient key starts with pat_"
    ),
    auto_error=False,
)

# In-memory key store: api_key → {role, patient_id?, name}
_KEY_STORE: dict[str, dict[str, Any]] = {}
_loaded = False


def _ensure_loaded() -> None:
    """Load keys from disk on first use.  Create default keys if file missing."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    if AUTH_KEYS_PATH.exists():
        try:
            data = json.loads(AUTH_KEYS_PATH.read_text(encoding="utf-8"))
            for entry in data.get("keys", []):
                key = entry.get("api_key", "")
                if key:
                    _KEY_STORE[key] = {
                        "role":       entry.get("role", "patient"),
                        "patient_id": entry.get("patient_id"),
                        "name":       entry.get("name", ""),
                    }
            log.info("auth: loaded %d API keys from %s", len(_KEY_STORE), AUTH_KEYS_PATH)
            return
        except Exception as exc:
            log.warning("auth: failed to load %s: %s", AUTH_KEYS_PATH, exc)

    # First run — generate default keys and save them.
    _generate_defaults()


def _generate_defaults() -> None:
    """Create default auth_keys.json with one doctor key + per-patient keys."""
    from .integrations.patient_roster import PATIENT_EMAILS

    keys_list: list[dict[str, Any]] = []

    # Doctor master key
    doctor_key = f"doc_{secrets.token_hex(16)}"
    keys_list.append({
        "api_key":    doctor_key,
        "role":       "doctor",
        "name":       "Doctor (auto-generated)",
    })
    _KEY_STORE[doctor_key] = {"role": "doctor", "patient_id": None, "name": "Doctor (auto-generated)"}

    # Per-patient keys
    for pid in sorted(PATIENT_EMAILS.keys()):
        pkey = f"pat_{pid.lower()}_{secrets.token_hex(12)}"
        keys_list.append({
            "api_key":    pkey,
            "role":       "patient",
            "patient_id": pid,
            "name":       f"Patient {pid}",
        })
        _KEY_STORE[pkey] = {"role": "patient", "patient_id": pid, "name": f"Patient {pid}"}

    # Save to disk
    AUTH_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_KEYS_PATH.write_text(
        json.dumps({"keys": keys_list}, indent=2),
        encoding="utf-8",
    )
    log.info(
        "auth: generated default keys → %s\n"
        "  ╔══════════════════════════════════════════════════════════╗\n"
        "  ║  DOCTOR API KEY: %-38s ║\n"
        "  ║  Paste this in Swagger UI → Authorize 🔒                ║\n"
        "  ╚══════════════════════════════════════════════════════════╝",
        AUTH_KEYS_PATH, doctor_key,
    )


# ── Core lookup ─────────────────────────────────────────────────────────────


def _lookup(api_key: str | None) -> dict[str, Any] | None:
    """Return identity dict for a valid key, or None."""
    _ensure_loaded()
    if not api_key:
        return None
    return _KEY_STORE.get(api_key)


# ── Dependency functions (use with FastAPI Depends) ─────────────────────────
#
# These are injected via Depends() in the router functions.
# FastAPI reads the _api_key_header dependency and shows the Authorize
# button in Swagger UI automatically.


def authenticate(api_key: str | None = Depends(_api_key_header)) -> dict[str, Any]:
    """Validate API key and return identity dict.

    Returns: {"role": "doctor"|"patient", "patient_id": str|None, "name": str}
    Raises 401 if missing/invalid.
    """
    identity = _lookup(api_key)
    if identity is None:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing or invalid API key.\n"
                "→ In Swagger UI: click the 🔒 Authorize button and paste your key.\n"
                "→ In curl/Postman: add header  X-API-Key: <your_key>\n"
                "→ Keys are in: neuro_agent/credentials/auth_keys.json"
            ),
        )
    return dict(identity)


def require_doctor(api_key: str | None = Depends(_api_key_header)) -> dict[str, Any]:
    """Authenticate and require doctor role."""
    identity = authenticate(api_key)
    if identity["role"] != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Doctor access required. Patient keys cannot access this endpoint.",
        )
    return identity


def require_patient_access_factory(patient_id_param: str = "patient_id"):
    """Create a dependency that checks patient access dynamically.

    Can't use Depends() for patient_id because it comes from the request body,
    so routers call require_patient_access() directly instead.
    """
    pass   # see require_patient_access() below


def require_patient_access(identity: dict[str, Any], patient_id: str) -> dict[str, Any]:
    """Verify the caller can access this patient's data.

    Doctors can access any patient.
    Patients can only access their own data.
    Call this AFTER authenticate().
    """
    if identity["role"] == "doctor":
        return identity
    if (identity.get("patient_id") or "").upper() != patient_id.upper():
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Your key is bound to patient "
                   f"{identity.get('patient_id')}, not {patient_id}.",
        )
    return identity
