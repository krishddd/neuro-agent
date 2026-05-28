"""Phase 5.8 / Extra C — SMART-on-FHIR launch endpoints.

Four endpoints implementing the SMART App Launcher OAuth2 PKCE flow:

    GET  /smart/launch
        EHR-initiated launch entrypoint. The EHR redirects the user here
        with ``iss`` (FHIR base URL) and ``launch`` (opaque launch ID)
        query params. We discover OAuth endpoints from the EHR's
        well-known config, generate a PKCE pair + state, stash them
        in an in-memory session map, and 302 to the EHR's authorize URL.

    GET  /smart/authorize
        Optional standalone-launch entrypoint when the user starts from
        our UI rather than inside the EHR. Same flow without ``launch``.

    GET  /smart/callback
        OAuth2 redirect target. Validates ``state``, exchanges ``code``
        for tokens (using the stashed PKCE verifier), persists the token
        per-patient, and redirects to the patient detail page or a
        success JSON.

    POST /smart/ingest/{fhir_patient_id}
        Pulls Patient + Condition + Observation + MedicationStatement +
        DiagnosticReport for ``fhir_patient_id`` and converts them into
        the existing ``Datasets/patients/<pid>/`` layout. After this
        succeeds the standard pipeline endpoints (POST /api/v1/run) work
        unchanged — *no ZIP upload required*.

Security
--------
The session map (``_PENDING_AUTH``) holds short-lived state +
PKCE verifier between ``launch`` and ``callback``. Entries auto-
expire after 10 min. Token persistence is per-patient at
``outputs/<pid>/.smart_token.json`` (mode 0600).

Graceful degrade
----------------
If ``authlib`` isn't installed, every endpoint returns 503 with an
explanatory message. The ZIP-upload pipeline path stays untouched.
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from ...config import (
    OUTPUTS_DIR,
    SMART_CLIENT_ID,
    SMART_DEFAULT_FHIR_BASE,
    SMART_REDIRECT_URI,
    SMART_SCOPES,
)

log = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory pending-auth session map ────────────────────────────────────────
# Maps state → {code_verifier, fhir_base, authorize_endpoint, token_endpoint,
#               local_pid, fhir_patient_id, created_at}
# Entries auto-expire after 10 minutes (the OAuth code itself is shorter-lived,
# typically 60 s, so 10 min is generous).
_PENDING_AUTH: dict[str, dict[str, Any]] = {}
_PENDING_TTL_S = 600


def _gc_pending() -> None:
    now = time.time()
    expired = [k for k, v in _PENDING_AUTH.items() if now - v.get("created_at", 0) > _PENDING_TTL_S]
    for k in expired:
        _PENDING_AUTH.pop(k, None)


def _safe_pid(pid: str) -> str:
    if not pid or "/" in pid or "\\" in pid or ".." in pid or not pid.isascii():
        raise HTTPException(400, "invalid patient_id")
    return pid.strip().upper()


def _require_smart_available():
    """Return import-time availability flag, 503 if not installed."""
    from ...integrations.fhir_client import SMART_AVAILABLE
    if not SMART_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "smart_unavailable",
                "message": (
                    "SMART-on-FHIR support requires authlib. Install with "
                    "`pip install authlib fhir.resources` and restart."
                ),
            },
        )
    if not SMART_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "smart_unconfigured",
                "message": "SMART_CLIENT_ID not set in env — register an app first.",
            },
        )


# ── /smart/launch ─────────────────────────────────────────────────────────────

@router.get("/smart/launch")
def smart_launch(
    iss:    str = Query(..., description="FHIR base URL (sent by EHR)"),
    launch: str = Query(..., description="Opaque launch context (sent by EHR)"),
    pid:    str | None = Query(None, description="Optional local pid hint"),
) -> Any:
    """EHR-initiated launch — kicks off the OAuth2 PKCE flow."""
    _require_smart_available()

    from ...integrations.fhir_client import SmartFHIRClient

    try:
        endpoints = SmartFHIRClient.discover_endpoints(iss)
    except Exception as exc:
        log.warning("smart_launch: discovery failed for %s: %s", iss, exc)
        raise HTTPException(502, {
            "error": "discovery_failed",
            "message": f"could not reach {iss}/.well-known/smart-configuration: {exc}",
        })

    state = secrets.token_urlsafe(24)
    verifier, challenge = SmartFHIRClient.generate_pkce_pair()
    _gc_pending()
    _PENDING_AUTH[state] = {
        "code_verifier":        verifier,
        "fhir_base":            iss,
        "authorize_endpoint":   endpoints.get("authorization_endpoint", ""),
        "token_endpoint":       endpoints.get("token_endpoint", ""),
        "local_pid":            (_safe_pid(pid) if pid else None),
        "created_at":           time.time(),
        "launch":               launch,
    }

    # We need a temporary client just to build the authorize URL (no token yet,
    # so we point it at any patient_out_dir — token persistence happens in callback).
    tmp_dir = OUTPUTS_DIR / "_pending_smart" / state
    client = SmartFHIRClient(
        patient_out_dir=tmp_dir,
        fhir_base=iss,
        authorize_endpoint=endpoints.get("authorization_endpoint"),
        token_endpoint=endpoints.get("token_endpoint"),
    )
    auth_url = client.authorize_url(state=state, code_challenge=challenge,
                                    launch=launch, aud=iss)
    return RedirectResponse(auth_url, status_code=302)


@router.get("/smart/authorize")
def smart_authorize(
    iss: str | None = Query(None, description="FHIR base URL (defaults to sandbox)"),
    pid: str | None = Query(None),
) -> Any:
    """Standalone launch (no EHR-supplied ``launch`` context)."""
    _require_smart_available()
    from ...integrations.fhir_client import SmartFHIRClient

    fhir_base = iss or SMART_DEFAULT_FHIR_BASE
    try:
        endpoints = SmartFHIRClient.discover_endpoints(fhir_base)
    except Exception as exc:
        raise HTTPException(502, f"discovery failed: {exc}")

    state = secrets.token_urlsafe(24)
    verifier, challenge = SmartFHIRClient.generate_pkce_pair()
    _gc_pending()
    _PENDING_AUTH[state] = {
        "code_verifier":        verifier,
        "fhir_base":            fhir_base,
        "authorize_endpoint":   endpoints.get("authorization_endpoint", ""),
        "token_endpoint":       endpoints.get("token_endpoint", ""),
        "local_pid":            (_safe_pid(pid) if pid else None),
        "created_at":           time.time(),
    }

    tmp_dir = OUTPUTS_DIR / "_pending_smart" / state
    client = SmartFHIRClient(
        patient_out_dir=tmp_dir, fhir_base=fhir_base,
        authorize_endpoint=endpoints.get("authorization_endpoint"),
        token_endpoint=endpoints.get("token_endpoint"),
    )
    auth_url = client.authorize_url(state=state, code_challenge=challenge, aud=fhir_base)
    return RedirectResponse(auth_url, status_code=302)


# ── /smart/callback ───────────────────────────────────────────────────────────

@router.get("/smart/callback")
def smart_callback(
    code:   str = Query(...),
    state:  str = Query(...),
) -> dict[str, Any]:
    """Receive the OAuth2 redirect, exchange code for tokens, persist."""
    _require_smart_available()
    _gc_pending()
    pending = _PENDING_AUTH.pop(state, None)
    if not pending:
        raise HTTPException(400, {
            "error": "invalid_state",
            "message": "unknown or expired state — restart the launch flow",
        })

    from ...integrations.fhir_client import SmartFHIRClient

    # Local pid: prefer the launch-time hint; otherwise fall back to a
    # placeholder until /smart/ingest tells us the FHIR patient id.
    local_pid = pending.get("local_pid") or "_PENDING_SMART"
    patient_out = OUTPUTS_DIR / local_pid

    client = SmartFHIRClient(
        patient_out_dir=patient_out,
        fhir_base=pending["fhir_base"],
        authorize_endpoint=pending["authorize_endpoint"],
        token_endpoint=pending["token_endpoint"],
    )
    try:
        token = client.exchange_code(code=code, code_verifier=pending["code_verifier"])
    except Exception as exc:
        log.error("smart_callback: token exchange failed: %s", exc)
        raise HTTPException(401, f"token exchange failed: {exc}")

    return {
        "ok":         True,
        "message":    "SMART authorization successful — token persisted.",
        "patient_id": local_pid,
        "fhir_base":  pending["fhir_base"],
        # Surface the patient context the EHR returned, when present.
        "fhir_patient_context": token.get("patient"),
        "next_step": (
            f"POST /api/v1/smart/ingest/{token.get('patient', '<fhir_patient_id>')}"
            f"?local_pid={local_pid} to pull resources into the pipeline."
        ),
    }


# ── /smart/ingest/{fhir_patient_id} ───────────────────────────────────────────

@router.post("/smart/ingest/{fhir_patient_id}")
def smart_ingest(
    fhir_patient_id: str,
    local_pid: str | None = Query(None,
                                  description="Pipeline patient_id (defaults to FHIR id)"),
) -> dict[str, Any]:
    """Pull FHIR resources and write the pipeline-internal layout."""
    _require_smart_available()

    from ...integrations.fhir_client import SmartFHIRClient, TokenExpiredError
    from ...utils.fhir_to_pipeline import import_patient_from_fhir

    pid = _safe_pid(local_pid or fhir_patient_id)
    patient_out = OUTPUTS_DIR / pid

    if not (patient_out / ".smart_token.json").exists():
        # Fall back to the _PENDING_SMART staging dir (where /callback writes
        # before we know the real pid).
        pending = OUTPUTS_DIR / "_PENDING_SMART" / ".smart_token.json"
        if pending.exists():
            patient_out.mkdir(parents=True, exist_ok=True)
            (patient_out / ".smart_token.json").write_bytes(pending.read_bytes())

    client = SmartFHIRClient(
        patient_out_dir=patient_out,
        fhir_base=SMART_DEFAULT_FHIR_BASE,
    )
    # Pull endpoints from well-known so refresh works.
    try:
        endpoints = SmartFHIRClient.discover_endpoints(client.fhir_base)
        client.token_endpoint     = endpoints.get("token_endpoint")
        client.authorize_endpoint = endpoints.get("authorization_endpoint")
    except Exception as exc:
        log.warning("smart_ingest: endpoint discovery failed: %s", exc)

    try:
        result = import_patient_from_fhir(client, fhir_patient_id, local_pid=pid)
    except TokenExpiredError as exc:
        raise HTTPException(401, {
            "error": "token_expired",
            "message": str(exc),
            "next_step": "GET /smart/launch to re-authorize",
        })
    except Exception as exc:
        log.error("smart_ingest: import failed: %s", exc)
        raise HTTPException(502, f"FHIR import failed: {exc}")

    result["next_step"] = (
        f"POST /api/v1/run with patient_id={pid} (no file required — "
        f"FHIR data already on disk)."
    )
    return result
