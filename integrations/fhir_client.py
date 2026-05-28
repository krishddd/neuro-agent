"""Phase 5.8 / Extra C — SMART-on-FHIR client + TokenManager.

Authenticates the agent against an EHR's SMART App Launcher (Epic /
Cerner / public sandbox) and issues authenticated FHIR R4 requests.

Why TokenManager exists
-----------------------
Standard EHR OAuth2 access tokens expire in 15-30 min. A cold pipeline
run takes ~36 min. A FHIR pull at minute 0 + a CarePlan write-back
during synthesis at minute 35 will hit a dead token and 401. The
``offline_access`` scope (in ``SMART_SCOPES``) grants a refresh token;
``_ensure_fresh_token()`` proactively refreshes if <120 s remain.

Token persistence
-----------------
Tokens are stored per-patient at
``outputs/<pid>/.smart_token.json`` with mode 0600 (owner read/write
only). The file is excluded from git via ``.gitignore`` and from
finalize() legacy cleanup. On refresh-token expiry (~90 days for most
vendors) ``_ensure_fresh_token()`` raises ``TokenExpiredError`` so the
caller can surface a pending-reauth state to the UI.

Graceful degrade
----------------
``authlib`` is an optional dependency — when not installed,
``SMART_AVAILABLE = False`` and the smart_fhir router 503s. The
existing ZIP-upload path stays untouched.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any, Optional

from .. import config

log = logging.getLogger(__name__)

# ── authlib import: fail soft ─────────────────────────────────────────────────
try:
    from authlib.integrations.requests_client import OAuth2Session  # type: ignore
    SMART_AVAILABLE = True
except Exception as _exc:
    OAuth2Session = None  # type: ignore
    SMART_AVAILABLE = False
    log.info("fhir_client: authlib not available (%s) — SMART mode disabled", _exc)


TOKEN_FILENAME = ".smart_token.json"


class TokenExpiredError(RuntimeError):
    """Raised when both access_token and refresh_token can no longer be refreshed."""


# ── token persistence ─────────────────────────────────────────────────────────

def _token_path(patient_out_dir: Path) -> Path:
    return Path(patient_out_dir) / TOKEN_FILENAME


def _save_token(patient_out_dir: Path, token: dict[str, Any]) -> Path:
    """Atomically persist token JSON with mode 0600."""
    path = _token_path(patient_out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(token, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)   # 0600 (POSIX)
    except OSError:
        pass  # Windows ignores; ACLs handled separately.
    tmp.replace(path)
    return path


def _load_token(patient_out_dir: Path) -> dict[str, Any] | None:
    path = _token_path(patient_out_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("fhir_client: token load failed for %s: %s", path, exc)
        return None


# ── SMART client ──────────────────────────────────────────────────────────────

class SmartFHIRClient:
    """Thin SMART-on-FHIR wrapper.

    Lifecycle:
        1. ``authorize_url(state, code_challenge)`` — start OAuth2 PKCE flow.
        2. ``exchange_code(code, code_verifier)`` — get initial token.
        3. Subsequent ``request(method, url)`` calls use ``_ensure_fresh_token()``
           to keep the access token live across long pipeline runs.
    """

    def __init__(
        self,
        patient_out_dir: Path,
        *,
        fhir_base: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scopes: Optional[str] = None,
        authorize_endpoint: Optional[str] = None,
        token_endpoint: Optional[str] = None,
    ):
        if not SMART_AVAILABLE:
            raise RuntimeError("authlib not installed — SmartFHIRClient unavailable")

        self.patient_out_dir = Path(patient_out_dir)
        self.fhir_base       = (fhir_base or config.SMART_DEFAULT_FHIR_BASE).rstrip("/")
        self.client_id       = client_id     or config.SMART_CLIENT_ID
        self.client_secret   = client_secret or config.SMART_CLIENT_SECRET
        self.redirect_uri    = redirect_uri  or config.SMART_REDIRECT_URI
        self.scopes          = scopes        or config.SMART_SCOPES
        # Endpoints come from the SMART well-known config. Callers can
        # override or pre-fetch via ``discover_endpoints``.
        self.authorize_endpoint = authorize_endpoint
        self.token_endpoint     = token_endpoint

        # Hydrate any persisted token for this patient.
        self._token: dict[str, Any] | None = _load_token(self.patient_out_dir)

    # ── discovery ─────────────────────────────────────────────────────
    @staticmethod
    def discover_endpoints(fhir_base: str) -> dict[str, str]:
        """Hit ``.well-known/smart-configuration`` and pluck OAuth endpoints."""
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"requests required for SMART discovery: {exc}")

        url = f"{fhir_base.rstrip('/')}/.well-known/smart-configuration"
        resp = requests.get(url, timeout=10.0)
        resp.raise_for_status()
        cfg = resp.json() or {}
        return {
            "authorization_endpoint": cfg.get("authorization_endpoint", ""),
            "token_endpoint":         cfg.get("token_endpoint", ""),
            "introspection_endpoint": cfg.get("introspection_endpoint", ""),
            "revocation_endpoint":    cfg.get("revocation_endpoint", ""),
        }

    # ── PKCE ──────────────────────────────────────────────────────────
    @staticmethod
    def generate_pkce_pair() -> tuple[str, str]:
        """Return (code_verifier, code_challenge) per RFC 7636 (S256)."""
        import base64
        import hashlib
        import secrets
        verifier = secrets.token_urlsafe(64)[:128]
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return (verifier, challenge)

    # ── authorize / token-exchange ────────────────────────────────────
    def authorize_url(self, state: str, code_challenge: str,
                      *, launch: str | None = None,
                      aud: str | None = None) -> str:
        """Build the URL the EHR redirects the user to."""
        if not self.authorize_endpoint:
            raise RuntimeError("authorize_endpoint not set — call discover_endpoints first")
        params = {
            "response_type":         "code",
            "client_id":             self.client_id,
            "redirect_uri":          self.redirect_uri,
            "scope":                 self.scopes,
            "state":                 state,
            "code_challenge":        code_challenge,
            "code_challenge_method": "S256",
            "aud":                   aud or self.fhir_base,
        }
        if launch:
            params["launch"] = launch
        from urllib.parse import urlencode
        return f"{self.authorize_endpoint}?{urlencode(params)}"

    def exchange_code(self, code: str, code_verifier: str) -> dict[str, Any]:
        """Exchange auth code for tokens; persist + cache the result."""
        if not self.token_endpoint:
            raise RuntimeError("token_endpoint not set — call discover_endpoints first")
        sess = OAuth2Session(  # type: ignore[misc]
            self.client_id, self.client_secret,
            redirect_uri=self.redirect_uri, scope=self.scopes,
            code_challenge_method="S256",
        )
        token = sess.fetch_token(
            self.token_endpoint, code=code,
            code_verifier=code_verifier,
            client_id=self.client_id,
        )
        self._cache_token(token)
        return token

    # ── refresh logic ────────────────────────────────────────────────
    def _cache_token(self, token: dict[str, Any]) -> None:
        # authlib returns ``expires_at`` (epoch); fall back to expires_in.
        if "expires_at" not in token and "expires_in" in token:
            token["expires_at"] = int(time.time()) + int(token.get("expires_in", 0))
        self._token = token
        try:
            _save_token(self.patient_out_dir, token)
        except OSError as exc:
            log.warning("fhir_client: token persist failed: %s", exc)

    def _ensure_fresh_token(self, margin_s: int | None = None) -> str:
        """Return a non-expired access_token; refresh if needed."""
        if not self._token:
            raise TokenExpiredError("no token on file — run authorize flow")
        margin = margin_s if margin_s is not None else config.SMART_TOKEN_REFRESH_MARGIN_S
        expires_at = int(self._token.get("expires_at") or 0)
        if expires_at and time.time() + margin < expires_at:
            return str(self._token.get("access_token") or "")

        # Refresh.
        refresh_token = self._token.get("refresh_token")
        if not refresh_token:
            raise TokenExpiredError(
                "access token expired and no refresh_token (offline_access not granted)"
            )
        if not self.token_endpoint:
            raise RuntimeError("token_endpoint not set — cannot refresh")

        sess = OAuth2Session(  # type: ignore[misc]
            self.client_id, self.client_secret, scope=self.scopes,
        )
        try:
            new_token = sess.refresh_token(self.token_endpoint, refresh_token=refresh_token)
        except Exception as exc:
            raise TokenExpiredError(f"refresh failed: {exc}") from exc
        # Vendors sometimes omit refresh_token in the refresh response —
        # retain the old one when that happens.
        if "refresh_token" not in new_token and refresh_token:
            new_token["refresh_token"] = refresh_token
        self._cache_token(new_token)
        return str(new_token.get("access_token") or "")

    # ── HTTP ──────────────────────────────────────────────────────────
    def request(self, method: str, path_or_url: str, **kwargs) -> Any:
        """Issue a FHIR HTTP request with auto-refreshed bearer auth."""
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"requests required for FHIR calls: {exc}")

        token = self._ensure_fresh_token()
        url = path_or_url
        if not url.startswith("http"):
            url = f"{self.fhir_base}/{url.lstrip('/')}"
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("Authorization", f"Bearer {token}")
        headers.setdefault("Accept",        "application/fhir+json")
        headers.setdefault("User-Agent",    "neuro-agent/0.2.0")
        kwargs.setdefault("timeout", 15.0)
        return requests.request(method, url, headers=headers, **kwargs)

    def get(self, path_or_url: str, **kwargs) -> Any:
        return self.request("GET", path_or_url, **kwargs)

    def post(self, path_or_url: str, **kwargs) -> Any:
        return self.request("POST", path_or_url, **kwargs)


__all__ = [
    "SMART_AVAILABLE",
    "SmartFHIRClient",
    "TokenExpiredError",
    "TOKEN_FILENAME",
]
