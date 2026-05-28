"""Gmail API client — OAuth2, send patient letters and GP handovers.

Token lifecycle
───────────────
• First run: token.json absent → `setup_oauth.py` must be run once interactively.
• Subsequent runs: token.json loaded, auto-refreshed if expired (refresh_token present).
• If token is invalid and can't refresh: `ready=False`, all send methods return False
  and log a warning — the pipeline continues without email delivery.

Scopes used: gmail.send only (minimal permission surface).
"""
from __future__ import annotations

import base64
import logging
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Resolved at import time — avoids circular config imports.
_PKG = Path(__file__).resolve().parent.parent   # neuro_agent/ package root
CREDENTIALS_PATH = _PKG / "credentials" / "credentials.json"
TOKEN_PATH        = _PKG / "credentials" / "token.json"
# All three scopes — token must be generated with all of them via setup_oauth.py
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/calendar",
]
SENDER            = "krishnahutrik.n@gmail.com"
DOCTOR_EMAIL      = "krishnahutrik.n@gmail.com"

# ── SPIKES safety policy ────────────────────────────────────────────────────
# Bad-news / diagnostic / prognostic communications are NEVER sent directly to
# the patient by this pipeline. They are routed to the doctor's inbox marked
# "[FOR DOCTOR REVIEW — DO NOT FORWARD AS-IS]" so a clinician can deliver the
# information in person (SPIKES protocol). The patient may receive a neutral
# Google Chat invite ONLY if PATIENT_DIRECT_COMMS_ENABLED is set to "true".
import os as _os
PATIENT_DIRECT_COMMS_ENABLED = _os.environ.get(
    "PATIENT_DIRECT_COMMS_ENABLED", "false"
).strip().lower() in ("1", "true", "yes")

_SPIKES_BANNER = (
    "⚠️  DRAFT — FOR DOCTOR REVIEW ONLY\n"
    "────────────────────────────────────────────────────────\n"
    "SPIKES protocol requires a clinician to deliver diagnostic and "
    "prognostic information in person (or by telehealth). Do NOT forward "
    "this email to the patient as-is. Use it as a starting point for the "
    "consultation conversation.\n"
    "────────────────────────────────────────────────────────\n"
)

# ── HTML email shell ─────────────────────────────────────────────────────────
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,Helvetica,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0"
       style="background:#f0f2f5;padding:32px 0">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0"
           style="background:#ffffff;border-radius:10px;overflow:hidden;
                  box-shadow:0 4px 16px rgba(0,0,0,.10)">
      <!-- Header -->
      <tr><td style="background:{header_color};padding:28px 36px">
        <h1 style="color:#fff;margin:0;font-size:20px;font-weight:700;
                   letter-spacing:.3px">🏥&nbsp; Neuro-Oncology Clinic</h1>
        <p  style="color:rgba(255,255,255,.85);margin:6px 0 0;font-size:13px">
            {subtitle}</p>
      </td></tr>
      <!-- Body -->
      <tr><td style="padding:32px 36px;color:#2c2c2c;line-height:1.75;font-size:15px">
        <pre style="white-space:pre-wrap;font-family:Arial,Helvetica,sans-serif;
                    margin:0;font-size:15px;color:#2c2c2c">{body}</pre>
      </td></tr>
      <!-- Disclaimer -->
      <tr><td style="background:#f8f9fa;padding:16px 36px;
                     border-top:1px solid #e8e8e8">
        <p style="color:#888;font-size:11px;margin:0;line-height:1.6">
          {disclaimer}</p>
      </td></tr>
    </table>
  </td></tr>
</table>
</body>
</html>"""

_DISCLAIMER = (
    "This communication was generated with AI assistance and has been reviewed "
    "by the clinical team. It does not replace a formal clinical consultation. "
    "For urgent concerns, contact your care team directly or call 999&nbsp;/&nbsp;112."
)


def _html(body: str, subtitle: str, header_color: str = "#1a5276",
          disclaimer: str = _DISCLAIMER) -> str:
    import html as _html_mod
    return _HTML_TEMPLATE.format(
        header_color=header_color,
        subtitle=_html_mod.escape(subtitle),
        body=_html_mod.escape(body),
        disclaimer=disclaimer,
    )


# ── credential helpers ────────────────────────────────────────────────────────

def _load_creds():
    """Load, refresh if expired, and return Credentials — or None."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        log.warning("gmail: google-auth not installed — run: pip install google-auth google-auth-oauthlib google-api-python-client")
        return None

    if not TOKEN_PATH.exists():
        log.warning("gmail: token.json not found — run setup_oauth.py first")
        return None

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            log.warning("gmail: token refresh failed: %s", exc)
            return None

    if not creds.valid:
        log.warning("gmail: token invalid and could not be refreshed — re-run setup_oauth.py")
        return None

    return creds


# ── GmailClient ──────────────────────────────────────────────────────────────

class GmailClient:
    """Thin wrapper around the Gmail API send endpoint."""

    def __init__(self) -> None:
        self._service: Any = None
        self._ready = False
        self._init()

    def _init(self) -> None:
        creds = _load_creds()
        if creds is None:
            return
        try:
            from googleapiclient.discovery import build
            self._service = build("gmail", "v1", credentials=creds)
            self._ready = True
        except Exception as exc:
            log.warning("gmail: could not build service: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    # ── internal send ────────────────────────────────────────────────────────

    def _send(
        self,
        to: str,
        subject: str,
        html_body: str,
        plain_body: str,
        attachments: list[Path] | None = None,
    ) -> bool:
        """Build a MIME message and POST it via the Gmail API."""
        if not self._ready:
            return False

        msg = MIMEMultipart("mixed")
        msg["From"]    = SENDER
        msg["To"]      = to
        msg["Subject"] = subject

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_body, "plain", "utf-8"))
        alt.attach(MIMEText(html_body,  "html",  "utf-8"))
        msg.attach(alt)

        for path in (attachments or []):
            if not path.exists():
                continue
            part = MIMEBase("application", "octet-stream")
            part.set_payload(path.read_bytes())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment",
                            filename=path.name)
            msg.attach(part)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        try:
            self._service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            log.info("gmail: sent '%s' → %s", subject, to)
            return True
        except Exception as exc:
            log.warning("gmail: send failed [%s → %s]: %s", subject, to, exc)
            return False

    # ── public API ───────────────────────────────────────────────────────────

    def send_patient_letter(
        self,
        to: str = "",
        patient_id: str = "",
        letter_text: str = "",
        diagnosis: str = "",
        urgency_score: int = 1,
        patient_name: str = "",
        intended_recipient: str = "",
    ) -> bool:
        """Send the plain-language patient summary letter.

        SPIKES SAFETY: This method ALWAYS routes to DOCTOR_EMAIL, regardless
        of the `to` argument (kept for backward compat — its value is recorded
        as the `intended_recipient` for the doctor's reference). The doctor
        delivers the bad news to the patient in person.
        """
        if not DOCTOR_EMAIL:
            log.error("gmail: DOCTOR_EMAIL is empty — refusing to send patient letter")
            return False

        # The original `to` arg was the patient's email — preserve it for the
        # doctor to see who the letter is intended for, but route to doctor.
        intended = intended_recipient or to or "(no patient email on file)"

        name_tag = f" for {patient_name}" if patient_name and patient_name != patient_id else ""

        if urgency_score >= 5:
            urgency_tag = "🚨 URGENT"
            hdr_color   = "#922b21"
        elif urgency_score >= 4:
            urgency_tag = "⚠️ Priority"
            hdr_color   = "#a04000"
        else:
            urgency_tag = ""
            hdr_color   = "#1a5276"

        subject = (
            f"[FOR DOCTOR REVIEW — DO NOT FORWARD AS-IS] "
            f"{urgency_tag + ' — ' if urgency_tag else ''}"
            f"Patient Letter Draft [{patient_id}]"
        )
        if diagnosis:
            subject += f" — {diagnosis[:50]}"

        subtitle = (
            f"Draft patient letter{name_tag} — clinician review required "
            f"(intended recipient: {intended})"
        )

        # Prepend the SPIKES banner so it cannot be missed even if the email
        # is forwarded by accident.
        body_with_banner = (
            f"{_SPIKES_BANNER}\n"
            f"Intended patient recipient (after clinician review): {intended}\n"
            f"Patient ID: {patient_id}\n"
            f"Urgency score: {urgency_score}/5\n"
            f"{'─' * 56}\n\n"
            f"{letter_text}"
        )

        return self._send(
            to        = DOCTOR_EMAIL,                 # ← always doctor, never patient
            subject   = subject,
            html_body = _html(body_with_banner, subtitle, header_color=hdr_color),
            plain_body= f"{body_with_banner}\n\n---\n{_DISCLAIMER}",
        )

    def send_gp_handover(
        self,
        patient_id: str,
        gp_text: str,
        diagnosis: str = "",
        patient_name: str = "",
        attachments: list[Path] | None = None,
    ) -> bool:
        """Send the GP handover letter to the doctor's inbox with optional attachments."""
        display = patient_name if patient_name and patient_name != patient_id else patient_id
        subject = f"GP Handover — {display} [{patient_id}]"
        if diagnosis:
            subject += f" — {diagnosis[:50]}"

        subtitle = f"Handover summary for patient {display}"

        return self._send(
            to          = DOCTOR_EMAIL,
            subject     = subject,
            html_body   = _html(gp_text, subtitle, header_color="#1e8449"),
            plain_body  = gp_text,
            attachments = attachments or [],
        )

    def send_urgency_alert(
        self,
        patient_id: str,
        urgency_level: str,
        drivers: list[str],
        patient_email: str = "",
    ) -> bool:
        """Fire an immediate alert to the doctor when urgency score is critical (≥5)."""
        driver_list = "\n".join(f"  • {d}" for d in drivers) or "  • (no drivers recorded)"
        body = (
            f"URGENT CLINICAL ALERT\n"
            f"{'─' * 48}\n"
            f"Patient : {patient_id}\n"
            f"Level   : {urgency_level.upper()}\n"
            f"Email   : {patient_email or 'unknown'}\n\n"
            f"Urgency drivers:\n{driver_list}\n\n"
            f"Action required: review patient record and contact immediately.\n"
        )
        return self._send(
            to         = DOCTOR_EMAIL,
            subject    = f"🚨 URGENT ALERT — {patient_id} — {urgency_level.upper()}",
            html_body  = _html(body, f"Critical alert — {patient_id}",
                               header_color="#922b21"),
            plain_body  = body,
        )

    def send_mdt_alert(
        self,
        patient_id: str,
        proposal: Any,
        subject_prefix: str = "",
    ) -> bool:
        """Send an MDT board alert email after Phase 4 treatment optimisation.

        Fires when mdt_discussion_required=True or decision=REJECT.
        """
        from ..utils.schemas import TreatmentProposal
        prop = proposal if isinstance(proposal, TreatmentProposal) \
            else TreatmentProposal.model_validate(proposal)

        decision_icon = {
            "APPROVE": "✅", "MODIFY": "⚠️",
            "REJECT": "❌", "SKIP": "⏭️",
        }.get(prop.decision, "")

        body_lines = [
            f"MDT TREATMENT OPTIMISATION ALERT",
            f"{'─' * 48}",
            f"Patient  : {patient_id}",
            f"Decision : {decision_icon} {prop.decision}",
            f"Reason   : {prop.reason}",
        ]
        if prop.proposed_regimen:
            body_lines.append(f"Proposed regimen : {prop.proposed_regimen}")
        if prop.modifications:
            body_lines.append(f"Modifications    : {'; '.join(prop.modifications)}")
        if prop.guideline_alignment:
            body_lines.append(f"Guideline        : {prop.guideline_alignment}")
        if prop.rag_interaction_flags:
            body_lines.append(f"Interaction flags: {', '.join(prop.rag_interaction_flags)}")
        if prop.clinical_narrative:
            body_lines += ["", f"Clinical narrative:", prop.clinical_narrative[:500]]
        body_lines += [
            "",
            "⚠️ MDT board discussion required — please review and schedule.",
            "",
            "This is an automated alert from the Neuro-Oncology SMBO v3.0 pipeline.",
        ]
        body = "\n".join(body_lines)

        subject = (
            f"{subject_prefix}[{patient_id}] MDT Review Required — "
            f"Treatment Optimisation {prop.decision}"
        )
        return self._send(
            to         = DOCTOR_EMAIL,
            subject    = subject,
            html_body  = _html(body, f"MDT Alert — {patient_id}", header_color="#7d3c98"),
            plain_body  = body,
        )

    def send_chat_welcome_dm(
        self,
        to: str,
        patient_id: str,
        bot_name: str = "Neuro-Oncology Care Bot",
    ) -> bool:
        """Email the patient telling them how to reach the Google Chat bot.

        SPIKES SAFETY: Disabled by default. Even though this email contains
        no diagnostic content, it announces "your records have been processed"
        which can pre-empt a clinician-led conversation. Set the env var
        PATIENT_DIRECT_COMMS_ENABLED=true to allow this email to fire.
        """
        if not PATIENT_DIRECT_COMMS_ENABLED:
            log.info(
                "gmail: chat_welcome_dm to patient suppressed "
                "(PATIENT_DIRECT_COMMS_ENABLED=false) — SPIKES policy"
            )
            return False
        body = (
            f"Hello,\n\n"
            f"Your clinical records have been processed and are now available through "
            f"our secure care assistant.\n\n"
            f"You can chat with your personal care assistant at any time by opening "
            f"Google Chat and searching for:\n\n"
            f"    👉  {bot_name}\n\n"
            f"You can ask the bot questions like:\n"
            f"  • What did my last MRI scan show?\n"
            f"  • What medications am I currently on?\n"
            f"  • When is my next appointment?\n"
            f"  • What does my RECIST result mean?\n\n"
            f"Your records are ready. The bot will only discuss YOUR records and will "
            f"never share information with other patients.\n\n"
            f"⚠️  For urgent symptoms, always call 999 / 112 or go to A&E immediately.\n\n"
            f"Yours sincerely,\n"
            f"The Neuro-Oncology Team"
        )
        return self._send(
            to         = to,
            subject    = f"[{patient_id}] Your care assistant is ready — chat with us on Google Chat",
            html_body  = _html(body, "Your care assistant is ready",
                               header_color="#1a5276"),
            plain_body  = body,
        )


# ── Module-level convenience wrappers (used by treatment_opt_agent.py) ────────

def send_mdt_alert(
    patient_id: str,
    proposal: Any,
    subject_prefix: str = "",
) -> dict:
    """Module-level wrapper: instantiate GmailClient and call send_mdt_alert."""
    client = GmailClient()
    if not client.ready:
        return {"ok": False, "error": "Gmail not configured"}
    ok = client.send_mdt_alert(patient_id, proposal, subject_prefix=subject_prefix)
    return {"ok": ok}
