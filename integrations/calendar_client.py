"""Google Calendar client — schedule follow-up appointments, medication
reminders, and therapy sessions derived from RECIST + pharmacy pipeline outputs.

Event types created
───────────────────
1. Follow-up appointment  — interval driven by urgency score + RECIST response
      urgency 5   →  1 week   (🚨 critical)
      PD          →  2 weeks  (MDT review)
      PR / CR     →  6 weeks  (confirmation scan required per RECIST 1.1)
      SD / NE     →  8 weeks  (routine surveillance)

2. Medication reminders   — one recurring event per current drug
      frequency string parsed into RRULE: daily / twice-daily / weekly / monthly
      Temozolomide 5-day-on / 23-day-off pattern handled explicitly.

3. Therapy sessions        — extracted from discharge/correlation free text
      RT fractions, infusion days, steroid tapers.

All events are created on the doctor's primary calendar.
Patient is added as attendee (optional — only if patient_email is known).
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PKG       = Path(__file__).resolve().parent.parent   # neuro_agent/ package root
TOKEN_PATH = _PKG / "credentials" / "token.json"
# All three scopes — token must be generated with all of them via setup_oauth.py
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/calendar",
]
DOCTOR_EMAIL = "krishnahutrik.n@gmail.com"

# ── Patient-safety policy ───────────────────────────────────────────────────
# A proposed treatment from Phase 4 SMBO is just that — a proposal. It MUST
# NOT be auto-scheduled on the doctor's or patient's calendar before the
# Multi-Disciplinary Team has met AND the patient has given informed consent.
# Synthesis must NOT call create_proposed_regimen_events. The method is left
# defined so a future approved-execution path (see HITL gate, Task 9) may
# call it after a clinician explicitly approves the regimen.
AUTO_SCHEDULE_PROPOSED_TREATMENT = False

# All agent-created events are capped to this many days from the start date.
# Keeps the calendar clean during development and avoids flooding the doctor's
# calendar with multi-year projections from a single pipeline run.
_CALENDAR_WINDOW_DAYS = 30

# Calendar color IDs (Google's palette).
COLOR_RED      = "11"   # Tomato
COLOR_ORANGE   = "6"    # Tangerine
COLOR_GREEN    = "2"    # Sage
COLOR_BLUE     = "1"    # Lavender
COLOR_TEAL     = "7"    # Peacock


def _load_creds():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        log.warning("calendar: google-auth not installed")
        return None

    if not TOKEN_PATH.exists():
        log.warning("calendar: token.json not found — run setup_oauth.py first")
        return None

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            log.warning("calendar: token refresh failed: %s", exc)
            return None
    return creds if creds.valid else None


# ── frequency parsing ─────────────────────────────────────────────────────────

def _rrule_from_frequency(freq_str: str) -> tuple[str, int]:
    """Return (RRULE_string, events_per_day).

    events_per_day >1 means create that many events on day 0 (e.g. BD).
    """
    f = (freq_str or "").lower()
    if re.search(r"twice|bd|b\.d\.|bid|twice.daily", f):
        return "RRULE:FREQ=DAILY", 2
    if re.search(r"three.times|tds|t\.d\.s\.|tid|three.times.daily", f):
        return "RRULE:FREQ=DAILY", 3
    if re.search(r"weekly|once.week|qw|q\.w\.", f):
        return "RRULE:FREQ=WEEKLY", 1
    if re.search(r"monthly|once.month|qm|q\.m\.", f):
        return "RRULE:FREQ=MONTHLY", 1
    if re.search(r"5.day|5/23|tmz.5|day.1.to.5", f):
        return "TMZ_523", 1   # Special — handled below.
    # Default: once daily.
    return "RRULE:FREQ=DAILY", 1


def _tmz_523_events(
    drug_name: str,
    dose: str,
    start_date: date,
    patient_id: str,
    cycles: int = 1,
) -> list[dict[str, Any]]:
    """Generate individual events for Temozolomide 5-days-on / 23-days-off schedule.

    Defaults to 1 cycle (5 treatment days + 23-day rest = 28 days) so the
    events stay within the 1-month calendar window.  Pass a higher ``cycles``
    value only when explicitly requested.
    """
    cutoff = start_date + timedelta(days=_CALENDAR_WINDOW_DAYS)
    events: list[dict[str, Any]] = []
    d = start_date
    for cycle in range(1, cycles + 1):
        for day in range(1, 6):
            if d >= cutoff:
                break
            events.append(_all_day_event(
                summary     = f"[{patient_id}] 💊 {drug_name} {dose} — Cycle {cycle} Day {day}/5",
                description = f"Temozolomide 5/23 schedule. Cycle {cycle}, day {day} of 5.\n"
                              f"Dose: {dose}",
                event_date  = d,
                color       = COLOR_TEAL,
                patient_id  = patient_id,
            ))
            d += timedelta(days=1)
        d += timedelta(days=23)   # 23-day rest period.
    return events


# ── event builders ────────────────────────────────────────────────────────────

def _all_day_event(
    summary: str,
    description: str,
    event_date: date,
    color: str = COLOR_BLUE,
    patient_id: str = "",
    attendee_email: str | None = None,
    reminders: list[dict] | None = None,
) -> dict[str, Any]:
    evt: dict[str, Any] = {
        "summary":     summary,
        "description": description,
        "start":       {"date": event_date.isoformat()},
        "end":         {"date": (event_date + timedelta(days=1)).isoformat()},
        "colorId":     color,
        "reminders": {
            "useDefault": False,
            "overrides": reminders or [
                {"method": "email",  "minutes": 1440},   # 1 day before
                {"method": "popup",  "minutes":  120},   # 2 hours before
            ],
        },
    }
    if attendee_email:
        evt["attendees"] = [
            {"email": DOCTOR_EMAIL},
            {"email": attendee_email},
        ]
    return evt


# ── CalendarClient ────────────────────────────────────────────────────────────

class CalendarClient:

    def __init__(self) -> None:
        self._svc: Any = None
        self._ready = False
        self._init()

    def _init(self) -> None:
        creds = _load_creds()
        if creds is None:
            return
        try:
            from googleapiclient.discovery import build
            self._svc = build("calendar", "v3", credentials=creds)
            self._ready = True
        except Exception as exc:
            log.warning("calendar: could not build service: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    def _event_exists(self, summary: str, event_date: date) -> bool:
        """Return True if an event with the same summary already exists on this date."""
        try:
            time_min = f"{event_date.isoformat()}T00:00:00Z"
            time_max = f"{(event_date + timedelta(days=1)).isoformat()}T00:00:00Z"
            results  = self._svc.events().list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                q=summary[:50],        # search by title prefix
                singleEvents=True,
                maxResults=10,
            ).execute()
            for item in results.get("items", []):
                if item.get("summary", "") == summary:
                    return True
            return False
        except Exception:
            return False   # if check fails, proceed with insert

    def _insert(self, event: dict[str, Any]) -> bool:
        summary    = event.get("summary", "")
        start_date_str = (event.get("start") or {}).get("date")
        if start_date_str:
            try:
                event_date = date.fromisoformat(start_date_str)
                if self._event_exists(summary, event_date):
                    log.info("calendar: skip duplicate [%s] on %s", summary, start_date_str)
                    return True   # already exists — not an error
            except Exception:
                pass
        try:
            self._svc.events().insert(
                calendarId="primary",
                body=event,
                sendUpdates="all",
            ).execute()
            log.info("calendar: created [%s]", summary)
            return True
        except Exception as exc:
            log.warning("calendar: insert failed [%s]: %s", summary, exc)
            return False

    # ── public methods ────────────────────────────────────────────────────────

    def create_medication_reminders(
        self,
        patient_id: str,
        medications: list[dict[str, Any]],
        patient_email: str | None = None,
    ) -> int:
        """Create recurring calendar reminders for each current medication.

        Returns the count of successfully created events.
        """
        if not self._ready:
            return 0

        created = 0
        today = date.today()

        for med in medications:
            name      = med.get("name", "Unknown drug")
            dose      = med.get("dose") or ""
            frequency = med.get("frequency") or "daily"
            route     = med.get("route") or ""
            start_raw = med.get("start_date")

            try:
                start_d = date.fromisoformat(str(start_raw)) if start_raw else today
            except ValueError:
                start_d = today

            # Anchor past-dated prescriptions to today — historical start_dates
            # from ingested records would create useless backdated reminders
            # (and get dropped by the dedupe step against present calendar state).
            if start_d < today:
                log.info(
                    "calendar: medication %s start_date %s is in the past — "
                    "anchoring to today (%s)", name, start_d, today,
                )
                start_d = today

            rrule, per_day = _rrule_from_frequency(frequency)

            # TMZ 5/23 — generate individual day events (no RRULE).
            if rrule == "TMZ_523":
                events = _tmz_523_events(name, dose, start_d, patient_id)
                for evt in events:
                    if patient_email:
                        evt["attendees"] = [{"email": DOCTOR_EMAIL}, {"email": patient_email}]
                    if self._insert(evt):
                        created += 1
                continue

            # Only show dose in event title when it contains a number
            # (avoids titles like "Dexamethasone mg/m²" when LLM omits the quantity).
            import re as _re
            dose_tag = f" {dose}" if dose and _re.search(r"\d", dose) else ""

            # Cap recurring events to _CALENDAR_WINDOW_DAYS from start date.
            # Appending UNTIL= to the RRULE stops the series after 1 month so
            # the doctor's calendar is not filled with years of reminders.
            until_str = (start_d + timedelta(days=_CALENDAR_WINDOW_DAYS)).strftime("%Y%m%dT235959Z")
            rrule_capped = f"{rrule};UNTIL={until_str}"

            # Standard recurring event.
            desc = (
                f"Drug      : {name}\n"
                f"Dose      : {dose or 'see prescription'}\n"
                f"Frequency : {frequency}\n"
                f"Route     : {route}\n\n"
                f"Patient   : {patient_id}\n"
                f"Auto-generated by Neuro-Oncology Agent."
            )
            for i in range(per_day):
                suffix = f" (dose {i+1}/{per_day})" if per_day > 1 else ""
                event = _all_day_event(
                    summary        = f"[{patient_id}] 💊 {name}{dose_tag}{suffix}",
                    description    = desc,
                    event_date     = start_d,
                    color          = COLOR_TEAL,
                    patient_id     = patient_id,
                    attendee_email = patient_email,
                )
                event["recurrence"] = [rrule_capped]
                if self._insert(event):
                    created += 1

        log.info("calendar: created %d medication events for %s", created, patient_id)
        return created

    def clear_agent_events(
        self,
        patient_id: str | None = None,
        days_back: int = 730,
        days_forward: int = 730,
    ) -> dict[str, int]:
        """Delete all calendar events created by this agent.

        Agent events are identified by the '[' prefix in the summary field
        (all agent-created events use the pattern '[PATIENT_ID] ...').

        Parameters
        ----------
        patient_id:
            When given, only delete events matching '[PATIENT_ID]' prefix.
            When None, delete ALL agent-created events for any patient.
        days_back:
            How far back in time to search (default 730 days = ~2 years).
        days_forward:
            How far forward to search (default 730 days).

        Returns
        -------
        dict with keys ``deleted`` (success count) and ``failed`` (error count).
        """
        if not self._ready:
            log.warning("calendar: clear_agent_events called but service not ready")
            return {"deleted": 0, "failed": 0}

        today     = date.today()
        time_min  = (today - timedelta(days=days_back)).isoformat() + "T00:00:00Z"
        time_max  = (today + timedelta(days=days_forward)).isoformat() + "T23:59:59Z"

        # Build query prefix — narrow to one patient or catch-all '[' character.
        if patient_id:
            q_prefix = f"[{patient_id.upper()}]"
        else:
            q_prefix = "["

        deleted = 0
        failed  = 0
        page_token: str | None = None

        log.info(
            "calendar: clearing agent events  prefix=%r  range=%s..%s",
            q_prefix, time_min[:10], time_max[:10],
        )

        while True:
            try:
                params: dict = dict(
                    calendarId   = "primary",
                    timeMin      = time_min,
                    timeMax      = time_max,
                    q            = q_prefix,
                    singleEvents = True,
                    maxResults   = 250,
                )
                if page_token:
                    params["pageToken"] = page_token

                result = self._svc.events().list(**params).execute()
                items  = result.get("items", [])

                for item in items:
                    summary = item.get("summary", "")
                    # Double-check summary matches the expected prefix pattern.
                    if not summary.startswith("["):
                        continue
                    if patient_id and not summary.startswith(q_prefix):
                        continue

                    event_id = item["id"]
                    try:
                        self._svc.events().delete(
                            calendarId="primary",
                            eventId=event_id,
                        ).execute()
                        log.info("calendar: deleted [%s] id=%s", summary, event_id)
                        deleted += 1
                    except Exception as exc:
                        log.warning("calendar: failed to delete [%s]: %s", summary, exc)
                        failed += 1

                page_token = result.get("nextPageToken")
                if not page_token:
                    break

            except Exception as exc:
                log.warning("calendar: list events failed: %s", exc)
                break

        log.info(
            "calendar: clear_agent_events done  deleted=%d  failed=%d",
            deleted, failed,
        )
        return {"deleted": deleted, "failed": failed}

    def create_therapy_sessions(
        self,
        patient_id: str,
        correlation_summary: str,
        patient_email: str | None = None,
    ) -> int:
        """Parse correlation/discharge free text and create therapy session events.

        Looks for radiotherapy fractions, infusion days, and steroid taper mentions.
        Returns count of created events.
        """
        if not self._ready or not correlation_summary:
            return 0

        created = 0
        today   = date.today()
        cutoff  = today + timedelta(days=_CALENDAR_WINDOW_DAYS)
        text    = correlation_summary.lower()

        # Radiotherapy fractions — daily consecutive events, naturally stay within
        # the window as long as n_fractions ≤ 30 (typical RT course is 30 fractions).
        rt_match = re.search(r"(\d+)\s*fraction", text)
        if rt_match:
            n_fractions = min(int(rt_match.group(1)), _CALENDAR_WINDOW_DAYS)
            for i in range(n_fractions):
                event_date = today + timedelta(days=i)
                if event_date >= cutoff:
                    break
                event = _all_day_event(
                    summary     = f"[{patient_id}] 🔬 RT Fraction {i+1}/{n_fractions}",
                    description = f"Radiotherapy fraction {i+1} of {n_fractions}.\nPatient: {patient_id}",
                    event_date  = event_date,
                    color       = COLOR_ORANGE,
                    attendee_email=patient_email,
                )
                if self._insert(event):
                    created += 1

        # Infusion / chemotherapy days — 3-week cycles, capped to window.
        # At 3-week spacing: cycle 1 = day 0, cycle 2 = day 21 (both within 30 days).
        if re.search(r"infusion|iv\s+chemo|bevacizumab|carboplatin|cisplatin", text):
            cycle_num = 1
            for week in range(6):
                event_date = today + timedelta(weeks=week * 3)
                if event_date >= cutoff:
                    break
                event = _all_day_event(
                    summary     = f"[{patient_id}] 💉 Chemotherapy Infusion (cycle {cycle_num})",
                    description = f"Scheduled infusion day.\nPatient: {patient_id}",
                    event_date  = event_date,
                    color       = COLOR_RED,
                    attendee_email=patient_email,
                )
                if self._insert(event):
                    created += 1
                cycle_num += 1

        # Steroid taper — weekly dose reduction steps, capped to window.
        # 4 steps × 7 days = 28 days (fits within 30-day window).
        if re.search(r"dexamethasone|dex\s+taper|steroid\s+taper|decadron", text):
            doses = ["8mg", "4mg", "2mg", "1mg", "0.5mg"]
            for i, d in enumerate(doses):
                event_date = today + timedelta(weeks=i)
                if event_date >= cutoff:
                    break
                event = _all_day_event(
                    summary     = f"[{patient_id}] 💊 Dexamethasone Taper — {d}",
                    description = f"Steroid taper step {i+1}/{len(doses)}: {d} daily.\nPatient: {patient_id}",
                    event_date  = event_date,
                    color       = COLOR_TEAL,
                    attendee_email=patient_email,
                )
                if self._insert(event):
                    created += 1

        log.info("calendar: created %d therapy session events for %s", created, patient_id)
        return created

    # ── Phase 4 Treatment Optimisation calendar hooks ──────────────────────────

    def create_mdt_meeting(
        self,
        patient_id: str,
        proposal: Any,
        patient_email: str | None = None,
    ) -> bool:
        """Create an MDT board meeting event 3 business days from today.

        Triggered when Phase 4 MDT reviewer sets mdt_discussion_required=True.
        """
        if not self._ready:
            return False

        from ..utils.schemas import TreatmentProposal
        prop = proposal if isinstance(proposal, TreatmentProposal) \
            else TreatmentProposal.model_validate(proposal)

        # Schedule 3 business days ahead
        today = date.today()
        meeting_date = today
        business_days = 0
        while business_days < 3:
            meeting_date += timedelta(days=1)
            if meeting_date.weekday() < 5:   # Mon–Fri
                business_days += 1

        summary = (
            f"[{patient_id}] 🏥 MDT Review — "
            f"Treatment Optimisation ({prop.decision})"
        )
        desc_lines = [
            f"Patient: {patient_id}",
            f"MDT Decision: {prop.decision}",
            f"Reason: {prop.reason}",
        ]
        if prop.proposed_regimen:
            desc_lines.append(f"Proposed regimen: {prop.proposed_regimen}")
        if prop.modifications:
            desc_lines.append(f"Modifications: {'; '.join(prop.modifications)}")
        if prop.rag_interaction_flags:
            desc_lines.append(f"Interaction flags: {', '.join(prop.rag_interaction_flags)}")
        desc_lines += [
            "",
            "Auto-scheduled by Neuro-Oncology SMBO v3.0 pipeline.",
            "⚠️ Review and confirm before prescribing.",
        ]

        event = _all_day_event(
            summary        = summary,
            description    = "\n".join(desc_lines),
            event_date     = meeting_date,
            color          = COLOR_ORANGE,
            patient_id     = patient_id,
            attendee_email = patient_email,
            reminders      = [
                {"method": "email", "minutes": 1440},   # 1 day before
                {"method": "popup", "minutes": 60},
            ],
        )
        ok = self._insert(event)
        if ok:
            log.info("calendar: MDT meeting created for %s on %s", patient_id, meeting_date)
        return ok

    def create_proposed_regimen_events(
        self,
        patient_id: str,
        proposed_regimen: str,
        patient_email: str | None = None,
    ) -> int:
        """Create [PROPOSED] calendar marker events for an SMBO-recommended regimen.

        Uses COLOR_ORANGE to distinguish from confirmed current-medication events.
        Creates a marker today + one at week 3 (expected first cycle assessment).
        Returns count of events created.
        """
        if not self._ready or not proposed_regimen:
            return 0

        today = date.today()
        events_to_create = [
            _all_day_event(
                summary     = f"[{patient_id}] 🔶 [PROPOSED] Start {proposed_regimen[:60]}",
                description = (
                    f"MDT-proposed regimen pending confirmation.\n"
                    f"Regimen: {proposed_regimen}\n"
                    f"Patient: {patient_id}\n"
                    f"⚠️ Proposal — confirm with MDT before prescribing."
                ),
                event_date     = today,
                color          = COLOR_ORANGE,
                attendee_email = patient_email,
            ),
            _all_day_event(
                summary     = (
                    f"[{patient_id}] 🔶 [PROPOSED] Cycle 1 Assessment — "
                    f"{proposed_regimen[:40]}"
                ),
                description = (
                    f"First response assessment for proposed regimen.\n"
                    f"Expected 3 weeks after proposed start date.\n"
                    f"Patient: {patient_id}"
                ),
                event_date     = today + timedelta(weeks=3),
                color          = COLOR_ORANGE,
                attendee_email = patient_email,
            ),
        ]
        created = sum(1 for ev in events_to_create if self._insert(ev))
        log.info("calendar: created %d proposed regimen events for %s", created, patient_id)
        return created

    def schedule_followup(
        self,
        patient_id: str,
        recist_response: str,
        urgency_score: int,
        patient_email: str | None = None,
        diagnosis: str = "",
        phase4_decision: str | None = None,
    ) -> bool:
        """Create the next scheduled appointment based on clinical state.

        When phase4_decision is APPROVE or MODIFY, override to a 4-week
        first response assessment for the new regimen.
        """
        if not self._ready:
            return False

        today = date.today()

        # Phase 4 override: when SMBO proposes a treatment change, schedule a
        # neutral oncology consultation +28d so the doctor and patient can
        # discuss the proposal. We DO NOT name the drug or imply commitment —
        # the regimen is only a proposal until the MDT meets and the patient
        # has given informed consent (SPIKES + consent policy).
        if phase4_decision in ("APPROVE", "MODIFY"):
            delta = timedelta(days=28)
            label = "Consultation with Oncologist — review proposed treatment plan"
            color = COLOR_TEAL
        elif urgency_score >= 5:
            delta, label, color = timedelta(days=7),  "🚨 URGENT Follow-Up",         COLOR_RED
        elif recist_response == "PD":
            delta, label, color = timedelta(days=14), "MDT Review — Progressive Disease", COLOR_ORANGE
        elif recist_response in ("PR", "CR"):
            delta, label, color = timedelta(days=42), "Confirmation Scan (RECIST CR/PR)", COLOR_TEAL
        else:  # SD, NE
            delta, label, color = timedelta(days=56), "Routine Surveillance MRI",     COLOR_BLUE

        appt_date = today + delta
        summary   = f"[{patient_id}] {label}"
        if diagnosis:
            summary += f" — {diagnosis[:50]}"

        desc = (
            f"Patient      : {patient_id}\n"
            f"RECIST       : {recist_response}\n"
            f"Urgency      : {urgency_score}/5\n"
            f"Diagnosis    : {diagnosis or 'see record'}\n"
            + (f"Phase 4 MDT  : {phase4_decision}\n" if phase4_decision else "")
            + "\nAuto-generated by Neuro-Oncology Unified Care Agent."
        )

        event = _all_day_event(
            summary        = summary,
            description    = desc,
            event_date     = appt_date,
            color          = color,
            patient_id     = patient_id,
            attendee_email = patient_email,
            reminders      = [
                {"method": "email", "minutes": 2880},
                {"method": "email", "minutes": 1440},
                {"method": "popup", "minutes":   60},
            ],
        )
        ok = self._insert(event)
        if ok:
            log.info("calendar: scheduled %s for %s on %s", label, patient_id, appt_date)
        return ok


# ── Module-level convenience wrappers (used by treatment_opt_agent.py) ────────

def create_mdt_meeting(
    patient_id: str,
    proposal: Any,
    patient_email: str | None = None,
) -> dict:
    """Module-level wrapper: instantiate CalendarClient and call create_mdt_meeting."""
    client = CalendarClient()
    if not client.ready:
        return {"ok": False, "error": "Calendar not configured"}
    ok = client.create_mdt_meeting(patient_id, proposal, patient_email=patient_email)
    return {"ok": ok}
