"""Google Workspace integrations — Gmail, Drive, Calendar.

All three clients share the same OAuth2 token (credentials/token.json).
They are imported lazily inside synthesis_agent so the pipeline still
runs in environments without google-api-python-client installed.
"""
from .calendar_client import CalendarClient
from .chat_bot import handle_message as handle_chat_message
from .drive_client import DriveClient
from .gmail_client import GmailClient
from .patient_roster import PATIENT_EMAILS, get_patient_email

__all__ = [
    "PATIENT_EMAILS",
    "get_patient_email",
    "GmailClient",
    "DriveClient",
    "CalendarClient",
    "handle_chat_message",
]
