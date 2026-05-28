"""One-time Google OAuth2 consent flow.

Run from the PROJECT ROOT:
    python -m neuro_agent.scripts.setup_oauth
    -- or --
    python neuro_agent/scripts/setup_oauth.py

What this script does
─────────────────────
1. Reads credentials/credentials.json (downloaded from Google Cloud Console).
2. Opens your browser at Google's consent screen.
3. Sign in with krishnahutrik.n@gmail.com and click Allow.
4. Saves credentials/token.json — the pipeline uses this from here on.
5. Auto-refreshes the token on every pipeline run (silent, no browser needed).

Prerequisites (one-time Google Cloud Console steps)
────────────────────────────────────────────────────
1. Go to https://console.cloud.google.com
2. Create a project (or select an existing one).
3. Enable these APIs:
       APIs & Services → Library → search each:
       • Gmail API           → Enable
       • Google Drive API    → Enable
       • Google Calendar API → Enable
4. Configure OAuth consent screen:
       APIs & Services → OAuth consent screen
       • User Type: External
       • App name: Neuro-Oncology Agent (dev)
       • Support email: krishnahutrik.n@gmail.com
       • Add test user: krishnahutrik.n@gmail.com
       • Scopes: (leave blank here — added by the app)
       • Save and Continue through all steps.
5. Create credentials:
       APIs & Services → Credentials → Create Credentials → OAuth client ID
       • Application type: Desktop app
       • Name: neuro-oncology-desktop
       • Download JSON → save as:
             credentials/credentials.json
6. Run this script:
       python -m neuro_agent.scripts.setup_oauth
"""
from __future__ import annotations

import sys
from pathlib import Path

# Package root = 2 levels up: scripts/ → neuro_agent/
ROOT = Path(__file__).resolve().parent.parent

# Search for credentials.json in order of preference:
#   1. neuro_agent/credentials/credentials.json   (canonical)
#   2. Current working directory / credentials.json
#   3. Current working directory / credentials / credentials.json
_CRED_CANDIDATES = [
    ROOT / "credentials" / "credentials.json",
    Path.cwd() / "credentials.json",
    Path.cwd() / "credentials" / "credentials.json",
]
CREDENTIALS_PATH = next(
    (p for p in _CRED_CANDIDATES if p.exists()),
    ROOT / "credentials" / "credentials.json",   # default (shown in error msg)
)
TOKEN_PATH = ROOT / "credentials" / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/calendar",
]


def _check_deps() -> bool:
    missing = []
    for pkg in ("google.auth", "google_auth_oauthlib", "googleapiclient"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[ERROR] Missing packages. Install them first:\n")
        print("    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client\n")
        return False
    return True


def _check_credentials() -> bool:
    if not CREDENTIALS_PATH.exists():
        print(f"\n[ERROR] credentials.json not found at:\n    {CREDENTIALS_PATH}\n")
        print("Follow the steps at the top of this file to download it from")
        print("Google Cloud Console, then re-run this script.\n")
        print("Quick guide:")
        print("  1. https://console.cloud.google.com → APIs & Services → Credentials")
        print("  2. Create Credentials → OAuth client ID → Desktop app")
        print("  3. Download JSON → rename to 'credentials.json'")
        print(f"  4. Move it to: {CREDENTIALS_PATH}\n")
        return False
    return True


def run_oauth_flow() -> None:
    """Interactive OAuth flow — opens browser, saves token."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    print("\n" + "=" * 60)
    print(" Neuro-Oncology Agent — Google OAuth Setup")
    print("=" * 60)
    print(f"\nCredentials : {CREDENTIALS_PATH}")
    print(f"Token will be saved to: {TOKEN_PATH}")
    print(f"\nScopes requested:")
    for s in SCOPES:
        print(f"  • {s}")
    print("\nA browser window will open. Sign in with:")
    print("    krishnahutrik.n@gmail.com")
    print("\nThen click 'Allow' on the consent screen.")
    print("-" * 60)

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDENTIALS_PATH),
        scopes=SCOPES,
    )
    creds = flow.run_local_server(
        port=0,
        prompt="consent",
        access_type="offline",
    )

    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"\n[OK] Token saved to: {TOKEN_PATH}")


def verify_gmail(creds) -> bool:
    import base64
    from email.mime.text import MIMEText
    from googleapiclient.discovery import build

    try:
        svc  = build("gmail", "v1", credentials=creds)
        msg  = MIMEText("OAuth setup successful — Neuro-Oncology Agent is authorised.")
        msg["From"]    = "krishnahutrik.n@gmail.com"
        msg["To"]      = "krishnahutrik.n@gmail.com"
        msg["Subject"] = "[Neuro-Oncology Agent] OAuth setup complete ✓"
        raw  = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        print("[OK] Gmail test — confirmation email sent to krishnahutrik.n@gmail.com")
        return True
    except Exception as exc:
        print(f"[WARN] Gmail test failed: {exc}")
        return False


def verify_drive(creds) -> bool:
    from googleapiclient.discovery import build

    try:
        svc = build("drive", "v3", credentials=creds)
        folder = svc.files().create(
            body={
                "name": "Neuro-Oncology Agent (OAuth test — safe to delete)",
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        ).execute()
        print(f"[OK] Drive test — folder created: id={folder['id']}")
        return True
    except Exception as exc:
        print(f"[WARN] Drive test failed: {exc}")
        return False


def verify_calendar(creds) -> bool:
    from googleapiclient.discovery import build

    try:
        svc = build("calendar", "v3", credentials=creds)
        cal = svc.calendars().get(calendarId="primary").execute()
        print(f"[OK] Calendar test — primary calendar: {cal.get('summary')}")
        return True
    except Exception as exc:
        print(f"[WARN] Calendar test failed: {exc}")
        return False


def main() -> int:
    if not _check_deps():
        return 1
    if not _check_credentials():
        return 1

    run_oauth_flow()

    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    print("\n" + "-" * 60)
    print(" Verifying API access …")
    print("-" * 60)
    gmail_ok    = verify_gmail(creds)
    drive_ok    = verify_drive(creds)
    calendar_ok = verify_calendar(creds)

    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"  Gmail    : {'✓ ready' if gmail_ok    else '✗ check scopes'}")
    print(f"  Drive    : {'✓ ready' if drive_ok    else '✗ check scopes'}")
    print(f"  Calendar : {'✓ ready' if calendar_ok else '✗ check scopes'}")
    print()

    if gmail_ok and drive_ok and calendar_ok:
        print("[SUCCESS] All Google APIs authorised.")
        print(f"          Token stored at: {TOKEN_PATH}")
        print("          The pipeline will use this token automatically.\n")
        return 0
    else:
        print("[PARTIAL] Some APIs failed — re-run setup or check Cloud Console scopes.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
