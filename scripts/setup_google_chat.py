"""Google Chat App registration guide + local tunnel setup.

Run from the PROJECT ROOT:
    python -m neuro_agent.scripts.setup_google_chat
    -- or --
    python neuro_agent/scripts/setup_google_chat.py

What this script does
─────────────────────
1. Checks that token.json exists (Gmail/Drive/Calendar OAuth already done).
2. Checks that ngrok is installed (needed to expose localhost to Google Chat).
3. Prints the exact configuration values to paste into Google Cloud Console.
4. Runs a local test against the bot endpoint to confirm it responds.

Why ngrok?
──────────
Google Chat needs a public HTTPS URL to call when a patient sends a message.
ngrok creates a secure tunnel from a random public URL → your localhost:8000.

    ngrok http 8000
    # → forwarding: https://abc123.ngrok-free.app → localhost:8000

Bot webhook URL = https://<your-ngrok-subdomain>.ngrok-free.app/api/v1/google-chat
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Package root = 2 levels up: scripts/ → neuro_agent/
ROOT       = Path(__file__).resolve().parent.parent
TOKEN_PATH = ROOT / "credentials" / "token.json"

# Ensure neuro_agent's parent is on sys.path so `neuro_agent` package is importable.
_PARENT = ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


def _check(ok: bool, label: str, fix: str = "") -> bool:
    icon = "✓" if ok else "✗"
    print(f"  {icon}  {label}")
    if not ok and fix:
        print(f"       → {fix}")
    return ok


def check_prereqs() -> bool:
    print("\n" + "=" * 60)
    print(" Pre-requisite Check")
    print("=" * 60)

    all_ok = True

    # OAuth token
    token_ok = TOKEN_PATH.exists()
    all_ok &= _check(
        token_ok,
        "OAuth token at credentials/token.json",
        "Run: python -m neuro_agent.scripts.setup_oauth" if not token_ok else "",
    )

    # google-auth installed
    try:
        import googleapiclient  # noqa: F401
        auth_ok = True
    except ImportError:
        auth_ok = False
    all_ok &= _check(
        auth_ok,
        "google-api-python-client installed",
        "Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client",
    )

    # ngrok — try multiple invocation styles (Windows PATH quirks)
    ngrok_ok  = False
    ngrok_ver = ""
    for cmd in (["ngrok", "version"], ["ngrok", "--version"], ["ngrok.exe", "version"]):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
                shell=(sys.platform == "win32"),
            )
            if result.returncode == 0 or "ngrok" in (result.stdout + result.stderr).lower():
                ngrok_ok  = True
                ngrok_ver = (result.stdout + result.stderr).strip().split("\n")[0]
                break
        except Exception:
            continue
    # Also accept if ngrok is already running (port 4040 = ngrok inspector)
    if not ngrok_ok:
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2)
            ngrok_ok  = True
            ngrok_ver = "(already running)"
        except Exception:
            pass
    all_ok &= _check(
        ngrok_ok,
        f"ngrok installed / running  {ngrok_ver}",
        "Download from https://ngrok.com/download  then run: ngrok http 8000",
    )

    # FastAPI running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8000/healthz", timeout=2)
        api_ok = True
    except Exception:
        api_ok = False
    _check(
        api_ok,
        "FastAPI server running on localhost:8000",
        "Run (in another terminal):  python run_server.py",
    )

    return all_ok


def print_google_console_steps() -> None:
    print("\n" + "=" * 60)
    print(" Google Cloud Console — Register the Chat App")
    print("=" * 60)
    print("""
STEP 1 — Enable the Google Chat API
─────────────────────────────────────────
  • Go to https://console.cloud.google.com
  • Select the same project you used for Gmail/Drive/Calendar OAuth
  • APIs & Services → Library → search "Google Chat API" → Enable

STEP 2 — Start ngrok (in a NEW terminal window)
─────────────────────────────────────────
  ngrok http 8000

  Copy the Forwarding URL that appears, e.g.:
      https://abc123.ngrok-free.app

  Your webhook URL will be:
      https://abc123.ngrok-free.app/api/v1/google-chat

  ⚠️  Keep ngrok running while testing. The URL changes each restart
     (unless you have an ngrok paid plan with a fixed domain).

STEP 3 — Create the Google Chat App
─────────────────────────────────────────
  • APIs & Services → Google Chat API → Configuration tab

  Fill in these fields:
  ┌─────────────────────────────────────────────────────────────┐
  │ App name:          Neuro-Oncology Care Bot                  │
  │ Avatar URL:        (leave blank or use any medical icon URL)│
  │ Description:       AI-powered patient care assistant        │
  │                                                             │
  │ Functionality:                                              │
  │   ✅ Receive 1:1 messages                                   │
  │   ✅ Join spaces and group conversations                    │
  │                                                             │
  │ Connection settings:                                        │
  │   ● App URL  (HTTP endpoint)                                │
  │   URL: https://abc123.ngrok-free.app/api/v1/google-chat    │
  │                                                             │
  │ Visibility:                                                 │
  │   ● Make this Chat app available to specific people and     │
  │     groups in Neuro Oncology                               │
  │   Add: krishnahutrik.n@gmail.com                           │
  │   (Add patient emails here if using Google Workspace)      │
  └─────────────────────────────────────────────────────────────┘

  Click SAVE

STEP 4 — Start a conversation with the bot (Dev test)
─────────────────────────────────────────
  Option A — Doctor tests:
    1. Open https://chat.google.com in your browser
    2. Click "New Chat" → search for "Neuro-Oncology Care Bot"
    3. Send a message: "Hello" → bot should reply with welcome message

  Option B — Test via API (no browser needed):
    1. Set env var: GOOGLE_CHAT_SKIP_AUTH=1
    2. Start server:  python run_server.py
    3. Open browser:  http://localhost:8000/api/v1/google-chat/test/P002?q=What+is+my+diagnosis

STEP 5 — Patient setup
─────────────────────────────────────────
  After a patient's pipeline completes, they automatically receive an email
  (sent by the pipeline) with instructions to find the bot in Google Chat.

  The patient:
    1. Opens https://chat.google.com
    2. Searches for "Neuro-Oncology Care Bot"
    3. Starts a direct message
    4. Asks any question about their records

  The bot:
    • Looks up their email → patient ID
    • Queries their ChromaDB collection using Gemma 4
    • Replies in plain language with citations
    • Fires an alert to the doctor if urgency keywords detected
""")


def print_patient_roster() -> None:
    print("=" * 60)
    print(" Patient Email Roster (P001–P020)")
    print("=" * 60)
    from neuro_agent.integrations.patient_roster import DOCTOR_EMAIL, PATIENT_EMAILS
    print(f"  Doctor : {DOCTOR_EMAIL}")
    print()
    for pid, email in PATIENT_EMAILS.items():
        print(f"  {pid} → {email}")


def run_local_test() -> None:
    """Quick smoke test against the local bot endpoint."""
    skip = os.environ.get("GOOGLE_CHAT_SKIP_AUTH", "")
    if not skip:
        print("\n[INFO] Set GOOGLE_CHAT_SKIP_AUTH=1 and restart to run local bot tests.")
        return

    print("\n" + "=" * 60)
    print(" Local Bot Smoke Test (GOOGLE_CHAT_SKIP_AUTH=1)")
    print("=" * 60)

    try:
        import json as _json
        import urllib.request
        fake_event = _json.dumps({
            "type": "MESSAGE",
            "user": {"email": "khadurgam@gmail.com", "displayName": "Test P002"},
            "message": {
                "text": "What is my diagnosis?",
                "thread": {"name": "spaces/TEST/threads/smoke"},
            },
            "space": {"name": "spaces/TEST", "type": "DM"},
        }).encode()

        req = urllib.request.Request(
            "http://localhost:8000/api/v1/google-chat",
            data=fake_event,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            reply = _json.loads(resp.read())
        print("\n  Question : What is my diagnosis?")
        print("  Patient  : P002 (khadurgam@gmail.com)")
        print("  Reply    :\n")
        text = reply.get("text", "(empty reply)")
        for line in text.splitlines():
            print(f"    {line}")
        print()
    except Exception as exc:
        print(f"\n  [WARN] Test failed: {exc}")
        print("  Make sure: python run_server.py is running")


def main() -> int:
    print("\n" + "=" * 60)
    print(" Neuro-Oncology Agent — Google Chat Bot Setup")
    print("=" * 60)

    all_ok = check_prereqs()
    print_google_console_steps()
    print_patient_roster()
    run_local_test()

    if all_ok:
        print("=" * 60)
        print(" Setup complete — follow STEP 3 above to register the bot.")
        print("=" * 60 + "\n")
        return 0
    else:
        print("=" * 60)
        print(" Fix the issues above (marked ✗) then re-run this script.")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
