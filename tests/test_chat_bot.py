"""Unit tests for the Google Chat bot logic (no external calls)."""
import pytest


def test_is_urgent_false():
    from neuro_agent.integrations.chat_bot import _is_urgent
    assert not _is_urgent("What is my next appointment?")


def test_is_urgent_true():
    from neuro_agent.integrations.chat_bot import _is_urgent
    assert _is_urgent("I am having a seizure right now")
    assert _is_urgent("chest pain and can't breathe")


def test_patient_lookup_unknown():
    from neuro_agent.integrations.patient_roster import get_patient_id
    result = get_patient_id("nobody@nowhere.com")
    assert result is None


def test_patient_lookup_known():
    from neuro_agent.integrations.patient_roster import PATIENT_EMAILS, get_patient_id
    # Pick the first patient and verify reverse lookup works
    pid, email = next(iter(PATIENT_EMAILS.items()))
    assert get_patient_id(email) == pid


def test_handle_added_to_space():
    from neuro_agent.integrations.chat_bot import handle_message
    reply = handle_message({"type": "ADDED_TO_SPACE"})
    assert "text" in reply
    assert len(reply["text"]) > 10


def test_handle_unknown_patient(monkeypatch):
    from neuro_agent.integrations import chat_bot
    reply = chat_bot.handle_message({
        "type": "MESSAGE",
        "user": {"email": "stranger@example.com"},
        "message": {"text": "Hello", "thread": {"name": "t/1"}},
        "space": {"name": "s/1", "type": "DM"},
    })
    assert "text" in reply
    # Should return a "not registered" style message, not crash
    assert len(reply["text"]) > 5
