"""Patient ID → email address mapping.

P001–P020 are the 20 test patients.  The doctor / sender account
(krishnahutrik.n@gmail.com) is stored separately as DOCTOR_EMAIL.
"""
from __future__ import annotations

DOCTOR_EMAIL = "krishnahutrik.n@gmail.com"

# Ordered P001 → P020 matching the reference dataset patients.
# HRK is the test/demo patient — mapped to the first patient email slot.
PATIENT_EMAILS: dict[str, str] = {
    "HRK":  "harish.krishna@testaing.com",
    "P001": "harish.krishna@testaing.com",
    "P002": "khadurgam@gmail.com",
    "P003": "harishkrishna549@gmail.com",
    "P004": "durgamhari461@gmail.com",
    "P005": "dkh7655@gmail.com",
    "P006": "hakr85071@gmail.com",
    "P007": "hak204647@gmail.com",
    "P008": "hakad6721@gmail.com",
    "P009": "krishnadharish@gmail.com",
    "P010": "ahs72927373@gmail.com",
    "P011": "krishnadharishg@gmail.com",
    "P012": "km970590553@gmail.com",
    "P013": "durgamm002@gmail.com",
    "P014": "maheshk748474@gmail.com",
    "P015": "ro4470815@gmail.com",
    "P016": "deepika.uniyal@testaing.com",
    "P017": "neelima@testaing.com",
    "P018": "srinivas@testaing.com",
    "P019": "jayapradeep@testaing.com",
    "P020": "shruti.butte@testaing.com",
}


def get_patient_email(patient_id: str) -> str | None:
    """Return email for a patient ID, or None if not in roster."""
    return PATIENT_EMAILS.get(patient_id.upper())


def get_patient_id(email: str) -> str | None:
    """Reverse lookup — email → patient ID."""
    email = email.lower()
    for pid, addr in PATIENT_EMAILS.items():
        if addr.lower() == email:
            return pid
    return None
