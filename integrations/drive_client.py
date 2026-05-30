"""Google Drive API client — mirror outputs/<pid>/ into a shared Drive folder.

Folder structure created under the doctor's Drive:
    Neuro-Oncology Agent/
        Patients/
            {pid}/
                Visit-{YYYY-MM-DD}/
                    report.md
                    fhir_bundle.json
                    patient_letter.txt
                    gp_handover.txt
                    S1_ingestion.json … S13_export.json
                    laboratory_results.json
                    P{pid}_full_pipeline.json
                    extended/
                        radiology_reports.json
                        pathology_report.txt
                        ...

Permissions:
    Doctor (krishnahutrik.n@gmail.com) — owner (implicit, it's their Drive).
    Patient                            — reader (view-only by email).

Token: shared credentials/token.json with scope drive.file
       (only files created by this app are visible — no full Drive access).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PKG             = Path(__file__).resolve().parent.parent   # neuro_agent/ package root
TOKEN_PATH       = _PKG / "credentials" / "token.json"
# All three scopes — token must be generated with all of them via setup_oauth.py
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/calendar",
]
ROOT_FOLDER_NAME = "Neuro-Oncology Agent"


def _load_creds():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        log.warning("drive: google-auth not installed")
        return None

    if not TOKEN_PATH.exists():
        log.warning("drive: token.json not found — run setup_oauth.py first")
        return None

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            log.warning("drive: token refresh failed: %s", exc)
            return None
    return creds if creds.valid else None


class DriveClient:
    """Sync a local output directory to Google Drive."""

    def __init__(self) -> None:
        self._svc: Any = None
        self._ready = False
        self._folder_cache: dict[str, str] = {}
        self._init()

    def _init(self) -> None:
        creds = _load_creds()
        if creds is None:
            return
        try:
            from googleapiclient.discovery import build
            self._svc = build("drive", "v3", credentials=creds)
            self._ready = True
        except Exception as exc:
            log.warning("drive: could not build service: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    # ── folder helpers ───────────────────────────────────────────────────────

    def _find_or_create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Return Drive folder id, creating it if absent."""
        cache_key = f"{parent_id or 'root'}:{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        q = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"

        res = self._svc.files().list(q=q, fields="files(id)").execute()
        files = res.get("files", [])
        if files:
            fid = files[0]["id"]
        else:
            meta: dict[str, Any] = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_id:
                meta["parents"] = [parent_id]
            fid = self._svc.files().create(body=meta, fields="id").execute()["id"]

        self._folder_cache[cache_key] = fid
        return fid

    def _upload_file(self, local_path: Path, parent_id: str) -> str | None:
        """Upload or update a file in Drive; return file id."""
        import mimetypes

        from googleapiclient.http import MediaFileUpload

        mime, _ = mimetypes.guess_type(local_path.name)
        mime = mime or "application/octet-stream"

        # Check if file already exists (update instead of duplicate).
        q = f"name='{local_path.name}' and '{parent_id}' in parents and trashed=false"
        existing = self._svc.files().list(q=q, fields="files(id)").execute().get("files", [])

        media = MediaFileUpload(str(local_path), mimetype=mime, resumable=False)
        if existing:
            fid = self._svc.files().update(
                fileId=existing[0]["id"],
                media_body=media,
            ).execute()["id"]
        else:
            fid = self._svc.files().create(
                body={"name": local_path.name, "parents": [parent_id]},
                media_body=media,
                fields="id",
            ).execute()["id"]
        return fid

    def _grant_reader(self, file_or_folder_id: str, email: str) -> None:
        """Give a specific Google account read access."""
        try:
            self._svc.permissions().create(
                fileId=file_or_folder_id,
                body={"type": "user", "role": "reader", "emailAddress": email},
                sendNotificationEmail=False,
            ).execute()
        except Exception as exc:
            log.warning("drive: could not grant reader to %s: %s", email, exc)

    # ── public API ───────────────────────────────────────────────────────────

    def sync_patient_outputs(
        self,
        patient_id: str,
        outputs_dir: Path,
        patient_email: str | None = None,
        visit_date: str | None = None,
    ) -> str | None:
        """
        Mirror outputs_dir into Drive:
            Neuro-Oncology Agent / Patients / {pid} / Visit-{date} /

        Returns the Drive folder URL for the visit folder, or None on failure.
        """
        if not self._ready:
            return None

        if not outputs_dir.exists():
            log.warning("drive: outputs_dir does not exist: %s", outputs_dir)
            return None

        visit_label = visit_date or date.today().isoformat()

        # Skip images/ folder — large binary files, not useful in Drive.
        _SKIP_DIRS = {"images"}

        # P001-RUN-FIX (#8): exclude pipeline-internal state from the
        # patient-facing Drive folder. The previous build uploaded all 41
        # files including ``working_memory.json`` (raw stage payloads),
        # ``audit.jsonl`` (HMAC-salted patient-id audit trail), and the
        # OAuth token sentinel — none of which a patient or partner
        # clinician should see. The numbered S## stage envelopes and
        # human-readable reports/plots/fhir bundles are still synced.
        _SKIP_FILE_NAMES = {
            "working_memory.json",   # full internal state dump
            "run_manifest.json",     # phase-log + timing telemetry
            "audit.jsonl",           # HMAC-salted access log
            ".smart_token.json",     # OAuth tokens (mode 0600 — never share)
            ".smart_token.json.tmp", # half-written token from atomic save
            ".build_status.json",    # LightRAG worker sentinel
            ".gitignore",            # repo housekeeping
        }
        # Hidden dotfiles also skipped wholesale (covers .DS_Store, .smart_*
        # variants, etc.).  Glob-style file extensions are dropped too:
        _SKIP_FILE_SUFFIXES = {".tmp", ".lock", ".part"}

        uploaded = failed = skipped = 0
        try:
            root_id  = self._find_or_create_folder(ROOT_FOLDER_NAME)
            pts_id   = self._find_or_create_folder("Patients",              root_id)
            pid_id   = self._find_or_create_folder(patient_id,              pts_id)
            visit_id = self._find_or_create_folder(f"Visit-{visit_label}",  pid_id)

            # Grant patient view-only access on their folder.
            if patient_email:
                self._grant_reader(pid_id, patient_email)
                log.info("drive: granted reader access to %s for %s", patient_email, patient_id)

            for item in sorted(outputs_dir.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(outputs_dir)
                # Skip the images/ subfolder — too large, no clinical value in Drive.
                if rel.parts[0] in _SKIP_DIRS:
                    continue
                # Skip pipeline-internal state files (#8 fix).
                if (
                    item.name in _SKIP_FILE_NAMES
                    or item.name.startswith(".")
                    or item.suffix.lower() in _SKIP_FILE_SUFFIXES
                ):
                    skipped += 1
                    log.debug("drive: skipping internal file %s", rel)
                    continue
                try:
                    if len(rel.parts) > 1:
                        sub_id = self._find_or_create_folder(rel.parts[0], visit_id)
                        self._upload_file(item, sub_id)
                    else:
                        self._upload_file(item, visit_id)
                    uploaded += 1
                except Exception as file_exc:
                    log.warning("drive: failed to upload %s: %s", item.name, file_exc)
                    failed += 1

            log.info(
                "drive: synced %s → Visit-%s  uploaded=%d  failed=%d  skipped_internal=%d",
                patient_id, visit_label, uploaded, failed, skipped,
            )
            return f"https://drive.google.com/drive/folders/{visit_id}"
        except Exception as exc:
            log.warning("drive: sync failed for %s: %s", patient_id, exc)
            return None

    def upload_phase4_reports(
        self,
        patient_id: str,
        out_dir: "Path",
    ) -> dict:
        """Upload Phase 4 output files (S14–S18, plots) to Drive.

        Uploads to Neuro-Oncology Agent/Patients/{pid}/Phase4-SMBO/.
        Best-effort: individual file failures are logged but don't abort.
        Returns {"uploaded": n, "failed": n, "folder_url": str|None}.
        """
        from pathlib import Path as _Path

        if not self._ready:
            return {"uploaded": 0, "failed": 0, "folder_url": None}

        phase4_files = [
            "S14_patient_state.json",
            "S15_prediction.json",
            "S16_optimization.json",
            "S17_shap.json",
            "S18_treatment_proposal.json",
        ]
        plot_names = ["bo_convergence.png", "bo_landscape.png", "shap_waterfall.png"]

        files_to_upload: list[_Path] = []
        for fname in phase4_files:
            p = _Path(out_dir) / fname
            if p.exists():
                files_to_upload.append(p)

        plots_dir = _Path(out_dir) / "plots"
        for pname in plot_names:
            p = plots_dir / pname
            if p.exists():
                files_to_upload.append(p)

        if not files_to_upload:
            log.info("drive: no Phase 4 files to upload for %s", patient_id)
            return {"uploaded": 0, "failed": 0, "folder_url": None}

        uploaded = failed = 0
        folder_url: str | None = None
        try:
            root_id   = self._find_or_create_folder(ROOT_FOLDER_NAME)
            pts_id    = self._find_or_create_folder("Patients",       root_id)
            pid_id    = self._find_or_create_folder(patient_id,       pts_id)
            phase4_id = self._find_or_create_folder("Phase4-SMBO",   pid_id)
            folder_url = f"https://drive.google.com/drive/folders/{phase4_id}"

            for fpath in files_to_upload:
                try:
                    self._upload_file(fpath, phase4_id)
                    uploaded += 1
                except Exception as exc:
                    log.warning("drive: phase4 upload failed %s: %s", fpath.name, exc)
                    failed += 1
        except Exception as exc:
            log.warning("drive: phase4 folder creation failed for %s: %s", patient_id, exc)

        log.info("drive: Phase4 upload %s — uploaded=%d  failed=%d", patient_id, uploaded, failed)
        return {"uploaded": uploaded, "failed": failed, "folder_url": folder_url}


# ── Module-level convenience wrappers (used by treatment_opt_agent.py) ────────

def upload_phase4_reports(patient_id: str, out_dir: "Path") -> dict:
    """Module-level wrapper: instantiate DriveClient and call upload_phase4_reports."""
    client = DriveClient()
    if not client.ready:
        return {"ok": False, "error": "Drive not configured"}
    result = client.upload_phase4_reports(patient_id, out_dir)
    return {"ok": result.get("failed", 1) == 0, **result}
