"""Central configuration for the Neuro-Oncology agent.

All paths, model names, thresholds, and rules live here so tools, the
orchestrator and the API layer import a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent  # project root
PKG = Path(__file__).resolve().parent          # neuro_agent/

# ---------- Unified dataset layout (v2) ----------
#   Datasets/
#   ├── patients/
#   │   └── P001/
#   │       ├── clinical/   (MRI images, DCM, PDFs — from legacy raw_docs/)
#   │       └── phase4/     (patient_intake_form.json, wearable_data.json)
#   └── reference/          (drug_encoding, drug_interaction_kb, nccn guidelines, reference_ranges)
#
# Legacy paths (raw_docs/, phase4_patient_data/, data/) still resolved as
# fallbacks for backwards compatibility with older pipeline runs.
DATASETS_ROOT     = ROOT / "Datasets"
PATIENTS_ROOT     = DATASETS_ROOT / "patients"         # v2 unified per-patient layout
REFERENCE_ROOT    = DATASETS_ROOT / "reference"        # v2 shared reference data

_legacy_raw_docs  = DATASETS_ROOT / "raw_docs"
_legacy_phase4    = DATASETS_ROOT / "phase4_patient_data"
_legacy_data      = DATASETS_ROOT / "data"

# DATA_ROOT: primary root for per-patient clinical files. Prefer unified v2 layout
# (patients/), fall back to legacy raw_docs/ if v2 doesn't exist.
if PATIENTS_ROOT.exists():
    DATA_ROOT = PATIENTS_ROOT                 # v2: patients/<pid>/clinical/*
else:
    _candidates = [_legacy_raw_docs, ROOT / "Dataset", PKG / "data"]
    DATA_ROOT = next((p for p in _candidates if p.exists()), _candidates[0])

# PHASE4_DATA_ROOT: where Phase 4 structured JSON lives.
#   v2: patients/<pid>/phase4/*.json
#   v1: phase4_patient_data/<pid>/*.json
if PATIENTS_ROOT.exists():
    PHASE4_DATA_ROOT = PATIENTS_ROOT           # resolved per-patient in ingest as <pid>/phase4/
else:
    PHASE4_DATA_ROOT = _legacy_phase4

# Reference data (shared across all patients)
if REFERENCE_ROOT.exists():
    DATA_REF_DIR = REFERENCE_ROOT
else:
    DATA_REF_DIR = _legacy_data
DRUG_ENCODING_PATH        = DATA_REF_DIR / "drug_encoding.json"
NCCN_GUIDELINES_PATH      = DATA_REF_DIR / "nccn_guidelines_summary.json"
REFERENCE_RANGES_PATH     = DATA_REF_DIR / "reference_ranges.json"
DRUG_TOXICITY_PROFILE_PATH = DATA_REF_DIR / "drug_toxicity_profiles.json"

# Drug interaction KB — in v2 it moves from raw_docs/ root into reference/
_drug_kb_v2 = DATA_REF_DIR / "drug_interaction_kb.json"
_drug_kb_v1 = _legacy_raw_docs / "drug_interaction_kb.json"
DRUG_KB_PATH = _drug_kb_v2 if _drug_kb_v2.exists() else _drug_kb_v1

OUTPUTS_DIR = PKG / "outputs"
UPLOADS_DIR = PKG / "uploads"
CHROMA_DIR = PKG / "chroma_db"
AUDIT_LOG = PKG / "audit.jsonl"
PROMPTS_DIR = PKG / "prompts"
MODELS_DIR  = PKG / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- v2 per-patient output layout ----------
# outputs/<pid>/{master,stages,reports,plots,images,fhir,extended,notifications}/
OUTPUT_SUBDIRS = {
    "master":         "master",         # P001_master.json, executive_summary.txt, run_manifest.json, working_memory.json
    "stages":         "stages",         # S01_ingestion.json … S18_treatment_proposal.json
    "reports":        "reports",        # report.md, gp_handover.txt, patient_letter.txt, mdt_package.json
    "plots":          "plots",          # bo_convergence.png, bo_landscape.png, shap_waterfall.png
    "images":         "images",         # v1/mri_norm.png (etc.)
    "fhir":           "fhir",           # fhir_bundle.json
    "extended":       "extended",       # pathology_report.txt, radiology_reports.json, …
    "notifications":  "notifications",  # gmail.json, sync.json, phase4.json
}


def patient_out_dir(pid: str, kind: str | None = None) -> Path:
    """Return the per-patient output directory, optionally scoped to a subdir."""
    base = OUTPUTS_DIR / pid
    if kind is None:
        return base
    sub = OUTPUT_SUBDIRS.get(kind, kind)
    d = base / sub
    d.mkdir(parents=True, exist_ok=True)
    return d

for d in (OUTPUTS_DIR, UPLOADS_DIR, CHROMA_DIR, PROMPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- models ----------
# Two-model architecture to avoid hallucination:
#   MODEL_PRIMARY  = qwen3:14b  — text reasoning, structured analysis, Q&A,
#                                 drug interactions, synthesis, orchestrator
#   MODEL_VISION   = gemma4:e4b — multimodal: MRI images, prescription scans,
#                                 RECIST lesion detection from images
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_PRIMARY  = os.environ.get("NEURO_MODEL",        "qwen3:14b")   # text + reasoning
MODEL_VISION   = os.environ.get("NEURO_MODEL_VISION", "gemma4:e4b")  # image analysis
MODEL_THINKING = os.environ.get("NEURO_MODEL_THINK",  "qwen3:14b")   # complex reasoning
MODEL_EMBED    = os.environ.get("NEURO_MODEL_EMBED",  "nomic-embed-text")
MODEL_FALLBACK_TEXT   = "qwen3:14b"
MODEL_FALLBACK_VISION = "gemma4:e4b"

# ---------- LLM defaults ----------
LLM_TEMPERATURE = 0.0
LLM_NUM_CTX = 16384
LLM_TIMEOUT_S = 600
LLM_MAX_RETRIES = 3

# ---------- RAG ----------
CHUNK_SIZE = 500            # tokens (approx via whitespace split)
CHUNK_OVERLAP = 50
RAG_TOP_K = 6
CHROMA_COLLECTION_FMT = "patient_{pid}"

# ---------- RECIST 1.1 rules ----------
RECIST_PR_THRESHOLD = -0.30  # >=30% decrease in sum of diameters
RECIST_PD_THRESHOLD = 0.20   # >=20% increase
RECIST_MIN_LESION_MM = 10.0

# ---------- Urgency triage rules (1..5) ----------
URGENCY_KEYWORDS_CRITICAL = [
    "hemorrhage", "herniation", "midline shift", "hydrocephalus",
    "status epilepticus", "rapid progression",
]
URGENCY_KEYWORDS_HIGH = [
    "mass effect", "edema", "new lesion", "enhancement",
    "progression", "recurrence",
]

# ---------- Ingestion keyword hints ----------
FILE_KEYWORDS = {
    "mri_image": ["mri", "brain", "axial", "scan", "t1", "t2", "flair"],
    "mri_report": ["mri_brain_report", "radiology_report", "mri_report"],
    "prescription": ["prescription", "rx", "medication_list"],
    "discharge": ["discharge"],
    "pathology": ["pathology", "histology", "biopsy"],
    "lab": ["lab", "laboratory", "blood"],
    "protocol": ["protocol", "treatment_plan", "regimen"],
    "timeline":     ["longitudinal_timeline", "timeline"],
    "correlation":  ["treatment_response_correlation", "response_correlation"],
    # Phase 4 structured data files (from phase4_patient_data/)
    "wearable":     ["wearable_data", "fitbit", "wearable"],
    "intake_form":  ["patient_intake_form", "intake_form"],
}

# Keywords that indicate a new lesion has appeared (RECIST 1.1 automatic PD trigger).
NEW_LESION_KEYWORDS = [
    "new lesion", "new enhancing", "new satellite", "new focus",
    "new foci", "new metastasis", "new metastases", "previously not seen",
    "not present on prior", "new area of enhancement", "new tumour",
    "new tumor",
]

# ---------- Phase 4 SMBO constants ----------
SMBO_BUDGET           = 60      # max SMBO iterations
SMBO_BATCH_SIZE       = 10      # candidates per iteration
SMBO_EI_POOL          = 5       # EI candidates per batch
SMBO_PRED_POOL        = 5       # PRED candidates per batch
SMBO_SIGMA_TRIGGER    = 0.25    # GP uncertainty → trigger full optimization
SMBO_SIGMA_EARLY_STOP = 0.12    # early-stop when best candidate σ < this
SMBO_LAMBDA           = 0.5     # RECIST penalty weight in scoring
SMBO_WARM_START_N     = 5       # NCCN SOC anchors for warm-start
SMBO_ANCHOR_WEIGHT    = 100     # replication factor for warm-start anchors (amplifies weight)
RECIST_DELTA_SD_TRIGGER = 15.0  # SD + GP predicted delta > this% → trigger optimization

# ---------- Clinical safety: critical biomarker hard-stop (Task 3) ----------
# For these cancer types, MGMT promoter methylation status AND IDH mutation
# status are decision-driving biomarkers. If either is missing/unknown the
# pipeline MUST halt Phase 4 (Treatment Optimisation) with a SKIP decision
# rather than impute — recommending a regimen without these markers is
# clinically dangerous.
BIOMARKER_REQUIRED_CANCERS = {
    "glioblastoma",
    "gbm",
    "glioma",
    "astrocytoma grade 3",
    "astrocytoma grade 4",
    "oligodendroglioma",
}
# Keywords that signal a brain primary (used by RANO branch, Task 4).
BRAIN_TUMOUR_KEYWORDS = [
    "glioma", "glioblastoma", "gbm", "astrocytoma",
    "oligodendroglioma", "ependymoma", "medulloblastoma", "meningioma",
]
# Multi-objective SMBO toxicity weight (Task 6) — tunable.
SMBO_TOXICITY_WEIGHT = 0.4
# Clinical trial matching trigger (Task 8).
TRIAL_MATCH_PFS_THRESHOLD_WEEKS = 26
# ---------- Developer / demo mode ----------
# DEV_MODE=True runs the full end-to-end pipeline in a single /api/v1/run call:
#   • APPROVAL_REQUIRED=False       — no clinician sign-off required
#   • biomarker hard-stop demoted   — 4b–4e run, RSF/GP/RFR .pkl files train
#   • SKIP phase-gate disabled      — pharma + synthesis always execute
#
# Default is True so /api/v1/run "just works" for development/demo.
# Set DEV_MODE=false in the environment (or edit here) for production HITL.
# NEVER ship production with DEV_MODE=True — bypasses Tasks 3 and 9 safety controls.
DEV_MODE = os.environ.get("DEV_MODE", "true").strip().lower() in ("1", "true", "yes", "on")

# HITL gate default (Task 9). Forced False when DEV_MODE is on.
APPROVAL_REQUIRED = False if DEV_MODE else True

# ---------- Volumetric segmentation (Task 5) ----------
# Backend selector: "nnunet" | "monai" | "none".
# "none" keeps the 2D PNG fallback and logs a warning; still emits the
# S04c_volumetric.json file with `volumetric_unavailable=true`.
BRATS_BACKEND = os.getenv("BRATS_BACKEND", "none").lower()

# ---------- Phase 5.1 — Radiomics (Module 4) ----------
# When True (default), the MRI phase runs extract_radiomics after
# segment_volumetric. Graceful-degrades to radiomics_unavailable=true
# if pyradiomics / SimpleITK / a mask aren't available.
RADIOMICS_ENABLED = os.environ.get(
    "RADIOMICS_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")

# ---------- Phase 5.2 / Module 5 — Pharmacogenomics ----------
# When True (default), SMBO candidate scoring multiplies toxicity and
# efficacy by per-drug CYP-phenotype factors from pgx_drug_map.json.
# If the drug or phenotype isn't found → 1.0 multipliers (no-op).
PGX_ENABLED = os.environ.get(
    "PGX_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Drug → enzyme → phenotype → (tox, eff) multipliers
PGX_DRUG_MAP_PATH      = DATA_REF_DIR / "pgx_drug_map.json"
# Enzyme → strength → inhibitor-drug list. Reserved for Phase 5.2b
# (dynamic phenoconversion); referenced by _resolve_effective_phenotype()
# but not yet consulted.
CYP_INHIBITORS_PATH    = DATA_REF_DIR / "cyp_inhibitors.json"

# ---------- Phase 5.4 / Module 1 — LightRAG (GraphRAG) ----------
# When True (default), recist_agent.index_rag() dual-writes to Chroma
# (synchronous) AND submits a background LightRAG graph-build job.
# pharma_agent's semantic fallback consults LightRAG hybrid query first
# and falls back to ChromaDB when the graph is still 'building' or
# the lightrag package isn't installed (graceful degrade).
LIGHTRAG_ENABLED = os.environ.get(
    "LIGHTRAG_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Per-patient LightRAG working dir (graphs live alongside ChromaDB).
LIGHTRAG_WORKING_DIR        = CHROMA_DIR / "_lightrag"
LIGHTRAG_SHARED_WORKING_DIR = CHROMA_DIR / "_lightrag_shared"  # drug KB
LIGHTRAG_WORKING_DIR.mkdir(parents=True, exist_ok=True)
LIGHTRAG_SHARED_WORKING_DIR.mkdir(parents=True, exist_ok=True)

# LightRAG must use the same embedder as ChromaDB so dual-written semantic
# neighbours are coherent. nomic-embed-text is the existing Ollama model.
LIGHTRAG_LLM_MODEL       = os.environ.get("LIGHTRAG_LLM_MODEL", "qwen3:14b")
LIGHTRAG_EMBED_MODEL     = os.environ.get("LIGHTRAG_EMBED_MODEL", "nomic-embed-text")
LIGHTRAG_OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Background graph-build worker timeout: how long finalize() will wait
# for a pending LightRAG ingest before giving up. The graph is append-
# capable on next run, so accepting partial graphs is safe.
LIGHTRAG_FLUSH_TIMEOUT_S = int(os.environ.get("LIGHTRAG_FLUSH_TIMEOUT_S", "30"))

# ---------- Phase 5.5 / Module 3 — PubMed Evidence RAG ----------
# Authenticated NCBI E-utilities tier: 10 req/s. Without the key the public
# tier is throttled to 3 req/s. Mixed-case .env keys are tolerated (the user's
# .env has ``NCBI_API_Key``).
PUBMED_ENABLED = os.environ.get(
    "PUBMED_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
NCBI_API_KEY = (
    os.environ.get("NCBI_API_KEY")
    or os.environ.get("NCBI_API_Key")
    or os.environ.get("ncbi_api_key")
    or ""
).strip().strip('"').strip("'")
PUBMED_CACHE_TTL_HOURS = int(os.environ.get("PUBMED_CACHE_TTL_HOURS", "24"))
PUBMED_MAX_RESULTS     = int(os.environ.get("PUBMED_MAX_RESULTS", "5"))
PUBMED_MAX_AGE_YEARS   = int(os.environ.get("PUBMED_MAX_AGE_YEARS", "5"))

# ---------- Phase 5.7 / Extra B — FDA FAERS ----------
# Real-world adverse-event signals for off-label / novel combinations.
# Public tier is 240 req/min, no key required; an OpenFDA API key bumps
# the limit to 1000 req/min. Mixed-case .env keys are tolerated (the
# user's .env has ``OpenFDA_API_Key``).
FAERS_ENABLED = os.environ.get(
    "FAERS_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
OPENFDA_API_KEY = (
    os.environ.get("OPENFDA_API_KEY")
    or os.environ.get("OpenFDA_API_Key")
    or os.environ.get("openfda_api_key")
    or ""
).strip().strip('"').strip("'")
FAERS_CACHE_TTL_HOURS = int(os.environ.get("FAERS_CACHE_TTL_HOURS", "24"))
FAERS_MAX_REACTIONS   = int(os.environ.get("FAERS_MAX_REACTIONS", "10"))

# ---------- Phase 5.8 / Extra C — SMART on FHIR EHR Launch ----------
# OAuth2 / SMART App Launcher integration for Epic / Cerner / public sandbox.
# When unconfigured, the smart_fhir router returns 503 — the pipeline still
# accepts ZIP uploads as before.
#
# offline_access in scopes is critical: standard EHR access tokens expire
# in 15-30 min, but a cold pipeline run is ~36 min. Without a refresh
# token the synthesis-phase write-back hits 401. See TokenManager in
# integrations/fhir_client.py.
SMART_CLIENT_ID     = os.environ.get("SMART_CLIENT_ID", "").strip()
SMART_CLIENT_SECRET = os.environ.get("SMART_CLIENT_SECRET", "").strip()
SMART_REDIRECT_URI  = os.environ.get(
    "SMART_REDIRECT_URI", "http://localhost:8000/smart/callback",
).strip()
SMART_SCOPES = os.environ.get(
    "SMART_SCOPES",
    "launch openid fhirUser offline_access "
    "patient/*.read patient/CarePlan.write patient/DocumentReference.write",
).strip()
# Refresh window: refresh access_token if <SMART_TOKEN_REFRESH_MARGIN_S
# seconds remaining at request time. 120 s leaves headroom for slow LLM
# rounds that might fire just before expiry.
SMART_TOKEN_REFRESH_MARGIN_S = int(os.environ.get("SMART_TOKEN_REFRESH_MARGIN_S", "120"))
# Public SMART App Launcher sandbox — used for dev / smoke tests.
SMART_DEFAULT_FHIR_BASE = os.environ.get(
    "SMART_DEFAULT_FHIR_BASE",
    "https://launch.smarthealthit.org/v/r4/fhir",
).strip()
# Worker backend for offloading heavy CPU segmentation so FastAPI requests
# don't block. "inline" | "process" | "celery".
#   inline   — run in-thread (OK on GPU, risky on CPU)
#   process  — ProcessPoolExecutor (default, no external dep)
#   celery   — Celery+Redis (requires SEG_REDIS_URL)
SEG_WORKER_BACKEND = os.getenv("SEG_WORKER_BACKEND", "process").lower()
# nnUNet model cache dir — weights downloaded here on first successful run.
NNUNET_MODELS_DIR = PKG / "models" / "nnunet"
# NIfTI file extensions accepted by ingest.
_NIFTI_EXT = {".nii", ".gz"}  # .nii.gz handled by suffix .gz
NIFTI_EXTENSIONS = (".nii", ".nii.gz")

# ---------- Orchestrator phases ----------
PHASE_STEP_BUDGET = 6        # max tool-calls per phase
PHASE_NAMES = ["ingest", "mri", "recist", "treatment_opt", "pharma", "synthesis"]

# ---------- Chat / QA ----------
CHAT_HISTORY_TURNS = 10
CHAT_REQUIRE_CITATIONS = True
DISCLAIMER = (
    "Research prototype — outputs must be verified by a licensed "
    "healthcare professional before clinical use."
)

# ---------- Google Workspace integration ----------
GOOGLE_CREDENTIALS_PATH = PKG / "credentials" / "credentials.json"
GOOGLE_TOKEN_PATH        = PKG / "credentials" / "token.json"
GOOGLE_SENDER            = "krishnahutrik.n@gmail.com"
GOOGLE_SCOPES            = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/calendar",
]

# Feature flags — each defaults True when token.json exists, False otherwise.
# Override via environment: GMAIL_ENABLED=false python -m neuro_agent ...
def _google_flag(name: str) -> bool:
    env = os.environ.get(name, "").lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    # Auto-detect: enabled if token exists.
    return (PKG / "credentials" / "token.json").exists()

GMAIL_ENABLED    = _google_flag("GMAIL_ENABLED")
DRIVE_ENABLED    = _google_flag("DRIVE_ENABLED")
CALENDAR_ENABLED = _google_flag("CALENDAR_ENABLED")

# ---------- Audit ----------
HASH_SALT = os.environ.get("NEURO_HASH_SALT", "neuro-agent-local")

# ---------- Output stage file map ----------
# Maps WorkingMemory key → (file_stem, stage_number, stage_label)
# Drives the S{N}_stagename.json naming convention and the stage envelope.
STAGE_FILE_MAP: dict[str, tuple[str, int, str]] = {
    "ingestion":    ("S01_ingestion",    1,  "Multi-source File Ingestion"),
    "vision":       ("S02_vision",       2,  "Gemma4 Vision Analysis"),
    "record":       ("S03_record",       3,  "Structured Patient Record Extraction"),
    "recist":       ("S04_recist",       4,  "RECIST 1.1 / RANO Lesion Measurement"),
    "rano":         ("S04b_rano",        4,  "RANO Response Assessment (brain tumours)"),
    "volumetric":   ("S04c_volumetric",  4,  "3D Volumetric Tumour Segmentation (BraTS)"),
    "radiomics":    ("S04d_radiomics",   4,  "Radiomic Features Extraction"),
    "rag":          ("S05_index",        5,  "ChromaDB Vector Indexing"),
    "urgency":      ("S06_urgency",      6,  "Urgency Triage & Clinical Scoring"),
    "medications":  ("S07_medications",  7,  "Medication Ingestion (Vision + OCR)"),
    "interactions": ("S08_interactions", 8,  "Drug Interaction Safety Check"),
    "correlation":  ("S09_correlation",  9,  "Treatment vs Scan Correlation"),
    "timeline":     ("S10_timeline",     10, "Longitudinal Timeline"),
    "qa_examples":  ("S11_qa_examples",  11, "Clinical Q&A (RAG + Citations)"),
    "summary":      ("S12_summary",      12, "Patient Summary & GP Letter"),
    "export":              ("S13_export",              13, "FHIR R4 Export & MDT Package"),
    # Phase 4 — Treatment Optimization (SMBO v3.0)
    "patient_state":       ("S14_patient_state",       14, "Patient State Vectorization"),
    "prediction":          ("S15_prediction",          15, "Survival Prediction GP+RSF"),
    "optimization":        ("S16_optimization",        16, "Batched SMBO Optimization"),
    "shap":                ("S17_shap",                17, "SHAP Explainability"),
    "trial_matches":       ("S17b_clinical_trials",    17, "ClinicalTrials.gov Matching"),
    "pubmed_evidence":     ("S17c_pubmed",             17, "PubMed Evidence Retrieval"),
    "faers_signals":       ("S17d_faers",              17, "FDA FAERS Adverse-Event Signals"),
    "treatment_proposal":  ("S18_treatment_proposal",  18, "MDT Treatment Proposal"),
}

# Map WorkingMemory key → output subfolder kind (used by _persist_key).
# Any key not listed falls back to "stages" (stage envelope JSON) or the patient root.
STAGE_SUBDIR_MAP: dict[str, str] = {
    # All S01-S18 numbered stages go into stages/
    "ingestion":    "stages", "vision":       "stages", "record":       "stages",
    "recist":       "stages", "rano":         "stages",
    "volumetric":   "stages", "radiomics":    "stages",
    "rag":          "stages", "urgency":      "stages",
    "medications":  "stages", "interactions": "stages", "correlation":  "stages",
    "timeline":     "stages", "qa_examples":  "stages", "summary":      "stages",
    "export":       "stages",
    "patient_state":       "stages", "prediction":   "stages",
    "optimization":        "stages", "shap":         "stages",
    "trial_matches":       "stages",
    "pubmed_evidence":     "stages",
    "faers_signals":       "stages",
    "treatment_proposal":  "stages",
    # Notification bags → notifications/
    "notifications_gmail":  "notifications",
    "notifications_sync":   "notifications",
    "notifications_phase4": "notifications",
    # Non-stage keys stay at patient root.
}
