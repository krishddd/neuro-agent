"""OpenAI/Ollama-compatible tool schemas for every sub-agent call.

The guarded orchestrator exposes only the subset of tools relevant to
the current phase. See orchestrator.PHASES for the mapping.
"""
from __future__ import annotations

from typing import Any


def _tool(name: str, description: str, properties: dict[str, Any],
          required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


# ---------- Phase 1: ingest (deterministic, no tools exposed to LLM) ----------
INGEST_PATIENT_FILES = _tool(
    "ingest_patient_files",
    "Discover, classify and load all files for a patient from the dataset root.",
    {"patient_id": {"type": "string"}},
    ["patient_id"],
)


# ---------- Phase 2: MRI ----------
ANALYZE_SCAN = _tool(
    "analyze_scan",
    "Run vision analysis on one MRI visit. Returns findings, mass effect, "
    "hemorrhage and a discrepancy flag against the radiology report.",
    {
        "visit": {"type": "string", "description": "visit id, e.g. v1 or v2"},
    },
    ["visit"],
)

COMPARE_SCANS = _tool(
    "compare_scans",
    "Compare two MRI visits and summarize interval change. Only valid when "
    "prior scans exist.",
    {
        "baseline_visit": {"type": "string"},
        "current_visit": {"type": "string"},
    },
    ["baseline_visit", "current_visit"],
)

FLAG_DISCREPANCY = _tool(
    "flag_discrepancy",
    "Re-read radiology report and vision findings, flag any disagreement "
    "between them.",
    {"visit": {"type": "string"}},
    ["visit"],
)

EXTRACT_RECORD = _tool(
    "extract_patient_record",
    "Produce a structured PatientRecord from the radiology report and the "
    "latest vision observation.",
    {},
)

EXTRACT_RADIOMICS = _tool(
    "extract_radiomics",
    "Compute 5 PyRadiomics features (GLCM contrast/correlation, shape "
    "sphericity, surface-volume ratio, first-order entropy) on the "
    "volumetric mask. Graceful-degrades to radiomics_unavailable when "
    "pyradiomics or a mask is unavailable.",
    {"visit": {"type": "string", "description": "visit id, e.g. v1"}},
    ["visit"],
)


# ---------- Phase 3: RECIST + urgency ----------
MEASURE_LESIONS = _tool(
    "measure_lesions",
    "Measure target lesions on the current visit. If a baseline visit "
    "exists, also return the baseline measurements.",
    {"visit": {"type": "string"}},
    ["visit"],
)

CLASSIFY_RESPONSE = _tool(
    "classify_response",
    "Classify RECIST 1.1 response (CR/PR/SD/PD) from baseline and current "
    "lesion measurements already in memory.",
    {},
)

SCORE_URGENCY = _tool(
    "score_urgency",
    "Compute a 1..5 urgency score from findings, RECIST response and "
    "red-flag keywords.",
    {},
)

INDEX_RAG = _tool(
    "index_rag",
    "Chunk and embed all patient text into the per-patient Chroma collection.",
    {},
)


# ---------- Phase 4: pharma ----------
EXTRACT_MEDICATIONS = _tool(
    "extract_medications",
    "Extract current and historical medications from prescriptions and "
    "discharge documents.",
    {},
)

CHECK_INTERACTIONS = _tool(
    "check_interactions",
    "Check medication interactions using the local drug KB and LLM "
    "reasoning. Returns severity flags.",
    {},
)

CORRELATE_TREATMENT = _tool(
    "correlate_treatment",
    "Correlate RECIST response timeline with medication start/stop dates.",
    {},
)


# ---------- Phase 5: synthesis ----------
BUILD_TIMELINE = _tool(
    "build_timeline",
    "Merge all visit events, medication events and labs into a chronological "
    "timeline with a short narrative.",
    {},
)

WRITE_SUMMARY = _tool(
    "write_summary",
    "Produce a patient-facing letter and a GP handover note.",
    {},
)

EXPORT_FHIR = _tool(
    "export_fhir",
    "Build and write a FHIR R4 bundle plus an MDT package to disk.",
    {},
)


# ---------- Chat sub-agent tools ----------
RECALL_MEMORY = _tool(
    "recall_memory",
    "Read a structured value from WorkingMemory by key "
    "(e.g. 'recist', 'medications', 'interactions', 'urgency').",
    {"key": {"type": "string"}},
    ["key"],
)

SEARCH_RECORDS = _tool(
    "search_records",
    "Semantic + keyword search over the patient's document chunks in Chroma.",
    {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 6},
    },
    ["query"],
)

GET_RECIST_TREND = _tool(
    "get_recist_trend",
    "Return the RECIST response and percent-change between baseline and "
    "current visit.",
    {},
)

GET_CURRENT_MEDS = _tool(
    "get_current_meds",
    "Return the current medication list.",
    {},
)

GET_INTERACTIONS = _tool(
    "get_interactions",
    "Return the interaction report with severity flags.",
    {},
)


# ---------- Phase 4 — Treatment Optimization (SMBO v3.0) ----------
EXTRACT_PATIENT_STATE = _tool(
    "extract_patient_state",
    "Build 20-dim PatientStateVector from intake form, wearable data, and prior "
    "pipeline stages. Also pre-seeds the RAG drug-interaction penalty cache.",
    {},
)

PREDICT_RECIST_PFS = _tool(
    "predict_recist_pfs",
    "Predict RECIST delta (%) with GP uncertainty σ and PFS median with 95% CI "
    "using RSF/Weibull models. Sets the optimization_triggered flag.",
    {},
)

RUN_SMBO = _tool(
    "run_smbo_optimization",
    "Run 60-iteration Batched SMBO (RFR-GP dual-sort, warm-start NCCN priors, "
    "active-inference acquisition) to find the optimal treatment regimen. "
    "Skips if optimization_triggered is False.",
    {},
)

EXPLAIN_SHAP = _tool(
    "explain_with_shap",
    "Generate SHAP KernelExplainer waterfall explainability for the best SMBO "
    "candidate. Outputs top-5 PFS drivers and a waterfall PNG. "
    "Skips if optimization_triggered is False.",
    {},
)

MATCH_CLINICAL_TRIALS = _tool(
    "match_clinical_trials",
    "Query ClinicalTrials.gov v2 for recruiting studies matching this patient's "
    "diagnosis, score candidates (condition match, biomarker eligibility, novel "
    "intervention, phase bonus), and return the top-3. Triggered when predicted "
    "PFS is below threshold or response is PD.",
    {},
)

RETRIEVE_PUBMED_EVIDENCE = _tool(
    "retrieve_pubmed_evidence",
    "Pull top-5 PubMed abstracts (Reviews, RCTs, Meta-analyses; last 5 years) "
    "for this patient's cancer type + top-3 SMBO drugs. Output flows into the "
    "MDT Neuro-Oncologist persona prompt so recommendations cite live PMIDs.",
    {},
)

CHECK_ADVERSE_EVENTS = _tool(
    "check_adverse_events",
    "Query openFDA FAERS for real-world adverse-event signals on SMBO "
    "candidates flagged off_label or novel_combo. Standard-of-care regimens "
    "are skipped to save API budget. Output augments the Pharmacist persona "
    "in the MDT debate.",
    {},
)

REVIEW_MDT = _tool(
    "review_proposal_mdt",
    "Qwen3:14b MDT board reviewer: APPROVE / MODIFY / REJECT / SKIP the top "
    "SMBO treatment proposal after checking contraindications, renal/hepatic "
    "safety, drug interactions, and NCCN guideline alignment.",
    {},
)

# ---------- Phase → tool list mapping ----------
PHASE_TOOLS: dict[str, list[dict[str, Any]]] = {
    "ingest": [],  # deterministic — the orchestrator just calls ingest.run
    "mri": [ANALYZE_SCAN, COMPARE_SCANS, FLAG_DISCREPANCY,
            EXTRACT_RADIOMICS, EXTRACT_RECORD],
    "recist": [MEASURE_LESIONS, CLASSIFY_RESPONSE, SCORE_URGENCY, INDEX_RAG],
    "treatment_opt": [
        EXTRACT_PATIENT_STATE, PREDICT_RECIST_PFS,
        RUN_SMBO, EXPLAIN_SHAP,
        RETRIEVE_PUBMED_EVIDENCE,    # Phase 5.5 / Module 3 — sub-step 4d.4
        MATCH_CLINICAL_TRIALS,
        CHECK_ADVERSE_EVENTS,        # Phase 5.7 / Extra B — sub-step 4d.6
        REVIEW_MDT,
    ],
    "pharma": [EXTRACT_MEDICATIONS, CHECK_INTERACTIONS, CORRELATE_TREATMENT],
    "synthesis": [BUILD_TIMELINE, WRITE_SUMMARY, EXPORT_FHIR],
}

CHAT_TOOLS: list[dict[str, Any]] = [
    RECALL_MEMORY,
    SEARCH_RECORDS,
    GET_RECIST_TREND,
    GET_CURRENT_MEDS,
    GET_INTERACTIONS,
]
