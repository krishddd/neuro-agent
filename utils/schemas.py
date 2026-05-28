"""Pydantic models for every stage output.

These double as: (a) validation of LLM JSON responses, (b) the canonical
shape stored in WorkingMemory, (c) the contract for the FastAPI layer.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------- Ingestion ----------
FileKind = Literal[
    "mri_image", "mri_report", "prescription", "discharge",
    "pathology", "lab", "protocol", "timeline", "correlation",
    "wearable", "intake_form", "other",
]


class IngestedFile(BaseModel):
    path: str
    kind: FileKind
    visit: str = "v1"                   # "v1", "v2", ... derived from folder
    mime: str
    size_bytes: int
    text: Optional[str] = None          # extracted text (PDF) — not logged
    image_path: Optional[str] = None    # normalized PNG for vision models
    meta: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    patient_id: str
    files: list[IngestedFile]
    visits: list[str]
    has_prior_scans: bool


# ---------- Vision / MRI ----------
class MRIFinding(BaseModel):
    description: str
    location: Optional[str] = None
    size_mm: Optional[float] = None
    enhancement: Optional[str] = None


class VisionObservation(BaseModel):
    visit: str
    findings: list[MRIFinding]
    impression: str
    mass_effect: bool = False
    hemorrhage: bool = False
    discrepancy_with_report: bool = False
    discrepancy_notes: Optional[str] = None


# ---------- Patient record ----------
class PatientRecord(BaseModel):
    patient_id: str
    # MULTI-PATIENT-FIX: ``patient_name`` was missing from the schema, so
    # the patient letter / GP letter / report.md fell through to "Dear
    # Patient" / pid for every patient even when the intake JSON had
    # ``patient_name`` populated. Now extracted directly from intake by
    # ``mri_agent.extract_patient_record`` and surfaced here.
    patient_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    diagnosis: Optional[str] = None
    diagnosis_date: Optional[date] = None
    findings: list[MRIFinding] = Field(default_factory=list)
    impression: Optional[str] = None


# ---------- RECIST ----------
class RECISTLesion(BaseModel):
    lesion_id: str
    location: str
    longest_diameter_mm: float
    visit: str
    target: bool = True


RECISTResponseCode = Literal["CR", "PR", "SD", "PD", "NE"]


class RECISTAssessment(BaseModel):
    baseline_sum_mm: Optional[float]
    current_sum_mm: Optional[float]
    pct_change: Optional[float]
    response: RECISTResponseCode
    lesions_baseline: list[RECISTLesion] = Field(default_factory=list)
    lesions_current: list[RECISTLesion] = Field(default_factory=list)
    rationale: str
    new_lesion_detected: bool = False      # RECIST 1.1: any new lesion = automatic PD
    confirmation_required: bool = False    # CR/PR requires confirmation scan ≥4 weeks


# ---------- RANO (Neuro-Oncology) ----------
class RANOAssessment(BaseModel):
    """RANO 2010 response assessment for brain tumours.

    Unlike RECIST 1.1, RANO uses the *bidirectional product* (longest diameter
    × its perpendicular) of enhancing lesions on post-contrast T1, and also
    weighs T2/FLAIR change, corticosteroid dose change, and neurologic status.
    """
    bidirectional_product_mm2: float = 0.0
    baseline_product_mm2: Optional[float] = None
    delta_product_pct: Optional[float] = None
    t2_flair_change: Literal["decreased", "stable", "increased", "new", "unknown"] = "unknown"
    corticosteroid_dose_mg_per_day: Optional[float] = None
    corticosteroid_dose_change: Literal[
        "decreased", "stable", "increased", "new", "none", "unknown"
    ] = "unknown"
    neurologic_status: Literal["improved", "stable", "worsened", "unknown"] = "unknown"
    new_enhancing_lesion: bool = False
    nonmeasurable_disease_progression: bool = False
    response: Literal["CR", "PR", "SD", "PD", "NE"] = "NE"
    criteria_used: Literal["RANO", "RECIST_1.1"] = "RANO"
    rationale: str = ""


# ---------- Urgency ----------
class UrgencyAssessment(BaseModel):
    score: int = Field(ge=1, le=5)
    level: Literal["routine", "soon", "priority", "urgent", "critical"]
    drivers: list[str]
    rationale: str


# ---------- Medications ----------
class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[date] = None
    stop_date: Optional[date] = None
    indication: Optional[str] = None


class MedicationList(BaseModel):
    current: list[Medication] = Field(default_factory=list)
    historical: list[Medication] = Field(default_factory=list)


# ---------- Interactions ----------
InteractionSeverity = Literal[
    "none", "minor", "moderate", "major", "contraindicated",
]


class Interaction(BaseModel):
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    mechanism: Optional[str] = None
    recommendation: Optional[str] = None
    source: Optional[str] = None  # KB entry or "llm_inferred"


class InteractionReport(BaseModel):
    interactions: list[Interaction]
    highest_severity: InteractionSeverity
    flags: list[str] = Field(default_factory=list)


# ---------- Correlation / Timeline ----------
class MedEvent(BaseModel):
    drug: str
    event_type: Literal["start", "stop", "dose_change"]
    event_date: Optional[date] = None
    dose: Optional[str] = None


class CorrelationResult(BaseModel):
    summary: str
    treatment_started: Optional[date] = None
    response_observed: Optional[date] = None
    lag_days: Optional[int] = None
    consistent: bool = False
    notes: Optional[str] = None
    med_events: list[MedEvent] = Field(default_factory=list)  # full event timeline


class TimelineEvent(BaseModel):
    when: date | datetime
    kind: Literal["scan", "med_start", "med_stop", "lab", "visit", "note"]
    label: str
    source_file: Optional[str] = None


class Timeline(BaseModel):
    events: list[TimelineEvent]
    narrative: str


# ---------- Summary / export ----------
class PatientSummary(BaseModel):
    patient_letter:     str
    gp_handover_letter: str

    @model_validator(mode="before")
    @classmethod
    def _alias_gp_handover(cls, data: dict) -> dict:
        """Accept 'gp_handover' as an alias for 'gp_handover_letter'.

        Some model generations return the shorter key name; this normalises
        both spellings so the pipeline never fails on a key mismatch.
        """
        if isinstance(data, dict):
            if "gp_handover" in data and "gp_handover_letter" not in data:
                data = dict(data)
                data["gp_handover_letter"] = data.pop("gp_handover")
        return data


class ExportResult(BaseModel):
    fhir_bundle_path: str
    mdt_package_path: Optional[str] = None
    patient_letter_path: Optional[str] = None
    gp_handover_path: Optional[str] = None


# ---------- Q&A ----------
class QACitation(BaseModel):
    file: str
    visit: Optional[str] = None
    chunk: Optional[int] = None
    score: Optional[float] = None


class QAAnswer(BaseModel):
    answer: str
    sources: list[QACitation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    session_id: Optional[str] = None
    disclaimer: str = ""


# ── Phase 4 — Treatment Optimization (SMBO v3.0) ──────────────────────────────

class PatientStateVector(BaseModel):
    """25-dimensional normalised patient state vector for SMBO input.

    Dim layout (ordered, must stay stable — models are trained on this):
      0..3   tumour burden     (4)
      4..9   lab markers       (6)
      10..14 wearable vitals   (5)
      15..19 treatment history (5)
      20..24 radiomics         (5)   ← added in Phase 5.1
    """
    patient_id: str
    cancer_type: str = "unknown"

    # Tumour burden (4 dims)
    sum_of_diameters_mm:  float = 0.0
    delta_sod_pct:        float = 0.0
    lesion_count:         int   = 0
    new_lesion_flag:      int   = 0     # 0 or 1

    # Lab markers (6 dims)
    ldh_u_per_l:          float = 0.0
    crp_mg_per_l:         float = 0.0
    nlr:                  float = 0.0
    hemoglobin_g_per_dl:  float = 0.0
    albumin_g_per_dl:     float = 0.0
    egfr_ml_per_min:      float = 0.0

    # Wearable vitals (5 dims)
    daily_steps_7d_avg:   float = 0.0
    resting_hr_bpm:       float = 0.0
    sleep_hours_7d_avg:   float = 0.0
    hrv_ms:               float = 0.0
    ecog_ps_score:        float = 0.0

    # Treatment history (5 dims)
    treatment_cycles_completed: float = 0.0
    days_since_last_dose:       float = 0.0
    total_prior_lines:          float = 0.0
    dose_reduction_flag:        int   = 0   # 0 or 1
    treatment_duration_weeks:   float = 0.0

    # Longitudinal trajectory (4 dims — Phase 5.3 / Module 2)
    # Computed from outputs/<pid>/history/longitudinal.jsonl across prior
    # visits. On first visit (visit_count=0) all four are population-median
    # imputed (flagged in imputation_mask).
    sod_growth_rate_mm_per_week: float = 0.0   # Δ sum-of-diameters / Δt (weeks)
    pfs_trajectory_slope:        float = 0.0   # negative = worsening response
    treatment_response_streak:   int   = 0     # # consecutive SD/PR visits
    visit_count:                 int   = 0     # number of prior visits in chain

    # Radiomics (5 dims — Phase 5.1 / Module 4)
    # Extracted by PyRadiomics from the nnU-Net / MONAI mask of the
    # enhancing tumour. When the mask is unavailable, all five are
    # population-median-imputed and flagged in imputation_mask.
    glcm_contrast:                float = 0.0   # texture heterogeneity
    glcm_correlation:             float = 0.0   # texture directionality
    shape_sphericity:             float = 0.0   # 0=irregular, 1=sphere
    shape_surface_volume_ratio:   float = 0.0
    firstorder_entropy:           float = 0.0   # intensity chaos

    # Pharmacogenomics (Phase 5.2 / Module 5) — optional, never imputed
    # Missing genotypes default to "normal_metabolizer" with pgx_unavailable=True
    # so SMBO toxicity/efficacy multipliers collapse to 1.0.
    pgx_profile: Optional["PatientPharmacogenomics"] = None

    # Molecular biomarkers (Task 3 — never imputed; missing → hard stop)
    mgmt_methylation: Optional[Literal["methylated", "unmethylated", "unknown"]] = None
    idh_mutation:     Optional[Literal["mutant", "wildtype", "unknown"]]         = None
    biomarker_hard_stop:        bool          = False
    biomarker_hard_stop_reason: Optional[str] = None

    # Metadata
    normalized:       list[float] = Field(default_factory=list)
    imputation_mask:  list[int]   = Field(default_factory=list)
    raw_values:       dict[str, Any] = Field(default_factory=dict)

    @field_validator("normalized", mode="before")
    @classmethod
    def _coerce_floats(cls, v: Any) -> list[float]:
        """Convert numpy float32/64 to Python float (JSON serialisable)."""
        if isinstance(v, list):
            return [float(x) for x in v]
        return v


class PatientPharmacogenomics(BaseModel):
    """Patient CYP450 + related pharmacogenomic phenotypes (Phase 5.2 / Module 5).

    All fields default to ``normal_metabolizer`` so untested patients collapse to
    1.0 multipliers in ``apply_pgx()``. ``pgx_unavailable`` flags the absence of
    any real genotype so downstream logs can note population-default fallback.
    """
    cyp2d6:   str = "normal_metabolizer"
    cyp2c19:  str = "normal_metabolizer"
    cyp2c9:   str = "normal_metabolizer"
    cyp3a4:   str = "normal_metabolizer"
    cyp2b6:   str = "normal_metabolizer"
    cyp2c8:   str = "normal_metabolizer"
    tpmt:     str = "normal_metabolizer"
    dpyd:     str = "normal_metabolizer"
    ugt1a1:   str = "normal_metabolizer"
    mthfr:    str = "normal_metabolizer"
    gstp1:    str = "normal_metabolizer"
    cbr3:     str = "normal_metabolizer"
    hla_b:    str = "normal_metabolizer"    # HLA-B*1502 etc. — serialised as string
    # True iff intake did not supply any pharmacogenomic data.
    pgx_unavailable: bool = True
    # Raw ``pharmacogenomics`` block from intake (for audit), if present.
    raw: dict[str, Any] = Field(default_factory=dict)


class FAERSSignal(BaseModel):
    """One adverse-event signal from openFDA FAERS (Phase 5.7)."""
    reaction:        str                   # MedDRA preferred term
    n_reports:       int = 0               # how many adverse-event reports
    serious_pct:     float = 0.0           # % flagged "serious" (0–100)
    outcomes:        list[str] = Field(default_factory=list)  # death, hospitalization, ...


class FAERSReport(BaseModel):
    """Per-candidate FAERS lookup result (Phase 5.7 / Extra B).

    One block per SMBO candidate that triggered the call (off_label or
    novel_combo). ``signals`` is empty when openFDA has no records for
    the drug pair, the call was throttled, or FAERS_ENABLED=false.
    """
    rank:            int
    primary_drug:    str
    combo_drug:      str = "none"
    triggered_by:    list[str] = Field(default_factory=list)  # ["off_label","novel_combo"]
    n_total_reports: int = 0
    signals:         list[FAERSSignal] = Field(default_factory=list)
    cache_hit:       bool = False
    note:            Optional[str] = None
    faers_unavailable: bool = False
    # P001-RUN-FIX: when the pair query (drug_a AND drug_b) returns no
    # records (often a brand-vs-generic logging mismatch in raw FAERS),
    # the client falls back to a single-drug query on drug_a. The signals
    # then describe drug_a alone, not the pair — the Pharmacist persona
    # must NOT cite them as combo-specific. This flag surfaces that.
    fallback_used:   bool = False


class FAERSEvidence(BaseModel):
    """Aggregate FAERS output written to S17d_faers.json (Phase 5.7)."""
    n_candidates_queried: int = 0
    reports:              list[FAERSReport] = Field(default_factory=list)
    faers_unavailable:    bool = False
    note:                 Optional[str] = None


class PubMedResult(BaseModel):
    """One PubMed record (Phase 5.5 / Module 3).

    ``retrieval_strategy`` records which widening tier produced the hit
    (``primary``, ``no_pubtype_filter``, ``no_date_filter``,
    ``drop_last_term``) — surfaced in the JSON envelope so an auditor
    can see when evidence came from a narrowed query.

    Field projection matches ``pubmed_client._flatten_article`` so JSON
    persistence in S17c_pubmed.json round-trips cleanly.
    """
    pmid:              str
    title:             str
    abstract:          str = ""
    authors:           list[str] = Field(default_factory=list)
    pubdate:           str = ""             # YYYY-MM-DD (best-effort)
    journal:           str = ""
    publication_types: list[str] = Field(default_factory=list)
    # P001-RUN-FIX (#5 follow-up): preserve the widening-retry breadcrumb
    # so S17c_pubmed.json shows whether a record came from the strict
    # primary query or a relaxed fallback.
    retrieval_strategy: Optional[str] = None


class PubMedEvidence(BaseModel):
    """Output of sub-step 4d.4: PubMed evidence retrieval (Phase 5.5)."""
    query_terms:    list[str] = Field(default_factory=list)
    results:        list[PubMedResult] = Field(default_factory=list)
    n_results:      int = 0
    cache_hit:      bool = False
    pubmed_unavailable: bool = False
    note:           Optional[str] = None


class LongitudinalVisit(BaseModel):
    """One row in ``outputs/<pid>/history/longitudinal.jsonl`` (Phase 5.3 / Module 2).

    Append-only — never rewritten. Each successful pipeline run finalises
    one visit. The trajectory features on PatientStateVector are computed
    from a chronologically ordered list of these visits.
    """
    visit_id:    str             # ULID-like / run id
    visit_date:  str             # ISO 8601 timestamp
    sum_of_diameters_mm: float = 0.0
    pfs_median_weeks:    Optional[float] = None
    recist_response:     Optional[str]   = None   # CR | PR | SD | PD | None
    cancer_type:         Optional[str]   = None
    # Snapshot of the normalised vector for trajectory regression.
    normalized:          list[float] = Field(default_factory=list)


class LongitudinalHistory(BaseModel):
    """Ordered list of prior visits for one patient (Phase 5.3 / Module 2)."""
    patient_id: str
    visits: list[LongitudinalVisit] = Field(default_factory=list)

    @property
    def visit_count(self) -> int:
        return len(self.visits)


# Resolve forward reference on PatientStateVector.pgx_profile now that
# PatientPharmacogenomics is defined.
PatientStateVector.model_rebuild()


class PredictionResult(BaseModel):
    """Output of sub-step 4b: GP + RSF survival predictions."""
    recist_delta_pred:    float              # GP-predicted % RECIST change
    recist_sigma:         float              # GP uncertainty (std dev)
    pfs_median_weeks:     float
    pfs_ci_low:           float
    pfs_ci_high:          float
    survival_curve:       list[dict[str, float]] = Field(default_factory=list)
    optimization_triggered: bool = False
    trigger_reason:       Optional[str] = None


class SMBOCandidate(BaseModel):
    """One candidate from the SMBO dual-sort pool."""
    rank:               int
    primary_drug:       str
    combo_drug:         str
    dose_fraction:      float
    cycle_weeks:        int
    route:              str
    predicted_pfs_weeks:     float
    recist_delta_pred:       float
    rag_penalty:             float
    ei_score:                float
    pred_score:              float
    pool:                    Literal["EI", "PRED"]
    # Task 6: multi-objective SMBO — toxicity objective
    predicted_toxicity_score: float = 0.0
    top_aes:                 list[str] = Field(default_factory=list)
    # Phase 5.2 / Module 5 — Pharmacogenomic adjustment trail
    pgx_adjusted:            bool       = False
    pgx_notes:               list[str] = Field(default_factory=list)
    # Phase 5.7 / Extra B — flags that gate FDA FAERS adverse-event lookup.
    # ``off_label``: primary drug not in NCCN first/second-line for cancer_type.
    # ``novel_combo``: combo_drug also not in NCCN regimens for cancer_type.
    off_label:               bool      = False
    novel_combo:             bool      = False


class OptimizationResult(BaseModel):
    """Output of sub-step 4c: 60-iteration Batched SMBO results."""
    triggered:              bool
    n_iterations:           int = 0
    top_3_candidates:       list[SMBOCandidate] = Field(default_factory=list)
    convergence_plot_path:  Optional[str] = None
    landscape_plot_path:    Optional[str] = None
    final_best:             Optional[SMBOCandidate] = None
    sigma_at_convergence:   Optional[float] = None
    early_stopped:          bool = False
    warm_start_anchors:     list[str] = Field(default_factory=list)


class ShapDriver(BaseModel):
    feature:    str
    shap_value: float
    direction:  Literal["+", "-"]


class ShapResult(BaseModel):
    """Output of sub-step 4d: SHAP KernelExplainer results."""
    base_value:         float               # population average PFS for cancer type
    top_5_drivers:      list[ShapDriver] = Field(default_factory=list)
    waterfall_plot_path: Optional[str] = None
    all_shap_values:    dict[str, float] = Field(default_factory=dict)
    # Plain-English LLM narrative of the top drivers (qwen3:14b, best-effort).
    narrative:          str = ""


# ---------- MDT Debate (Task 7) ----------
class MDTPersonaTurn(BaseModel):
    """One statement from an MDT persona in the multi-agent debate."""
    persona: Literal["neuroradiologist", "neurosurgeon", "neurooncologist",
                     "pharmacist", "chair"]
    round: int
    statement: str
    concerns: list[str] = Field(default_factory=list)
    agreement_with_proposal: Literal["agree", "modify", "disagree"] = "modify"


class TreatmentProposal(BaseModel):
    """Output of sub-step 4e: Qwen3 MDT board reviewer decision."""
    decision:                   Literal["APPROVE", "MODIFY", "REJECT", "SKIP"]
    reason:                     str
    proposed_regimen:           Optional[str] = None
    modifications:              list[str] = Field(default_factory=list)
    contraindications_checked:  list[str] = Field(default_factory=list)
    guideline_alignment:        Optional[str] = None
    mdt_discussion_required:    bool = False
    rag_interaction_flags:      list[str] = Field(default_factory=list)
    clinical_narrative:         str = ""
    # Second-pass qwen3:14b self-critique output (best-effort; default empty).
    audit_concerns:             list[str] = Field(default_factory=list)
    audit_recommendation:       str = ""
    audit_rationale:            str = ""
    # Task 7: Multi-agent MDT debate transcript + consensus.
    debate_transcript:          list[MDTPersonaTurn] = Field(default_factory=list)
    consensus_score:            Optional[float] = None   # fraction of personas agreeing


# ---------- Clinical Trial Matching (Task 8) ----------
class ClinicalTrialMatch(BaseModel):
    """One ClinicalTrials.gov match after compression + scoring."""
    nct_id:              str
    title:               str
    phase:               str = ""
    status:              str = ""
    interventions:       list[str] = Field(default_factory=list)
    eligibility_summary: str = ""      # ≤500 chars, regex-compressed
    distance_km:         Optional[float] = None
    match_score:         float = 0.0
    match_reasoning:     str = ""


class TrialMatchResult(BaseModel):
    """Output of sub-step 4d.5: ClinicalTrials.gov search + match scoring."""
    triggered:        bool
    trigger_reason:   str = ""
    n_searched:       int = 0
    top_matches:      list[ClinicalTrialMatch] = Field(default_factory=list)
    search_query:     str = ""
    retrieved_at:     Optional[str] = None
