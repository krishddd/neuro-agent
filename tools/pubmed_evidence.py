"""Phase 5.5 / Module 3 — PubMed Evidence Retrieval (sub-step 4d.4).

Runs AFTER SHAP (4d) and BEFORE clinical-trial matching (4d.5). Pulls live
PubMed abstracts to ground the MDT Neuro-Oncologist persona's
recommendations in current literature instead of LLM priors alone.

Query construction:
    terms = [cancer_type] + top-3 SMBO drugs (primary + combo) + biomarker hints
    filter = Review/RCT/Meta-Analysis, English, last 5 years

Output: ``S17c_pubmed.json`` envelope with ``PubMedEvidence``. Top-5
records (≤600 char abstracts) flow into ``_build_mdt_context`` so the
Neuro-Oncologist persona can cite PMIDs.

Graceful degrade — any of these flips ``pubmed_unavailable=True`` and
returns an empty result without raising:
    * ``PUBMED_ENABLED=false``
    * ``requests`` not installed
    * Network error / NCBI rate-limit
    * No PMIDs match the query
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import PUBMED_ENABLED, PUBMED_MAX_AGE_YEARS, PUBMED_MAX_RESULTS
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.pubmed_client import search_evidence
from ..utils.schemas import (
    OptimizationResult,
    PatientStateVector,
    PubMedEvidence,
    PubMedResult,
)
from . import register

log = logging.getLogger(__name__)


def _query_terms(memory: WorkingMemory) -> list[str]:
    """Build a small, specific search-term list from working memory.

    Order matters — earlier terms anchor relevance ranking. We cap at
    ~5 terms total so PubMed's relevance score isn't diluted.
    """
    terms: list[str] = []

    # 1. Diagnosis / cancer type (most specific available)
    cancer_type: str | None = None
    try:
        ps_raw = memory.get(WorkingMemory.PATIENT_STATE)
        if ps_raw:
            ps = ps_raw if isinstance(ps_raw, PatientStateVector) \
                else PatientStateVector.model_validate(ps_raw)
            cancer_type = ps.cancer_type
    except Exception:
        pass
    if cancer_type and cancer_type.lower() != "unknown":
        terms.append(cancer_type)
    else:
        terms.append("glioblastoma")

    # 2. Top-3 SMBO drugs (primary + combo, deduplicated)
    drugs: list[str] = []
    try:
        opt_raw = memory.get(WorkingMemory.OPTIMIZATION)
        if opt_raw:
            opt = opt_raw if isinstance(opt_raw, OptimizationResult) \
                else OptimizationResult.model_validate(opt_raw)
            for c in (opt.top_3_candidates or [])[:3]:
                for d in (c.primary_drug, c.combo_drug):
                    if d and d.lower() not in ("none", "") and d not in drugs:
                        drugs.append(d)
    except Exception:
        pass
    # Limit to top 2 unique drugs to keep query specific.
    terms.extend(drugs[:2])

    # 3. Biomarker context (one short hint maxx) — keeps MGMT/IDH-aware reviews up
    try:
        ps_raw = memory.get(WorkingMemory.PATIENT_STATE)
        if ps_raw:
            ps = ps_raw if isinstance(ps_raw, PatientStateVector) \
                else PatientStateVector.model_validate(ps_raw)
            if (ps.mgmt_methylation or "").lower() == "methylated":
                terms.append("MGMT methylated")
            elif (ps.idh_mutation or "").lower() == "mutant":
                terms.append("IDH mutant")
    except Exception:
        pass

    # Dedup, preserve order, cap to 5.
    seen, out = set(), []
    for t in terms:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(t)
        if len(out) >= 5:
            break
    return out


@register("retrieve_pubmed_evidence")
def retrieve_pubmed_evidence(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    """Pull top-5 PubMed abstracts for the MDT Neuro-Oncologist persona."""
    pid = memory.patient_id
    with stage_timer("treatment_opt.pubmed_evidence", pid=pid,
                     tool="retrieve_pubmed_evidence") as _t:
        # Disabled? emit a stub envelope so downstream can find S17c.
        if not PUBMED_ENABLED:
            stub = PubMedEvidence(
                query_terms=[], results=[], n_results=0,
                pubmed_unavailable=True, note="PUBMED_ENABLED=false",
            )
            memory.set(WorkingMemory.PUBMED_EVIDENCE, stub)
            _t.meta["ok"] = True
            _t.meta["n_results"] = 0
            _t.meta["disabled"] = True
            return {"ok": True, "n_results": 0, "disabled": True}

        terms = _query_terms(memory)
        if not terms:
            ev = PubMedEvidence(
                query_terms=[], results=[], n_results=0,
                pubmed_unavailable=True,
                note="no query terms available (missing patient_state / optimization)",
            )
            memory.set(WorkingMemory.PUBMED_EVIDENCE, ev)
            _t.meta["ok"] = True
            _t.meta["n_results"] = 0
            return {"ok": True, "n_results": 0, "no_terms": True}

        try:
            raw = search_evidence(
                terms,
                max_results=PUBMED_MAX_RESULTS,
                max_age_years=PUBMED_MAX_AGE_YEARS,
            )
        except Exception as exc:
            log.warning("pubmed_evidence: search_evidence raised: %s", exc)
            raw = []

        results = [PubMedResult.model_validate(r) for r in raw]
        ev = PubMedEvidence(
            query_terms=terms,
            results=results,
            n_results=len(results),
            pubmed_unavailable=(len(results) == 0),
            note=None if results else "no PubMed records found",
        )
        memory.set(WorkingMemory.PUBMED_EVIDENCE, ev)

        log.info(
            "pubmed_evidence: %s — %d records for terms=%s",
            pid, len(results), terms,
        )
        _t.meta["ok"] = True
        _t.meta["n_results"] = len(results)
        return {
            "ok": True,
            "n_results": len(results),
            "query_terms": terms,
            "pmids": [r.pmid for r in results],
        }
