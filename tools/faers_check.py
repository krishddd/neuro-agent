"""Phase 5.7 / Extra B — FDA FAERS adverse-event check (sub-step 4d.6).

Runs AFTER PubMed evidence retrieval and BEFORE the MDT debate. Iterates
the top-3 SMBO candidates and queries openFDA only for the ones whose
``off_label`` or ``novel_combo`` flag is True. Standard-of-care regimens
skip the call entirely so we don't burn API budget on well-characterised
drugs.

Output: ``S17d_faers.json`` envelope with ``FAERSEvidence``. The
projection flows into ``_build_mdt_context`` so the Pharmacist persona
can cite report counts when reviewing experimental combos.

Graceful degrade — emits an empty stub envelope (no error) when:
    * ``FAERS_ENABLED=false``
    * No SMBO optimization in working memory
    * No candidates have off_label or novel_combo set
    * ``requests`` not installed / network throttled
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import FAERS_ENABLED
from ..memory import WorkingMemory
from ..utils.audit import stage_timer
from ..utils.faers_client import adverse_events
from ..utils.schemas import (
    FAERSEvidence,
    FAERSReport,
    FAERSSignal,
    OptimizationResult,
)
from . import register

log = logging.getLogger(__name__)


def _gate_reasons(cand) -> list[str]:
    """Return ['off_label', 'novel_combo'] subset for a candidate."""
    reasons: list[str] = []
    if getattr(cand, "off_label", False):
        reasons.append("off_label")
    if getattr(cand, "novel_combo", False):
        reasons.append("novel_combo")
    return reasons


@register("check_adverse_events")
def check_adverse_events(memory: WorkingMemory, **_: Any) -> dict[str, Any]:
    """Pull FAERS adverse-event signals for off-label / novel SMBO candidates."""
    pid = memory.patient_id
    with stage_timer("treatment_opt.faers", pid=pid,
                     tool="check_adverse_events") as _t:

        if not FAERS_ENABLED:
            stub = FAERSEvidence(faers_unavailable=True, note="FAERS_ENABLED=false")
            memory.set(WorkingMemory.FAERS_SIGNALS, stub)
            _t.meta["ok"] = True
            _t.meta["disabled"] = True
            return {"ok": True, "disabled": True, "n_queried": 0}

        opt_raw = memory.get(WorkingMemory.OPTIMIZATION)
        if not opt_raw:
            stub = FAERSEvidence(faers_unavailable=True, note="no SMBO optimization")
            memory.set(WorkingMemory.FAERS_SIGNALS, stub)
            _t.meta["ok"] = True
            _t.meta["n_queried"] = 0
            return {"ok": True, "n_queried": 0, "no_optimization": True}

        opt = opt_raw if isinstance(opt_raw, OptimizationResult) \
            else OptimizationResult.model_validate(opt_raw)
        candidates = opt.top_3_candidates or []

        reports: list[FAERSReport] = []
        n_queried = 0
        for cand in candidates:
            triggers = _gate_reasons(cand)
            if not triggers:
                continue   # SOC regimen — skip
            n_queried += 1

            try:
                ev = adverse_events(
                    cand.primary_drug,
                    cand.combo_drug if cand.combo_drug.lower() not in ("none", "") else None,
                )
            except Exception as exc:
                log.warning("faers: candidate #%d query raised: %s", cand.rank, exc)
                ev = {"n_total_reports": 0, "signals": [], "cache_hit": False,
                      "unavailable": True}

            signals = [
                FAERSSignal(
                    reaction=s.get("reaction", ""),
                    n_reports=int(s.get("n_reports", 0) or 0),
                    serious_pct=float(s.get("serious_pct", 0.0) or 0.0),
                    outcomes=list(s.get("outcomes") or []),
                )
                for s in (ev.get("signals") or [])
                if s.get("reaction")
            ]
            fallback_used = bool(ev.get("fallback_used", False))
            note = None
            if not signals and not ev.get("unavailable"):
                note = "no FAERS records"
            elif fallback_used:
                # Make the limitation visible in S17d_faers.json so a human
                # auditor can see at a glance that the listed reactions are
                # for the primary drug alone, not the combo.
                note = (
                    f"pair query empty — counts shown are for "
                    f"{cand.primary_drug} alone (single-drug fallback)"
                )

            reports.append(FAERSReport(
                rank=cand.rank,
                primary_drug=cand.primary_drug,
                combo_drug=cand.combo_drug or "none",
                triggered_by=triggers,
                n_total_reports=int(ev.get("n_total_reports", 0) or 0),
                signals=signals,
                cache_hit=bool(ev.get("cache_hit", False)),
                faers_unavailable=bool(ev.get("unavailable", False)),
                fallback_used=fallback_used,
                note=note,
            ))

        evidence = FAERSEvidence(
            n_candidates_queried=n_queried,
            reports=reports,
            faers_unavailable=(n_queried == 0),
            note=("no off-label or novel-combo candidates — FAERS skipped"
                  if n_queried == 0 else None),
        )
        memory.set(WorkingMemory.FAERS_SIGNALS, evidence)

        log.info(
            "faers: %s — queried %d/%d candidates  total_signals=%d",
            pid, n_queried, len(candidates),
            sum(len(r.signals) for r in reports),
        )
        _t.meta["ok"] = True
        _t.meta["n_queried"] = n_queried
        return {
            "ok": True,
            "n_queried": n_queried,
            "n_candidates": len(candidates),
            "n_total_signals": sum(len(r.signals) for r in reports),
        }
