"""Phase 5.7 / Extra B — openFDA FAERS adverse-event client.

Mirrors the ``clinicaltrials_client.py`` / ``pubmed_client.py`` patterns:
24h disk cache, projection, polite rate-limiting, graceful degrade.

Endpoint: ``https://api.fda.gov/drug/event.json`` (no auth required, but
an API key bumps the limit from 240 to 1000 req/min — the user's .env
has ``OpenFDA_API_Key`` so we honour mixed case).

Strategy
--------
For one drug or a (drug_a, drug_b) pair we do a single ``count`` query:

    search=patient.drug.medicinalproduct:"<drug>" [AND ...]
    count=patient.reaction.reactionmeddrapt.exact

This returns a frequency table of MedDRA reaction terms. We pull the
top ``FAERS_MAX_REACTIONS`` and a follow-up call with ``count=...
seriousnessdeath``-style limiters is skipped to keep this within
budget — instead we infer ``serious_pct`` from a single second query
on the same search with ``count=patient.reaction.reactionoutcome``.

Output projection per signal:
    {reaction, n_reports, serious_pct, outcomes:[]}

Always succeeds — network or parse error returns ``[]`` with the
caller logging a warning.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from ..config import (
    FAERS_CACHE_TTL_HOURS,
    FAERS_MAX_REACTIONS,
    OPENFDA_API_KEY,
    OUTPUTS_DIR,
)

log = logging.getLogger(__name__)

FAERS_BASE_URL = "https://api.fda.gov/drug/event.json"
FAERS_TIMEOUT_S = 10.0
_CACHE_DIR = OUTPUTS_DIR / "_cache" / "faers"

# openFDA outcome codes (1=recovered/resolved, 2=recovering, 3=not recovered,
# 4=recovered with sequelae, 5=fatal, 6=unknown). We label only the ones
# clinicians care about for safety review.
_OUTCOME_LABELS = {
    "1": "resolved",
    "2": "ongoing",
    "3": "not_recovered",
    "4": "sequelae",
    "5": "fatal",
    "6": "unknown",
}


# ── Disk cache ────────────────────────────────────────────────────────────────

def _cache_key(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_read(key: str) -> dict | None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    ttl = FAERS_CACHE_TTL_HOURS * 60 * 60
    if (time.time() - path.stat().st_mtime) > ttl:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_write(key: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
    except OSError as exc:
        log.warning("faers: cache write failed: %s", exc)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _build_search(drug_a: str, drug_b: str | None) -> str:
    """Build the openFDA ``search`` query string."""
    a = drug_a.replace('"', "").strip()
    if drug_b and drug_b.lower() not in ("none", ""):
        b = drug_b.replace('"', "").strip()
        return (
            f'patient.drug.medicinalproduct:"{a}"+AND+'
            f'patient.drug.medicinalproduct:"{b}"'
        )
    return f'patient.drug.medicinalproduct:"{a}"'


def _query_count(search: str, count_field: str, limit: int) -> tuple[list[dict], dict]:
    """Run one openFDA count query. Returns (results, raw_meta)."""
    try:
        import requests  # type: ignore
    except ImportError:
        log.warning("faers: `requests` not installed — empty result")
        return ([], {})

    params: dict[str, Any] = {
        "search": search,
        "count":  count_field,
        "limit":  str(max(1, min(limit, 100))),
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY

    try:
        resp = requests.get(FAERS_BASE_URL, params=params, timeout=FAERS_TIMEOUT_S)
        # 404 from openFDA = "no records match search" — not an error.
        if resp.status_code == 404:
            return ([], {})
        resp.raise_for_status()
        payload = resp.json() or {}
    except Exception as exc:
        log.warning("faers: query failed (search=%s, count=%s): %s",
                    search, count_field, exc)
        return ([], {})

    return (list(payload.get("results") or []), payload.get("meta") or {})


# ── Public ────────────────────────────────────────────────────────────────────

def adverse_events(
    drug_a: str,
    drug_b: str | None = None,
    *,
    max_reactions: int | None = None,
) -> dict[str, Any]:
    """Return aggregated adverse-event signals for a drug or pair.

    Output shape::
        {
            "n_total_reports": int,    # total reports for this search
            "signals": [
                {"reaction": str, "n_reports": int, "serious_pct": float,
                 "outcomes": list[str]},
                ...
            ],
            "cache_hit": bool,
            "unavailable": bool,
        }

    Always returns a dict — never raises. Empty signals + unavailable=True
    when the API rejected the call or yielded no records.
    """
    max_reactions = max_reactions or FAERS_MAX_REACTIONS
    if not drug_a or not drug_a.strip():
        return {"n_total_reports": 0, "signals": [], "cache_hit": False,
                "unavailable": True}

    search = _build_search(drug_a, drug_b)
    cache_key = _cache_key({"s": search, "n": max_reactions})
    cached = _cache_read(cache_key)
    if cached is not None:
        log.info("faers: cache hit (%s)", cache_key)
        cached["cache_hit"] = True
        return cached

    # 1. reaction frequency
    rxn_rows, _ = _query_count(
        search, "patient.reaction.reactionmeddrapt.exact", max_reactions,
    )

    # Pair-query fallback — if the dual-AND filter returns nothing
    # (often happens when one drug is logged under brand vs generic),
    # fall back to single-drug query on the primary drug. Less
    # specific, but better than emitting no FAERS signal at all.
    fallback_used = False
    if not rxn_rows and drug_b and drug_b.lower() not in ("none", ""):
        single_search = _build_search(drug_a, None)
        rxn_rows, _ = _query_count(
            single_search, "patient.reaction.reactionmeddrapt.exact", max_reactions,
        )
        if rxn_rows:
            search = single_search          # follow-up outcome query uses same scope
            fallback_used = True

    if not rxn_rows:
        result = {"n_total_reports": 0, "signals": [], "cache_hit": False,
                  "unavailable": True, "fallback_used": False}
        _cache_write(cache_key, result)
        return result

    # 2. outcome distribution (single follow-up call)
    out_rows, _ = _query_count(search, "patient.reaction.reactionoutcome", 6)
    outcome_counts: dict[str, int] = {}
    total_outcomes = 0
    for row in out_rows:
        term = str(row.get("term", "")).strip()
        cnt = int(row.get("count", 0) or 0)
        outcome_counts[term] = outcome_counts.get(term, 0) + cnt
        total_outcomes += cnt
    # Map codes → readable labels; pick the top 3 outcomes overall.
    sorted_outcomes = sorted(outcome_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    outcomes_top = [_OUTCOME_LABELS.get(code, code) for code, _ in sorted_outcomes]
    fatal_pct = (
        100.0 * (outcome_counts.get("5", 0) + outcome_counts.get("4", 0))
        / total_outcomes
    ) if total_outcomes else 0.0

    n_total = sum(int(r.get("count", 0) or 0) for r in rxn_rows)
    signals: list[dict[str, Any]] = []
    for row in rxn_rows[:max_reactions]:
        reaction = str(row.get("term", "")).strip().title()
        cnt = int(row.get("count", 0) or 0)
        if not reaction or cnt <= 0:
            continue
        signals.append({
            "reaction":     reaction,
            "n_reports":    cnt,
            "serious_pct":  round(fatal_pct, 1),  # population-level proxy
            "outcomes":     outcomes_top,
        })

    result = {
        "n_total_reports": n_total,
        "signals":         signals,
        "cache_hit":       False,
        "unavailable":     False,
        "fallback_used":   fallback_used,
    }
    _cache_write(cache_key, result)
    log.info("faers: %s%s → %d reactions, %d total reports",
             drug_a,
             f" + {drug_b}" if drug_b and drug_b.lower() not in ("none", "") else "",
             len(signals), n_total)
    return result


__all__ = ["adverse_events"]
