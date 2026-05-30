"""ClinicalTrials.gov v2 API client (Task 8).

Thin wrapper over the public REST endpoint at:
    https://clinicaltrials.gov/api/v2/studies

Two hard requirements of this wrapper — both for context-window protection
downstream in the MDT debate (Task 7) and for polite rate limiting:

1. **Field projection** — the API supports `fields=...` to select specific
   JSONPath nodes. We request ONLY identification, status, phase,
   interventions, and eligibility — NOT locations, sponsors, documents,
   IPD plans, outcomes. This keeps raw response size ~1–2 KB per study.

2. **Eligibility compression** — raw `eligibilityCriteria` text is often
   3–5 KB of bullet points. We regex-extract just the lines that mention
   the patient's diagnosis, biomarkers (MGMT/IDH), prior lines, ECOG, or
   common organ-function thresholds, and cap at 500 chars. No LLM calls.

Disk cache: 24 h TTL under ``outputs/_cache/ctgov/`` — one JSON per query
hash so repeat runs are free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from ..config import OUTPUTS_DIR

log = logging.getLogger(__name__)

CTGOV_BASE_URL   = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_TIMEOUT_S  = 10.0
CTGOV_CACHE_TTL  = 24 * 60 * 60  # 24 h
_CACHE_DIR       = OUTPUTS_DIR / "_cache" / "ctgov"

# Only request these JSONPath fields — drops ~80% of raw payload size.
_FIELDS = ",".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.designModule.phases",
    "protocolSection.armsInterventionsModule.interventions.name",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
    "protocolSection.eligibilityModule.sex",
])

# Keywords that make an eligibility line worth keeping during compression.
_ELIG_KEEP_PATTERNS = re.compile(
    r"(glio|gbm|glioblastoma|astrocyt|oligodendr|ependym|medulloblast|"
    r"mgmt|methylat|idh[- ]?1?[- ]?2?|mutant|wildtype|wild-type|"
    r"ecog|karnofsky|kps|"
    r"prior (line|therap|chemo|radiation|resection)|"
    r"bevacizumab|temozolomide|lomustine|nivolumab|pembrolizumab|"
    r"platelet|anc|neutrophil|hemoglobin|creatinine|bilirubin|alt|ast|"
    r"egfr|renal|hepatic|liver|kidney|"
    r"pregn|lactat|measur|rano|recist)",
    re.IGNORECASE,
)

_AGE_LIMIT_PATTERN = re.compile(
    r"(age|years?).{0,40}?(\d{1,3})\s*(to|-|–)\s*(\d{1,3})",
    re.IGNORECASE,
)


# ── Disk cache ────────────────────────────────────────────────────────────────

def _cache_key(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_read(key: str) -> dict | None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) > CTGOV_CACHE_TTL:
        try:
            path.unlink()
        except Exception:
            pass
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write(key: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning("ctgov: cache write failed: %s", exc)


# ── Eligibility compression ───────────────────────────────────────────────────

def compress_eligibility(raw_criteria: str, max_chars: int = 500) -> str:
    """Regex-compress eligibilityCriteria to a ≤500-char summary string.

    Keeps bullet/section lines that mention diagnosis, biomarkers, prior
    lines, ECOG/KPS, or common organ-function thresholds. Drops boilerplate.
    """
    if not raw_criteria:
        return ""

    # Normalise line breaks — CT.gov uses both actual newlines and \n\n breaks.
    text = raw_criteria.replace("\r", "")
    # Split on newline OR bullet glyphs OR periods followed by capital letter.
    lines = re.split(r"\n+|(?:(?<=[.;])\s+(?=[A-Z0-9-]))", text)
    kept: list[str] = []
    for ln in lines:
        ln = ln.strip(" -*•\t")
        if len(ln) < 10 or len(ln) > 300:
            continue
        if _ELIG_KEEP_PATTERNS.search(ln):
            kept.append(ln)
        if sum(len(s) + 2 for s in kept) > max_chars:
            break

    summary = " | ".join(kept)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


# ── HTTP call + projection ────────────────────────────────────────────────────

def _flatten_study(raw: dict) -> dict:
    """Flatten the nested v2 JSON into the compact fields we care about."""
    ps = (raw or {}).get("protocolSection", {}) or {}
    ident  = ps.get("identificationModule", {}) or {}
    status = ps.get("statusModule", {}) or {}
    design = ps.get("designModule", {}) or {}
    armsm  = ps.get("armsInterventionsModule", {}) or {}
    elig   = ps.get("eligibilityModule", {}) or {}

    interventions = [
        (iv.get("name") or "").strip()
        for iv in (armsm.get("interventions") or [])
        if iv.get("name")
    ]
    phases = design.get("phases") or []
    phase_str = "/".join(p.replace("PHASE", "Phase ") for p in phases) or ""

    raw_elig = elig.get("eligibilityCriteria") or ""
    return {
        "nct_id":              ident.get("nctId") or "",
        "title":               (ident.get("briefTitle") or "").strip(),
        "phase":               phase_str,
        "status":              status.get("overallStatus") or "",
        "interventions":       interventions[:8],
        "eligibility_summary": compress_eligibility(raw_elig),
        "min_age":             elig.get("minimumAge") or "",
        "max_age":             elig.get("maximumAge") or "",
        "sex":                 elig.get("sex") or "",
    }


def search_trials(
    condition: str,
    intervention: str | None = None,
    status: str = "RECRUITING",
    max_results: int = 20,
) -> list[dict]:
    """Query CT.gov v2 /studies endpoint and return compact trial dicts.

    Network errors (timeout / no connectivity) → empty list, logged.
    """
    params: dict[str, Any] = {
        "query.cond":  condition,
        "filter.overallStatus": status,
        "fields":      _FIELDS,
        "pageSize":    max(1, min(int(max_results), 50)),
        "format":      "json",
    }
    if intervention:
        params["query.intr"] = intervention

    cache_key = _cache_key(params)
    cached = _cache_read(cache_key)
    if cached is not None:
        log.info("ctgov: cache hit (%s)", cache_key)
        studies = cached.get("studies") or []
    else:
        try:
            import requests  # type: ignore
        except ImportError:
            log.warning("ctgov: `requests` not installed — returning empty list")
            return []

        try:
            resp = requests.get(CTGOV_BASE_URL, params=params, timeout=CTGOV_TIMEOUT_S)
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as exc:
            log.warning("ctgov: query failed (%s)", exc)
            return []

        studies = payload.get("studies") or []
        _cache_write(cache_key, {"studies": studies, "ts": time.time()})
        log.info("ctgov: fetched %d studies for %r (intervention=%r)",
                 len(studies), condition, intervention)

    return [_flatten_study(s) for s in studies]
