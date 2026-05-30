"""Phase 5.5 / Module 3 — NCBI PubMed E-utilities client.

Mirrors the ``clinicaltrials_client.py`` pattern: 24h disk cache, field
projection, abstract truncation, graceful degrade on network/parse error.

Two-step flow per query:
    1. esearch.fcgi → list of PMIDs matching query
    2. efetch.fcgi  → MEDLINE XML for those PMIDs

Field projection keeps only ``{pmid, title, abstract, authors[:3], pubdate,
journal, publication_types}``. Abstracts are hard-truncated to ≤600 chars
so the MDT Neuro-Oncologist persona prompt stays under the qwen3:14b
context budget (≤3 KB total for 5 abstracts).

Filters applied to esearch:
    * last ``PUBMED_MAX_AGE_YEARS`` years (default 5)
    * English language
    * Review[pt] OR Randomized Controlled Trial[pt] OR Meta-Analysis[pt]

Auth: ``NCBI_API_KEY`` from .env grants 10 req/s; without it the public
tier is throttled to 3 req/s. The wrapper adds the key when available
and never errors when it's absent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

from ..config import (
    NCBI_API_KEY,
    OUTPUTS_DIR,
    PUBMED_CACHE_TTL_HOURS,
    PUBMED_MAX_AGE_YEARS,
    PUBMED_MAX_RESULTS,
)

log = logging.getLogger(__name__)

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_TIMEOUT_S = 12.0
_CACHE_DIR = OUTPUTS_DIR / "_cache" / "pubmed"

# Restrict to study types most useful for clinical decision support.
_DEFAULT_FILTER = (
    '(Review[pt] OR "Randomized Controlled Trial"[pt] OR Meta-Analysis[pt] '
    'OR "Clinical Trial"[pt]) AND English[lang]'
)

_ABSTRACT_MAX_CHARS = 600
_AUTHORS_MAX = 3


# ── Disk cache ────────────────────────────────────────────────────────────────

def _cache_key(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_read(key: str) -> dict | None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    ttl = PUBMED_CACHE_TTL_HOURS * 60 * 60
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
        log.warning("pubmed: cache write failed: %s", exc)


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _common_params() -> dict[str, str]:
    p = {"tool": "neuro-agent", "email": "neuro-agent@local"}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def _esearch(query: str, retmax: int) -> list[str]:
    """Return ordered list of PMIDs for a query (newest first)."""
    try:
        import requests  # type: ignore
    except ImportError:
        log.warning("pubmed: `requests` not installed — returning empty PMID list")
        return []

    params: dict[str, Any] = {
        **_common_params(),
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max(1, min(int(retmax), 25))),
        "sort": "relevance",
    }
    try:
        resp = requests.get(
            f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=PUBMED_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
    except Exception as exc:
        log.warning("pubmed: esearch failed (%s)", exc)
        return []

    return list((payload.get("esearchresult") or {}).get("idlist") or [])


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


def _parse_pubdate(article_el: ET.Element) -> str:
    """Best-effort YYYY-MM-DD from <PubDate> or <ArticleDate>."""
    for path in (
        ".//Article/Journal/JournalIssue/PubDate",
        ".//ArticleDate",
        ".//PubMedPubDate[@PubStatus='pubmed']",
    ):
        el = article_el.find(path)
        if el is None:
            continue
        y = (el.findtext("Year") or "").strip()
        m = (el.findtext("Month") or "").strip()
        d = (el.findtext("Day") or "").strip()
        if y:
            # Month may be "Jan", normalise to numeric.
            try:
                if m and not m.isdigit():
                    m_num = datetime.strptime(m[:3], "%b").month
                    m = f"{m_num:02d}"
            except ValueError:
                m = m or "01"
            d = d.zfill(2) if d.isdigit() else "01"
            m = m.zfill(2) if m else "01"
            return f"{y}-{m or '01'}-{d or '01'}"
    return ""


def _flatten_article(article_el: ET.Element) -> dict[str, Any]:
    """Project a PubmedArticle XML element into our compact dict."""
    pmid = (article_el.findtext(".//PMID") or "").strip()
    title = (article_el.findtext(".//ArticleTitle") or "").strip()

    # Abstract may be split into multiple <AbstractText Label="...">.
    abstract_parts: list[str] = []
    for at in article_el.findall(".//Abstract/AbstractText"):
        label = (at.attrib.get("Label") or "").strip()
        text = (at.text or "").strip()
        if label and text:
            abstract_parts.append(f"{label}: {text}")
        elif text:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    journal = (article_el.findtext(".//Journal/Title") or "").strip()
    pubdate = _parse_pubdate(article_el)

    authors: list[str] = []
    for au in article_el.findall(".//AuthorList/Author")[:_AUTHORS_MAX]:
        last = (au.findtext("LastName") or "").strip()
        init = (au.findtext("Initials") or "").strip()
        if last:
            authors.append(f"{last} {init}".strip())

    pub_types: list[str] = [
        (pt.text or "").strip()
        for pt in article_el.findall(".//PublicationTypeList/PublicationType")
        if pt.text
    ]

    return {
        "pmid":             pmid,
        "title":            _truncate(title, 240),
        "abstract":         _truncate(abstract, _ABSTRACT_MAX_CHARS),
        "authors":          authors,
        "pubdate":          pubdate,
        "journal":          _truncate(journal, 120),
        "publication_types": pub_types[:5],
    }


def _efetch(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch abstract XML for the given PMIDs and return flattened dicts."""
    if not pmids:
        return []
    try:
        import requests  # type: ignore
    except ImportError:
        return []

    params: dict[str, Any] = {
        **_common_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        resp = requests.get(
            f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=PUBMED_TIMEOUT_S,
        )
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as exc:
        log.warning("pubmed: efetch failed (%s)", exc)
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("pubmed: xml parse failed (%s)", exc)
        return []

    out: list[dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        try:
            out.append(_flatten_article(art))
        except Exception as exc:
            log.debug("pubmed: skip malformed article: %s", exc)
    return out


# ── Public ────────────────────────────────────────────────────────────────────

def search_evidence(
    query_terms: list[str],
    *,
    max_results: int | None = None,
    max_age_years: int | None = None,
) -> list[dict[str, Any]]:
    """High-level: ``query_terms`` → ranked list of compact PubMed records.

    Builds a query of the form
    ``(term1 AND term2 AND ...) AND <DEFAULT_FILTER> AND last-N-years``,
    runs esearch + efetch, applies cache, returns ≤``max_results`` rows.

    Always succeeds — network/parse failures yield an empty list.
    """
    max_results = max_results or PUBMED_MAX_RESULTS
    max_age_years = max_age_years or PUBMED_MAX_AGE_YEARS

    cleaned = [t.strip() for t in query_terms if t and t.strip()]
    if not cleaned:
        return []

    # Date filter: relative date range supported by NCBI as ``"<n> years"[dp]``.
    date_filter = f'"last {int(max_age_years)} years"[dp]'
    quoted = [f'"{t}"' if " " in t else t for t in cleaned]
    primary_query = (
        f"({' AND '.join(quoted)}) AND {_DEFAULT_FILTER} AND {date_filter}"
    )

    cache_key = _cache_key({"q": primary_query, "n": max_results})
    cached = _cache_read(cache_key)
    if cached is not None:
        log.info("pubmed: cache hit (%s)", cache_key)
        return list(cached.get("results") or [])[:max_results]

    # P001-RUN-FIX (#5): retry strategy on empty result.
    # Real-world example: ``glioblastoma + bevacizumab + entrectinib`` ANDed
    # with the publication-type filter returned 0 PMIDs because entrectinib
    # has no GBM RCTs/reviews. We progressively widen the query rather
    # than telling the MDT "no evidence" prematurely:
    #   1. full filter + date              (most specific)
    #   2. drop publication-type filter
    #   3. drop date filter too
    #   4. drop the LAST query term (often the rarest drug)
    fallback_queries: list[tuple[str, str]] = [
        (primary_query, "primary"),
        (f"({' AND '.join(quoted)}) AND {date_filter}", "no_pubtype_filter"),
        (f"({' AND '.join(quoted)}) AND English[lang]", "no_date_filter"),
    ]
    if len(quoted) >= 2:
        narrower = quoted[:-1]
        fallback_queries.append((
            f"({' AND '.join(narrower)}) AND {_DEFAULT_FILTER} AND {date_filter}",
            "drop_last_term",
        ))

    pmids: list[str] = []
    used_query: str = primary_query
    used_strategy: str = "primary"
    for q, strategy in fallback_queries:
        pmids = _esearch(q, retmax=max_results * 2)
        if pmids:
            used_query = q
            used_strategy = strategy
            if strategy != "primary":
                log.info("pubmed: primary query empty — recovered via '%s' strategy", strategy)
            break

    if not pmids:
        _cache_write(cache_key, {"results": [], "ts": time.time()})
        return []
    records = _efetch(pmids[: max_results * 2])
    # Drop records without a PMID or title.
    records = [r for r in records if r.get("pmid") and r.get("title")][:max_results]
    # Tag each record with the retrieval strategy so the MDT prompt can
    # surface "evidence retrieved with widened query" when relevant.
    for r in records:
        r.setdefault("retrieval_strategy", used_strategy)

    _cache_write(cache_key, {
        "results": records, "ts": time.time(), "query": used_query,
        "strategy": used_strategy,
    })
    log.info("pubmed: fetched %d records for terms=%s (strategy=%s)",
             len(records), cleaned, used_strategy)
    return records


__all__ = ["search_evidence"]
