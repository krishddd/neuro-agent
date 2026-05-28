"""End-to-end evaluation harness.

Run a guarded pipeline on each patient in a gold file, then check stage
outputs against the expected values. Also runs the chatbot against the
patient's gold Q&A set and verifies must-contain text + citation files.

Usage:
    python -m neuro_agent.eval.harness --gold neuro_agent/eval/gold.example.json
    python -m neuro_agent.eval.harness --gold gold.json --skip-pipeline   # use cached outputs
    python -m neuro_agent.eval.harness --gold gold.json --xlsx report.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..memory import WorkingMemory
from ..orchestrator import run_patient
from ..tools.chat_agent import answer
from ..utils.schemas import (
    InteractionReport,
    MedicationList,
    PatientRecord,
    RECISTAssessment,
    UrgencyAssessment,
)
from ..utils.tool_helpers import load_model

_SEVERITY_RANK = {
    "none": 0, "minor": 1, "moderate": 2, "major": 3, "contraindicated": 4,
}


# ---------- check primitives ----------
@dataclass
class Check:
    name: str
    ok: bool
    expected: Any = None
    actual: Any = None
    note: str = ""


@dataclass
class PatientResult:
    patient_id: str
    duration_s: float
    pipeline_ok: bool
    checks: list[Check] = field(default_factory=list)
    qa_checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks + self.qa_checks if c.ok)

    @property
    def total(self) -> int:
        return len(self.checks) + len(self.qa_checks)


def _check(name: str, ok: bool, expected: Any = None, actual: Any = None, note: str = "") -> Check:
    return Check(name=name, ok=ok, expected=expected, actual=actual, note=note)


# ---------- pipeline checks ----------
def _eval_record(memory: WorkingMemory, expected: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    rec = load_model(memory, WorkingMemory.RECORD, PatientRecord)
    sub = (expected.get("diagnosis_substring") or "").lower()
    if sub:
        actual = ((rec.diagnosis if rec else "") or "").lower()
        out.append(_check("diagnosis_substring", sub in actual, sub, actual))
    return out


def _eval_recist(memory: WorkingMemory, expected: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    if "recist_response" not in expected:
        return out
    r = load_model(memory, WorkingMemory.RECIST, RECISTAssessment)
    actual = r.response if r else None
    out.append(_check("recist_response", actual == expected["recist_response"], expected["recist_response"], actual))
    return out


def _eval_urgency(memory: WorkingMemory, expected: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    if "urgency_min_score" not in expected:
        return out
    u = load_model(memory, WorkingMemory.URGENCY, UrgencyAssessment)
    actual = u.score if u else None
    out.append(_check(
        "urgency_min_score",
        bool(actual is not None and actual >= expected["urgency_min_score"]),
        f">= {expected['urgency_min_score']}", actual,
    ))
    return out


def _eval_meds(memory: WorkingMemory, expected: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    must = [m.lower() for m in expected.get("must_contain_meds", [])]
    if not must:
        return out
    m = load_model(memory, WorkingMemory.MEDICATIONS, MedicationList)
    names = {x.name.lower() for x in (m.current + m.historical)} if m else set()
    for needle in must:
        out.append(_check(f"med_present:{needle}", any(needle in n for n in names), needle, sorted(names)))
    return out


def _eval_interactions(memory: WorkingMemory, expected: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    if "interactions_min_severity" not in expected:
        return out
    target = expected["interactions_min_severity"].lower()
    target_rank = _SEVERITY_RANK.get(target, 0)
    i = load_model(memory, WorkingMemory.INTERACTIONS, InteractionReport)
    actual = i.highest_severity if i else "none"
    actual_rank = _SEVERITY_RANK.get(actual, 0)
    out.append(_check(
        "interactions_min_severity",
        actual_rank >= target_rank, f">= {target}", actual,
    ))
    return out


def _eval_visits(memory: WorkingMemory, expected: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    must = expected.get("must_have_visits") or []
    if not must:
        return out
    ing = memory.get(WorkingMemory.INGESTION)
    visits = set((ing or {}).get("visits", [])) if isinstance(ing, dict) else set(getattr(ing, "visits", []))
    for v in must:
        out.append(_check(f"visit_present:{v}", v in visits, v, sorted(visits)))
    return out


# ---------- Q&A checks ----------
def _eval_qa(memory: WorkingMemory, qa_specs: list[dict[str, Any]]) -> list[Check]:
    out: list[Check] = []
    for i, spec in enumerate(qa_specs):
        q = spec.get("question", "")
        try:
            ans = answer(memory, q)
            text = (ans.answer or "").lower()
            cited_files = {c.file for c in ans.sources}
        except Exception as e:
            out.append(_check(f"qa[{i}].error", False, note=str(e)[:160]))
            continue

        for needle in spec.get("must_contain", []):
            out.append(_check(
                f"qa[{i}].must_contain:{needle}",
                needle.lower() in text, needle, ans.answer[:200],
            ))
        any_needles = spec.get("must_contain_any", [])
        if any_needles:
            out.append(_check(
                f"qa[{i}].must_contain_any",
                any(n.lower() in text for n in any_needles),
                any_needles, ans.answer[:200],
            ))
        any_cites = spec.get("must_cite_any", [])
        if any_cites:
            out.append(_check(
                f"qa[{i}].must_cite_any",
                any(c in cited_files for c in any_cites),
                any_cites, sorted(cited_files),
            ))
    return out


# ---------- runner ----------
def run_patient_eval(
    spec: dict[str, Any],
    *,
    skip_pipeline: bool = False,
) -> PatientResult:
    pid = spec["patient_id"]
    expected = spec.get("expected", {})
    qa_specs = spec.get("qa", [])

    t0 = time.perf_counter()
    pipeline_ok = True
    if not skip_pipeline:
        try:
            r = run_patient(pid)
            pipeline_ok = all(p.get("ok") for p in r["phases"])
        except Exception:
            pipeline_ok = False

    memory = WorkingMemory.load(pid)
    checks: list[Check] = []
    checks += _eval_record(memory, expected)
    checks += _eval_recist(memory, expected)
    checks += _eval_urgency(memory, expected)
    checks += _eval_meds(memory, expected)
    checks += _eval_interactions(memory, expected)
    checks += _eval_visits(memory, expected)

    qa_checks = _eval_qa(memory, qa_specs) if qa_specs else []

    return PatientResult(
        patient_id=pid,
        duration_s=round(time.perf_counter() - t0, 2),
        pipeline_ok=pipeline_ok,
        checks=checks,
        qa_checks=qa_checks,
    )


def run_eval(gold_path: Path, *, skip_pipeline: bool = False) -> list[PatientResult]:
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    results: list[PatientResult] = []
    for spec in data.get("patients", []):
        results.append(run_patient_eval(spec, skip_pipeline=skip_pipeline))
    return results


# ---------- reporting ----------
def print_report(results: list[PatientResult]) -> int:
    total = sum(r.total for r in results)
    passed = sum(r.passed for r in results)
    print(f"\n=== Eval report ({passed}/{total} checks passed) ===")
    for r in results:
        flag = "OK" if r.pipeline_ok else "PIPELINE_FAIL"
        print(f"\n[{r.patient_id}] {r.passed}/{r.total} checks ({r.duration_s}s) {flag}")
        for c in r.checks + r.qa_checks:
            mark = "✓" if c.ok else "✗"
            extra = ""
            if not c.ok:
                extra = f"  expected={c.expected!r} actual={c.actual!r}"
                if c.note:
                    extra += f" note={c.note!r}"
            print(f"  {mark} {c.name}{extra}")
    return 0 if passed == total else 1


def write_xlsx(results: list[PatientResult], path: Path) -> None:
    try:
        from openpyxl import Workbook
    except Exception:
        print("openpyxl not installed; skipping xlsx export", file=sys.stderr)
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["patient_id", "passed", "total", "pipeline_ok", "duration_s"])
    for r in results:
        ws.append([r.patient_id, r.passed, r.total, r.pipeline_ok, r.duration_s])
    detail = wb.create_sheet("checks")
    detail.append(["patient_id", "kind", "name", "ok", "expected", "actual", "note"])
    for r in results:
        for c in r.checks:
            detail.append([r.patient_id, "stage", c.name, c.ok, repr(c.expected), repr(c.actual), c.note])
        for c in r.qa_checks:
            detail.append([r.patient_id, "qa", c.name, c.ok, repr(c.expected), repr(c.actual), c.note])
    wb.save(path)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="reuse previously persisted WorkingMemory instead of re-running run_patient")
    ap.add_argument("--xlsx", type=Path, help="optional xlsx report path")
    args = ap.parse_args()

    results = run_eval(args.gold, skip_pipeline=args.skip_pipeline)
    code = print_report(results)
    if args.xlsx:
        write_xlsx(results, args.xlsx)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
