"""Typer CLI for the Neuro-Oncology Unified Care Agent.

Commands
--------
    python -m neuro_agent.main run    --patient P001 [--stop-after mri]
    python -m neuro_agent.main qa     --patient P001 --q "What changed?"
    python -m neuro_agent.main chat   --patient P001
    python -m neuro_agent.main batch  [--start P001] [--limit 5]
    python -m neuro_agent.main list-patients
    python -m neuro_agent.main ping
    python -m neuro_agent.main serve  [--host 0.0.0.0 --port 8000]
"""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import DATA_ROOT, DISCLAIMER, PHASE_NAMES
from .llm import ping as llm_ping
from .memory import WorkingMemory
from .orchestrator import run_patient
from .tools.chat_agent import answer

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


def _patient_dirs() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


@app.command()
def ping() -> None:
    """Check that the local Ollama server is reachable."""
    ok = llm_ping()
    console.print(f"ollama: {'[green]ok[/]' if ok else '[red]down[/]'}")
    raise typer.Exit(code=0 if ok else 1)


@app.command("list-patients")
def list_patients() -> None:
    """List patient folders discovered under DATA_ROOT."""
    pids = _patient_dirs()
    table = Table(title=f"Patients in {DATA_ROOT}")
    table.add_column("#", justify="right")
    table.add_column("patient_id")
    for i, pid in enumerate(pids, 1):
        table.add_row(str(i), pid)
    console.print(table)
    console.print(f"[dim]{len(pids)} patient(s) found[/]")


@app.command()
def run(
    patient: str = typer.Option(..., "--patient", "-p"),
    stop_after: Optional[str] = typer.Option(
        None, "--stop-after",
        help=f"One of: {', '.join(PHASE_NAMES)}",
    ),
    mode: str = typer.Option(
        "prep", "--mode",
        help="HITL mode: prep (phases 1-4, default), execute (5-6, needs APPROVED.json), full (legacy).",
    ),
    approver_email: Optional[str] = typer.Option(
        None, "--approver-email",
        help="For --mode=execute: the clinician who approved. Writes APPROVED.json if absent.",
    ),
    approve_decision: str = typer.Option(
        "APPROVE", "--approve-decision",
        help="APPROVE | REJECT | MODIFY — used only when --approver-email is set.",
    ),
    override_regimen: Optional[str] = typer.Option(
        None, "--override-regimen",
        help="Clinician-edited regimen (required when --approve-decision=MODIFY).",
    ),
) -> None:
    """Run the guarded pipeline for one patient."""
    if stop_after and stop_after not in PHASE_NAMES:
        console.print(f"[red]invalid --stop-after; choose from {PHASE_NAMES}[/]")
        raise typer.Exit(2)
    if mode not in ("prep", "execute", "full"):
        console.print(f"[red]invalid --mode {mode!r}; choose prep|execute|full[/]")
        raise typer.Exit(2)

    # CLI convenience: if the user passes --approver-email alongside mode=execute,
    # auto-write APPROVED.json before running phases 5-6.
    if mode == "execute" and approver_email:
        from .utils.approval import record_approval
        if approve_decision not in ("APPROVE", "REJECT", "MODIFY"):
            console.print(
                f"[red]invalid --approve-decision {approve_decision!r}; "
                f"choose APPROVE|REJECT|MODIFY[/]"
            )
            raise typer.Exit(2)
        if approve_decision == "MODIFY" and not (override_regimen or "").strip():
            console.print("[red]--approve-decision=MODIFY requires --override-regimen[/]")
            raise typer.Exit(2)
        rec = record_approval(
            patient,
            approver_email=approver_email,
            decision=approve_decision,  # type: ignore[arg-type]
            clinician_notes="CLI approval",
            override_regimen=override_regimen,
        )
        console.print(f"[yellow]wrote approval marker[/] decision={rec.decision}")

    console.print(f"[bold]Running pipeline[/] for [cyan]{patient}[/] (mode={mode})")
    result = run_patient(patient, stop_after=stop_after, mode=mode)
    table = Table(title=f"Job {result['job_id']} — {patient}")
    table.add_column("phase")
    table.add_column("ok")
    table.add_column("steps", justify="right")
    for ph in result["phases"]:
        table.add_row(
            ph.get("phase", "?"),
            "[green]yes[/]" if ph.get("ok") else "[red]no[/]",
            str(ph.get("steps", 0)),
        )
    console.print(table)
    console.print(f"qa_ready: {result['qa_ready']}")
    if result.get("pending_approval"):
        pa = result["pending_approval"]
        console.print(
            f"[yellow]PENDING_APPROVAL.json written[/] — decision={pa.get('mdt_decision')}, "
            f"regimen={pa.get('proposed_regimen')!r}"
        )
        console.print(
            f"Next: [cyan]python -m neuro_agent.main run --patient {patient} "
            f"--mode execute --approver-email <you@hospital>[/]"
        )


@app.command()
def batch(
    start: Optional[str] = typer.Option(None, "--start"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    stop_after: Optional[str] = typer.Option(None, "--stop-after"),
) -> None:
    """Run the pipeline across all (or a subset of) patients in DATA_ROOT."""
    pids = _patient_dirs()
    if start:
        pids = [p for p in pids if p >= start]
    if limit:
        pids = pids[:limit]
    if not pids:
        console.print("[yellow]no patients to process[/]")
        raise typer.Exit(0)

    summary: list[dict] = []
    for pid in pids:
        console.print(f"[bold]→[/] {pid}")
        try:
            r = run_patient(pid, stop_after=stop_after)
            summary.append({
                "patient": pid,
                "ok": all(p.get("ok") for p in r["phases"]),
                "phases": len(r["phases"]),
            })
        except Exception as e:
            summary.append({"patient": pid, "ok": False, "error": str(e)[:120]})

    table = Table(title="Batch summary")
    table.add_column("patient")
    table.add_column("ok")
    table.add_column("phases", justify="right")
    table.add_column("error")
    for s in summary:
        table.add_row(
            s["patient"],
            "[green]yes[/]" if s["ok"] else "[red]no[/]",
            str(s.get("phases", "")),
            s.get("error", ""),
        )
    console.print(table)


@app.command()
def qa(
    patient: str = typer.Option(..., "--patient", "-p"),
    q: str = typer.Option(..., "--q", "--question"),
) -> None:
    """Single-turn cited Q&A against an already-processed patient."""
    memory = WorkingMemory.load(patient)
    if not memory.has(WorkingMemory.INGESTION):
        console.print(f"[red]patient {patient} has not been processed yet[/]")
        raise typer.Exit(2)
    out = answer(memory, q)
    console.print(f"[bold]Answer:[/] {out.answer}")
    if out.sources:
        console.print("[bold]Sources:[/]")
        for s in out.sources:
            extras = []
            if s.visit:
                extras.append(f"visit {s.visit}")
            if s.chunk is not None:
                extras.append(f"chunk {s.chunk}")
            console.print(f"  - {s.file} ({', '.join(extras)})" if extras else f"  - {s.file}")
    console.print(f"[dim]confidence: {out.confidence}[/]")
    console.print(f"[dim]{DISCLAIMER}[/]")


@app.command()
def chat(
    patient: str = typer.Option(..., "--patient", "-p"),
) -> None:
    """Interactive multi-turn chat session against one patient."""
    from . import chat_sessions

    memory = WorkingMemory.load(patient)
    if not memory.has(WorkingMemory.INGESTION):
        console.print(f"[red]patient {patient} has not been processed yet[/]")
        raise typer.Exit(2)

    sess = chat_sessions.create(patient)
    console.print(f"[bold]Chat session[/] [cyan]{sess.session_id[:8]}[/] for [cyan]{patient}[/]")
    console.print("[dim]type /exit to quit, /reset to clear history[/]")

    while True:
        try:
            line = console.input("[bold green]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            break
        if line == "/reset":
            sess.history.clear()
            console.print("[dim]history cleared[/]")
            continue

        out = answer(memory, line, history=sess.messages())
        sess.append("user", line)
        sess.append("assistant", out.answer)
        console.print(f"[bold blue]agent ›[/] {out.answer}")
        if out.sources:
            cites = ", ".join(
                f"{s.file}{f' v{s.visit}' if s.visit else ''}{f' c{s.chunk}' if s.chunk is not None else ''}"
                for s in out.sources
            )
            console.print(f"[dim]sources: {cites}[/]")
        console.print(f"[dim]confidence: {out.confidence}[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Launch the FastAPI gateway."""
    import uvicorn

    uvicorn.run(
        "neuro_agent.api.app:app",
        host=host, port=port, reload=reload,
    )


@app.command()
def memory_dump(
    patient: str = typer.Option(..., "--patient", "-p"),
) -> None:
    """Print the persisted WorkingMemory snapshot for a patient."""
    mem = WorkingMemory.load(patient)
    console.print_json(json.dumps(mem.snapshot_for_llm(), default=str))


if __name__ == "__main__":
    app()
