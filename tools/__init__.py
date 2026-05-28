"""Tool registry.

Each phase's tools are implemented in one module. The registry maps a
tool *name* (as declared in tool_schemas) to a Python callable with the
signature:

    def tool_fn(memory: WorkingMemory, **arguments) -> dict

The orchestrator looks up the callable by name and dispatches the
arguments the LLM emitted during its tool-call round.
"""
from __future__ import annotations

from typing import Any, Callable

from ..memory import WorkingMemory

ToolFn = Callable[..., dict[str, Any]]

_REGISTRY: dict[str, ToolFn] = {}


def register(name: str) -> Callable[[ToolFn], ToolFn]:
    def deco(fn: ToolFn) -> ToolFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> ToolFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown tool: {name}")
    return _REGISTRY[name]


def dispatch(name: str, memory: WorkingMemory, arguments: dict[str, Any]) -> dict[str, Any]:
    fn = get(name)
    return fn(memory=memory, **(arguments or {}))


def known() -> list[str]:
    return sorted(_REGISTRY.keys())


# Import side-effect modules so their @register decorators run.
from . import ingest  # noqa: E402,F401
from . import mri_agent  # noqa: E402,F401
from . import recist_agent  # noqa: E402,F401
from . import treatment_opt_agent  # noqa: E402,F401  Phase 4 SMBO v3.0
from . import clinical_trial_match  # noqa: E402,F401  Phase 4 Task 8
from . import pubmed_evidence       # noqa: E402,F401  Phase 5.5 / Module 3
from . import faers_check           # noqa: E402,F401  Phase 5.7 / Extra B
from . import pharma_agent  # noqa: E402,F401
from . import synthesis_agent  # noqa: E402,F401
from . import chat_agent  # noqa: E402,F401
