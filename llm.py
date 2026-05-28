"""Thin wrapper around the local Ollama server.

Exposes four call styles the rest of the agent relies on:

    chat(messages)                 -> str
    json_call(messages, Model)     -> Pydantic instance  (validated, retried)
    tool_call(messages, tools)     -> {"content": str, "tool_calls": [...]}
    vision(prompt, images)         -> str
    stream(messages)               -> Iterator[str]
    embed(texts)                   -> list[list[float]]

All calls route through one Ollama client pinned to OLLAMA_HOST.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterator, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None  # type: ignore

from .config import (
    LLM_MAX_RETRIES,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    MODEL_EMBED,
    MODEL_PRIMARY,
    MODEL_VISION,
    OLLAMA_HOST,
)

M = TypeVar("M", bound=BaseModel)


def _client() -> "ollama.Client":
    if ollama is None:
        raise RuntimeError("ollama package not installed — pip install ollama")
    return ollama.Client(host=OLLAMA_HOST, timeout=LLM_TIMEOUT_S)


_DEFAULT_OPTIONS: dict[str, Any] = {
    "temperature": LLM_TEMPERATURE,
    "num_ctx": LLM_NUM_CTX,
}


# ---------- basic chat ----------
def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = MODEL_PRIMARY,
    options: dict[str, Any] | None = None,
) -> str:
    opts = {**_DEFAULT_OPTIONS, **(options or {})}
    resp = _client().chat(model=model, messages=messages, options=opts)
    return resp["message"]["content"]


def stream(
    messages: list[dict[str, Any]],
    *,
    model: str = MODEL_PRIMARY,
    options: dict[str, Any] | None = None,
) -> Iterator[str]:
    opts = {**_DEFAULT_OPTIONS, **(options or {})}
    for chunk in _client().chat(
        model=model, messages=messages, options=opts, stream=True
    ):
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


# ---------- structured JSON ----------
class LLMJSONError(RuntimeError):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    retry=retry_if_exception_type((LLMJSONError, ValidationError)),
)
def json_call(
    messages: list[dict[str, Any]],
    schema: Type[M],
    *,
    model: str = MODEL_PRIMARY,
    options: dict[str, Any] | None = None,
) -> M:
    """Call Ollama in JSON mode and validate against a Pydantic schema."""
    opts = {**_DEFAULT_OPTIONS, **(options or {})}
    resp = _client().chat(
        model=model, messages=messages, format="json", options=opts
    )
    raw = resp["message"]["content"] or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMJSONError(f"invalid JSON from model: {e}") from e
    return schema.model_validate(data)


# ---------- function / tool calling ----------
def tool_call(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    model: str = MODEL_PRIMARY,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose an Ollama-native function-calling round.

    Returns {"content": str|None, "tool_calls": [{"name", "arguments"}...]}.
    The caller (orchestrator) dispatches each tool_call and appends the
    tool response to the message list before looping.
    """
    opts = {**_DEFAULT_OPTIONS, **(options or {})}
    resp = _client().chat(
        model=model, messages=messages, tools=tools, options=opts
    )
    msg = resp["message"]
    # Defensive extraction — some Ollama models occasionally return malformed
    # tool_call entries with missing `function` or `function.name`. Skip those
    # instead of crashing the phase with a KeyError.
    parsed_calls: list[dict[str, Any]] = []
    for tc in (msg.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        parsed_calls.append({"name": name, "arguments": fn.get("arguments", {})})
    return {"content": msg.get("content"), "tool_calls": parsed_calls}


# ---------- vision ----------
def _encode_image(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def vision(
    prompt: str,
    images: list[str | Path],
    *,
    model: str = MODEL_VISION,
    options: dict[str, Any] | None = None,
) -> str:
    """Send one user turn containing text + one or more images."""
    opts = {**_DEFAULT_OPTIONS, **(options or {})}
    msg = {
        "role": "user",
        "content": prompt,
        "images": [_encode_image(p) for p in images],
    }
    resp = _client().chat(model=model, messages=[msg], options=opts)
    return resp["message"]["content"]


def vision_json(
    prompt: str,
    images: list[str | Path],
    schema: Type[M],
    *,
    model: str = MODEL_VISION,
    options: dict[str, Any] | None = None,
) -> M:
    """Vision call forced into JSON + Pydantic validation."""
    opts = {**_DEFAULT_OPTIONS, **(options or {})}
    msg = {
        "role": "user",
        "content": prompt,
        "images": [_encode_image(p) for p in images],
    }
    resp = _client().chat(
        model=model, messages=[msg], format="json", options=opts
    )
    raw = resp["message"]["content"] or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMJSONError(f"invalid JSON from vision model: {e}") from e
    return schema.model_validate(data)


# ---------- embeddings ----------
def embed(texts: list[str], *, model: str = MODEL_EMBED) -> list[list[float]]:
    if not texts:
        return []
    c = _client()
    # Newer ollama-python supports batched .embed(input=[...]); fall back to
    # per-item .embeddings(prompt=...) for older versions.
    if hasattr(c, "embed"):
        r = c.embed(model=model, input=texts)
        return list(r.get("embeddings") or [])
    return [c.embeddings(model=model, prompt=t)["embedding"] for t in texts]


# ---------- health ----------
def ping() -> bool:
    try:
        _client().list()
        return True
    except Exception:
        return False
