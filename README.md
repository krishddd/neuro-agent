# neuro-agent

> A tool-using clinical AI agent for neuro-oncology workflows.

`neuro-agent` is a FastAPI service that helps clinicians triage neuro-oncology
patient cases. It summarises imaging and pathology, retrieves relevant
guidelines and literature, and drafts care-plan suggestions — always surfaced
for human review.

> ⚠️ **Research / portfolio code.** Not a medical device. Not for clinical
> use. Never makes autonomous treatment decisions.

## Features

- **Multi-tool agent loop** — imaging analysis, literature retrieval,
  guideline lookup, FHIR helpers.
- **Volumetric segmentation utilities** for MRI / CT inputs.
- **Per-patient ChromaDB** — every patient gets an isolated vector store, no
  cross-patient leakage.
- **Drug-interaction KB** indexed once and shared across patients.
- **Google integrations** — Gmail / Drive / Chat / Calendar adapters for
  workflow automation.
- **Patient memory** — bi-temporal store with stage-based file layout and
  legacy-cleanup on finalize.

## Tech stack

Python · FastAPI · ChromaDB · Ollama (qwen3:14b for summaries) · Google APIs ·
LLM tool-use

## Quickstart

```bash
git clone https://github.com/krishddd/neuro-agent.git
cd neuro-agent
pip install -r requirements.txt
cp .env.example .env  # add Ollama host + Google OAuth creds if using

# Start the API
uvicorn api.app:app --reload --port 8000
```

Run the test suite:

```bash
pytest tests/ -v
```

## Project structure

```
api/           FastAPI app + routers (chat, process, smart_fhir, ...)
agent/         Tool-using agent loop
integrations/  Gmail, Drive, Chat, Calendar clients
utils/         tool_helpers, volumetric_seg, clinical helpers
memory.py      Per-patient bi-temporal memory store
orchestrator.py  Top-level workflow runner
tests/         Unit + integration tests
```

## Status

Research prototype. Designed for human-in-the-loop oversight only.

## License

MIT
