# neuro-agent

> A tool-using clinical AI agent for neuro-oncology workflows — summarises
> imaging and pathology, retrieves literature and guidelines, drafts care-plan
> suggestions, and integrates with Gmail, Drive, Chat, and Calendar. Built
> around per-patient ChromaDB memory with strict isolation.

> ⚠️ **Research / portfolio code.** Not a medical device. Not for clinical
> use. Never makes autonomous treatment decisions — every output is staged
> for human review.

---

## What the agent actually does

`neuro-agent` is a FastAPI service that exposes an LLM-driven workflow for a
neuro-oncology MDT (multi-disciplinary team). A clinician (or an upstream
orchestrator) submits a patient case identifier; the agent then:

1. Loads the patient's bi-temporal memory store.
2. Runs a tool-using agent loop (`agent/`) that may call:
   - **Imaging tools** — volumetric segmentation helpers from
     `utils/volumetric_seg.py` (MRI / CT inputs).
   - **Literature tools** — retrieves and re-ranks recent papers from the
     drug-interaction KB and configurable PubMed/RAG sources.
   - **Guideline tools** — looks up NCCN / ESMO-style guidelines from a
     configurable knowledge folder.
   - **FHIR helpers** — `api/routers/smart_fhir.py` reads / writes against a
     SMART-on-FHIR endpoint to surface observations and conditions.
   - **Patient memory tools** — read / append / supersede facts in the
     patient's ChromaDB and bi-temporal store.
3. Synthesises a care-plan draft with citations and uncertainty markers.
4. Writes the draft to `wiki/review/` for clinician review.
5. Optionally delivers a notification through Gmail or Chat
   (`integrations/`).

The agent never auto-approves. The clinician explicitly accepts or rejects
the draft via the chat router or a downstream UI.

---

## End-to-end pipeline

```
patient_id + request
       │
       ▼
api/app.py  →  api/routers/process.py
       │
       ▼
load patient context
       ├─ memory.py             ← bi-temporal facts + access counts
       ├─ per-patient ChromaDB  ← chroma_db/<PID>/
       └─ drug-interaction KB   ← chroma_db/_shared/
       │
       ▼
orchestrator.py
       │
       ▼
agent loop  (LLM tool-use, qwen3:14b reasoning model)
       │
       ├──► imaging tools         (volumetric_seg)
       ├──► literature retrieval  (RAG over papers + KB)
       ├──► guideline lookup
       ├──► FHIR observations     (smart_fhir router)
       └──► memory read / append  (with bi-temporal supersession)
       │
       ▼
draft care plan with citations
       │
       ▼
finalize  →  legacy-cleanup of stale stages
       │
       ▼
wiki/review/<patient_id>/<date>.md
       │
       ▼
notify (optional) — Gmail / Chat integration
       │
       ▼
clinician accepts or rejects
       │
       ▼
on accept: stages/S05_index.json updated, audit log written
```

---

## Per-patient memory isolation

`memory.py` enforces strict patient isolation:

- Each patient gets their own ChromaDB collection at
  `chroma_db/<patient_id>/chroma.sqlite3`.
- The drug-interaction KB is the only **shared** collection, indexed once at
  `chroma_db/_shared/` and reused across patients.
- Patient facts are **bi-temporal**: `ingested_at`, `valid_from`,
  `valid_to`, `superseded_by`, `last_reinforced`, `access_count`. A new
  source never deletes a fact — it marks the old one superseded.
- `_finalize` runs a legacy cleanup pass: pads stage filenames
  (`S{int:02d}_{rest}.json`), moves stray root files into their canonical
  subfolders, and deletes known-stale outputs (`recist_lesions.json`).
- Every write is audit-logged with the patient ID and correlation ID.

---

## Tool catalogue

| Tool                 | Where                              | Purpose                                    |
|----------------------|------------------------------------|--------------------------------------------|
| `volumetric_seg`     | `utils/volumetric_seg.py`          | MRI / CT segmentation helpers              |
| `tool_helpers`       | `utils/tool_helpers.py`            | ChromaDB client + RAG helpers              |
| `memory.*`           | `memory.py`                        | Bi-temporal read / append / supersede      |
| `smart_fhir.*`       | `api/routers/smart_fhir.py`        | SMART-on-FHIR observations / conditions    |
| `process.*`          | `api/routers/process.py`           | Case-processing entry-points               |
| `chat.*`             | `api/routers/chat.py`              | Free-text Q&A over a patient               |
| `google_chat.*`      | `api/routers/google_chat.py`       | Google Chat webhook adapter                |
| `gmail_client`       | `integrations/gmail_client.py`     | Send / fetch notifications                 |
| `drive_client`       | `integrations/drive_client.py`     | Read patient documents from Drive          |
| `calendar_client`    | `integrations/calendar_client.py`  | Schedule follow-ups                        |
| `chat_bot`           | `integrations/chat_bot.py`         | Bot-mode chat interactions                 |

The agent loop in `agent/` selects from this set at every step.

---

## Quickstart

```bash
git clone https://github.com/krishddd/neuro-agent.git
cd neuro-agent
pip install -r requirements.txt
cp .env.example .env  # OLLAMA host + (optional) Google OAuth creds

# Start the API
uvicorn api.app:app --reload --port 8000

# Run the test suite
pytest tests/ -v
```

Process a patient case (HTTP):

```bash
curl -X POST http://localhost:8000/process \
     -H 'Content-Type: application/json' \
     -d '{"patient_id": "P001", "request": "Summarise the latest scan and propose a discussion plan for tomorrow's MDT."}'
```

Free-text Q&A:

```bash
curl -X POST http://localhost:8000/chat \
     -H 'Content-Type: application/json' \
     -d '{"patient_id": "P001", "message": "What changed between the last two MRI sessions?"}'
```

---

## Project structure

```
api/
├── app.py                FastAPI app factory
└── routers/
    ├── chat.py           Free-text Q&A
    ├── process.py        Case-processing entry-point
    ├── google_chat.py    Google Chat webhook adapter
    └── smart_fhir.py     SMART-on-FHIR helpers
agent/                    Tool-using agent loop
integrations/
├── chat_bot.py
├── calendar_client.py
├── drive_client.py
└── gmail_client.py
utils/
├── tool_helpers.py       ChromaDB client + helpers
└── volumetric_seg.py     Imaging segmentation helpers
memory.py                 Per-patient bi-temporal memory store
orchestrator.py           Top-level workflow runner
auth.py                   OAuth + token storage
__main__.py               python -m neuro_agent
tests/                    Unit + integration tests
ruff.toml                 Strict ruff config (pragmatic ignore list)
```

---

## Configuration

All knobs are env-driven. Key settings:

| Env var                  | Meaning                                                    |
|--------------------------|------------------------------------------------------------|
| `OLLAMA_HOST`            | URL of the Ollama server hosting `qwen3:14b`               |
| `CHROMA_DB_ROOT`         | Filesystem root for per-patient ChromaDB collections       |
| `GOOGLE_OAUTH_*`         | Client / token files for Gmail / Drive / Chat / Calendar  |
| `WIKI_REVIEW_DIR`        | Where review drafts are staged for clinician accept        |
| `LOG_LEVEL`              | JSON log verbosity                                         |

Drug-interaction KB is indexed once on first start; subsequent starts reuse
the cached index.

---

## CI

GitHub Actions runs:

- **ruff** — strict lint with the `ruff.toml` ignore list pragmatically
  tuned for medical-domain code.
- **syntax check** — `python -m compileall .`
- **import check** + **pytest** (non-blocking — many tests require Ollama).
- **Docker build** — validates the image builds.

CD only runs on `v*.*.*` tags and is gated behind `DOCKERHUB_USERNAME` /
`DOCKERHUB_TOKEN` secrets, so day-to-day pushes never fail on a missing
registry secret.

---

## Status

Research prototype. Designed for human-in-the-loop oversight only. Not
intended for clinical decision-making.

## License

MIT
