"""Start the FastAPI server.

Run from INSIDE the neuro_agent/ folder:
    python run_server.py
    python run_server.py --port 9000
    RELOAD=1 python run_server.py   (enable hot-reload, off by default)
"""
from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path

# ── sys.path fix (must be before any neuro_agent imports) ─────────────────────
_PKG    = Path(__file__).resolve().parent       # neuro_agent/
_PARENT = _PKG.parent                           # parent dir

if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ── Logging: show pipeline + uvicorn activity in the terminal ────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "pipeline": {
            "format": "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "pipeline",
        },
    },
    "loggers": {
        # Pipeline activity
        "neuro_agent": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        # Uvicorn startup/access/error messages
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {"level": "WARNING"},
})

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port   = int(os.environ.get("PORT", 8000))
    # Reload is OFF by default — on Windows, watchfiles scans all files in the
    # folder (including chroma_db/ and outputs/) which causes very slow startup.
    # Enable with:  RELOAD=1 python run_server.py
    reload = os.environ.get("RELOAD", "0").lower() not in ("0", "false", "no")

    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    uvicorn.run(
        "neuro_agent.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        reload_dirs=[str(_PKG)] if reload else None,
        log_config=None,   # logging configured above via dictConfig
    )
