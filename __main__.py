"""Allows: python -m neuro_agent  → starts the API server."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port   = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "0").lower() not in ("0", "false", "no")
    uvicorn.run(
        "neuro_agent.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
