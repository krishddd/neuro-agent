"""Smoke tests — verify all core modules import without errors.

These run in CI without Ollama / ChromaDB / Google credentials present.
"""


def test_config_imports():
    import neuro_agent.config as cfg
    assert cfg.MODEL_PRIMARY
    assert cfg.OUTPUTS_DIR


def test_api_imports():
    from neuro_agent.api.app import app
    assert app.title == "Neuro-Oncology Unified Care Agent"


def test_integrations_imports():
    from neuro_agent.integrations.patient_roster import PATIENT_EMAILS
    assert "P001" in PATIENT_EMAILS
    assert len(PATIENT_EMAILS) == 21   # P001-P020 + HRK demo patient


def test_tools_imports():
    import neuro_agent.tools.ingest          # noqa: F401
    import neuro_agent.tools.mri_agent       # noqa: F401
    import neuro_agent.tools.recist_agent    # noqa: F401
    import neuro_agent.tools.pharma_agent    # noqa: F401
    import neuro_agent.tools.synthesis_agent # noqa: F401


def test_google_chat_router():
    from neuro_agent.api.routers.google_chat import router
    assert router.prefix == ""
