"""Phase 5.2 / Module 5 — Pharmacogenomic (PGx) dose/response adjuster.

Maps a ``(drug, patient_genotype)`` pair to ``(efficacy_multiplier,
toxicity_multiplier, notes)`` using curated CPIC/PharmGKB phenotype tables
in ``Datasets/reference/pgx_drug_map.json``.

Phenoconversion (Phase 5.2b, deferred):
    The public signature ``apply_pgx(drug, patient_genotype, concomitant_meds=None)``
    reserves the API surface for dynamic phenoconversion — a patient who is
    genetically a ``normal_metabolizer`` functionally behaves like a
    ``poor_metabolizer`` while co-prescribed a strong CYP inhibitor.
    In Phase 5.2 ``_resolve_effective_phenotype()`` returns the raw
    genotype; the resolver will consult ``cyp_inhibitors.json`` + the
    concomitant medication list when 5.2b ships.

Graceful degrade:
    - missing map file, unknown drug, or unknown phenotype → (1.0, 1.0, [])
    - ``PGX_ENABLED=false`` → (1.0, 1.0, ["pgx disabled"])

The caller (``smbo_engine``) multiplies SMBO severity_index by ``tox_mult``
and PFS benefit by ``eff_mult``, then surfaces ``notes`` in the candidate's
``pgx_notes`` field for the MDT to cite.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .. import config

logger = logging.getLogger(__name__)

# ---- genotype field name → enzyme key in PGX map ----
# patient_state.py populates PatientPharmacogenomics with these field names
# (lowercase ``cyp2d6``/``tpmt``/...). The PGX map keys enzymes in uppercase
# (``CYP2D6``/``TPMT``/...) per clinical convention.
_GENOTYPE_FIELD_TO_ENZYME = {
    "cyp2d6":  "CYP2D6",
    "cyp2c19": "CYP2C19",
    "cyp2c9":  "CYP2C9",
    "cyp3a4":  "CYP3A4",
    "cyp2b6":  "CYP2B6",
    "cyp2c8":  "CYP2C8",
    "tpmt":    "TPMT",
    "dpyd":    "DPYD",
    "ugt1a1":  "UGT1A1",
    "mthfr":   "MTHFR",
    "gstp1":   "GSTP1",
    "cbr3":    "CBR3",
    "hla_b":   "HLA-B*1502",
    "mgmt":    "MGMT",
    "vegfa":   "VEGFA",
}

_DEFAULT_PHENOTYPE = "normal_metabolizer"


@lru_cache(maxsize=1)
def _load_drug_map() -> dict:
    """Load and cache pgx_drug_map.json. Empty dict on failure."""
    path = Path(config.PGX_DRUG_MAP_PATH)
    if not path.exists():
        logger.warning("pgx_drug_map.json not found at %s — PGx returns 1.0 multipliers", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop the meta ``_schema`` key if present.
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError) as e:
        logger.error("pgx_drug_map.json load failed: %s", e)
        return {}


@lru_cache(maxsize=1)
def _load_inhibitor_map() -> dict:
    """Load and cache cyp_inhibitors.json. Reserved for Phase 5.2b."""
    path = Path(config.CYP_INHIBITORS_PATH)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError):
        return {}


def _genotype_for_enzyme(enzyme: str, patient_genotype) -> str:
    """Look up the phenotype string for an enzyme in the patient profile.

    ``patient_genotype`` may be a ``PatientPharmacogenomics`` model instance,
    a plain ``dict``, or ``None``.
    """
    if patient_genotype is None:
        return _DEFAULT_PHENOTYPE

    # Reverse map enzyme → field name on the PatientPharmacogenomics model.
    field = next(
        (k for k, v in _GENOTYPE_FIELD_TO_ENZYME.items() if v == enzyme),
        None,
    )
    if field is None:
        return _DEFAULT_PHENOTYPE

    # Pydantic model
    if hasattr(patient_genotype, field):
        value = getattr(patient_genotype, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # dict / dict-like
    if isinstance(patient_genotype, dict):
        if field in patient_genotype and isinstance(patient_genotype[field], str):
            return patient_genotype[field].strip() or _DEFAULT_PHENOTYPE
        # Also accept the uppercase enzyme as a direct key
        if enzyme in patient_genotype and isinstance(patient_genotype[enzyme], str):
            return patient_genotype[enzyme].strip() or _DEFAULT_PHENOTYPE

    return _DEFAULT_PHENOTYPE


def _resolve_effective_phenotype(
    drug: str,
    patient_genotype,
    concomitant_meds: Optional[Iterable[str]] = None,
) -> tuple[str, str, list[str]]:
    """Resolve the *effective* phenotype for this drug, accounting for
    phenoconversion (Phase 5.2b).

    Returns ``(enzyme, effective_phenotype, notes)``.

    In Phase 5.2 this returns the raw genotype unchanged — ``concomitant_meds``
    is accepted for API stability but not yet consulted. When 5.2b activates,
    a strong inhibitor downgrades normal_metabolizer → poor_metabolizer, etc.,
    per ``cyp_inhibitors.json._schema.downgrade_rules``.
    """
    dmap = _load_drug_map()
    entry = dmap.get((drug or "").lower())
    if not entry:
        return ("", _DEFAULT_PHENOTYPE, [])

    enzyme = entry.get("enzyme", "")
    raw_phenotype = _genotype_for_enzyme(enzyme, patient_genotype)
    notes: list[str] = []

    # --- Phase 5.2b scaffold (inactive) ---
    # if concomitant_meds:
    #     imap = _load_inhibitor_map().get(enzyme, {})
    #     for strength in ("strong", "moderate"):
    #         inhibitors = set(d.lower() for d in imap.get(strength, []))
    #         hit = inhibitors.intersection({m.lower() for m in concomitant_meds})
    #         if hit:
    #             downgrade = imap.get("_schema", {}).get("downgrade_rules", {}).get(strength, {})
    #             new_pheno = downgrade.get(raw_phenotype)
    #             if new_pheno and new_pheno != raw_phenotype:
    #                 notes.append(
    #                     f"phenoconversion: {enzyme} {raw_phenotype} → {new_pheno} "
    #                     f"via {strength} inhibitor(s) {sorted(hit)}"
    #                 )
    #                 raw_phenotype = new_pheno

    return (enzyme, raw_phenotype, notes)


def apply_pgx(
    drug: str,
    patient_genotype,
    concomitant_meds: Optional[Iterable[str]] = None,
) -> tuple[float, float, list[str]]:
    """Return ``(efficacy_mult, toxicity_mult, notes)`` for a drug.

    Parameters
    ----------
    drug
        Drug name (case-insensitive; matches keys in pgx_drug_map.json).
    patient_genotype
        A ``PatientPharmacogenomics`` instance, a ``dict``, or ``None``.
    concomitant_meds
        Current medications; reserved for Phase 5.2b phenoconversion.

    Returns
    -------
    (eff_mult, tox_mult, notes)
        Multipliers default to ``(1.0, 1.0, [])`` when PGx is disabled,
        the drug is not in the map, or the patient's phenotype is unknown.
        ``notes`` contains 0+ human-readable strings describing why
        multipliers deviate from 1.0 (for SMBO candidate audit trail).
    """
    if not getattr(config, "PGX_ENABLED", False):
        return (1.0, 1.0, [])
    if not drug:
        return (1.0, 1.0, [])

    dmap = _load_drug_map()
    entry = dmap.get(drug.lower())
    if not entry:
        return (1.0, 1.0, [])

    enzyme, phenotype, notes = _resolve_effective_phenotype(
        drug, patient_genotype, concomitant_meds
    )

    phenotype_table = entry.get("phenotypes", {})
    cell = phenotype_table.get(phenotype)
    if not cell:
        # Unknown phenotype for this drug — silently no-op.
        return (1.0, 1.0, notes)

    tox = float(cell.get("tox", 1.0))
    eff = float(cell.get("eff", 1.0))
    cell_note = cell.get("note")

    # Only record a note if adjustment is non-trivial (>5% deviation).
    if abs(tox - 1.0) > 0.05 or abs(eff - 1.0) > 0.05:
        msg = (
            f"pgx: {drug} {enzyme}={phenotype} → tox×{tox:.2f} eff×{eff:.2f}"
        )
        if cell_note:
            msg += f" ({cell_note})"
        notes.append(msg)

    return (eff, tox, notes)


def summarize_patient_pgx(patient_genotype) -> str:
    """One-line human summary for treatment_opt_agent sub-step 4a logs."""
    if patient_genotype is None:
        return "pgx: no profile (all enzymes default to normal_metabolizer)"
    if getattr(patient_genotype, "pgx_unavailable", True):
        return "pgx: no patient genotype on file → defaulting to normal_metabolizer"

    rows: list[str] = []
    for field in _GENOTYPE_FIELD_TO_ENZYME:
        val = getattr(patient_genotype, field, None)
        if isinstance(val, str) and val and val != _DEFAULT_PHENOTYPE:
            rows.append(f"{_GENOTYPE_FIELD_TO_ENZYME[field]}={val}")
    if not rows:
        return "pgx: all enzymes normal_metabolizer"
    return "pgx: " + ", ".join(rows)


__all__ = ["apply_pgx", "summarize_patient_pgx"]
