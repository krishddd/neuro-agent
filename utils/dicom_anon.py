"""DICOM loading, anonymization, and PNG export for vision models.

We strip identifying tags before the pixel data ever touches an LLM
prompt, then export a windowed 8-bit grayscale PNG the vision model can read.

Dataset reality (P001–P020)
---------------------------
The synthetic .dcm files in this dataset have a malformed outer wrapper:
the first 148 bytes are a corrupt file-meta blob.  The real DICOM stream
(with a proper 0002-group file-meta preamble) begins at byte offset 148.
`load_dicom()` detects this and slices the file before handing it to
pydicom, so the standard anonymization + windowing pipeline works normally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import logging as _early_log
_log = _early_log.getLogger(__name__)

# Track which compressed-DICOM handlers managed to import. Surfaced at
# server startup AND in the failure path of ``dicom_to_png`` so the
# operator gets one clear actionable message rather than a generic
# "Cannot extract pixel data" trace.
_DICOM_HANDLERS_AVAILABLE: dict[str, bool] = {
    "gdcm":      False,
    "pylibjpeg": False,
    "pillow":    False,
}

try:
    import pydicom
    import pydicom.config
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    pydicom.config.convert_wrong_length_to_UN = True   # tolerate length errors

    # Register pixel-data handlers so pixel_array works on compressed DICOM.
    # Each handler is optional — import silently if missing but mark the
    # availability flag so we can give a precise error later.
    try:
        from pydicom.pixel_data_handlers import gdcm_handler
        pydicom.config.pixel_data_handlers.insert(0, gdcm_handler)
        _DICOM_HANDLERS_AVAILABLE["gdcm"] = True
    except Exception:
        pass
    try:
        from pydicom.pixel_data_handlers import pylibjpeg_handler
        pydicom.config.pixel_data_handlers.insert(0, pylibjpeg_handler)
        _DICOM_HANDLERS_AVAILABLE["pylibjpeg"] = True
    except Exception:
        pass
    try:
        from pydicom.pixel_data_handlers import pillow_handler
        pydicom.config.pixel_data_handlers.insert(0, pillow_handler)
        _DICOM_HANDLERS_AVAILABLE["pillow"] = True
    except Exception:
        pass

except Exception:  # pragma: no cover
    pydicom = None  # type: ignore
    FileDataset = Any  # type: ignore


def dicom_handler_summary() -> str:
    """Human-readable summary of which DICOM decompression libs loaded.

    Logged once at server startup (see ``api/app.py``) and embedded in
    the failure path of ``dicom_to_png`` so a "Cannot extract pixel data"
    error tells the operator exactly which package to install.
    """
    if pydicom is None:
        return (
            "pydicom NOT installed — every DICOM read will fail. "
            "Run: pip install pydicom pylibjpeg pylibjpeg-libjpeg "
            "pylibjpeg-openjpeg python-gdcm"
        )
    loaded = [k for k, v in _DICOM_HANDLERS_AVAILABLE.items() if v]
    missing = [k for k, v in _DICOM_HANDLERS_AVAILABLE.items() if not v]
    if not loaded:
        return (
            "pydicom OK but NO compressed-DICOM handlers loaded — "
            "JPEG/JPEG-LS/JPEG2000/RLE-encoded files will fail with "
            "'Cannot extract pixel data'. "
            "Run: pip install pylibjpeg pylibjpeg-libjpeg "
            "pylibjpeg-openjpeg python-gdcm"
        )
    msg = f"DICOM handlers loaded: {', '.join(loaded)}"
    if missing:
        hint_pkgs = {
            "gdcm":      "python-gdcm  (RLE + JPEG-LS)",
            "pylibjpeg": "pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg  (JPEG / JPEG2000)",
            "pillow":    "pillow  (uncompressed fallback — usually already present)",
        }
        miss_hint = "; ".join(hint_pkgs.get(m, m) for m in missing)
        msg += f"  (missing: {miss_hint})"
    return msg


def log_dicom_dependency_status() -> None:
    """Emit a single startup banner with current DICOM-decoding capability.

    Called from ``api/app.py`` at boot. Promotes to WARNING when nothing
    compressed can be decoded so the operator sees it in any log level.
    """
    summary = dicom_handler_summary()
    if pydicom is None or not any(_DICOM_HANDLERS_AVAILABLE.values()):
        _log.warning("dicom: %s", summary)
    else:
        _log.info("dicom: %s", summary)

# Standard DICOM preamble length + "DICM" magic = 132 bytes.
_PREAMBLE = b"\x00" * 128 + b"DICM"
# File-meta group tag (0002,0001) in little-endian — marks real DICOM header.
_META_MARKER = b"\x02\x00\x01\x00"


# Tags to blank before any downstream use.
_PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "ReferringPhysicianName",
    "PerformingPhysicianName", "OperatorsName", "InstitutionName",
    "InstitutionAddress", "StationName", "AccessionNumber",
    "StudyID", "IssuerOfPatientID",
]


def load_dicom(path: str | Path) -> "FileDataset":
    """Load a DICOM file, handling the malformed-outer-wrapper dataset pattern.

    Strategy:
    1. Read raw bytes.
    2. If the file starts with the proper 128+DICM preamble → read normally.
    3. Otherwise, search for the first occurrence of the (0002,0001) tag that
       marks the start of the embedded real DICOM file-meta and splice from
       there, prepending a correct 128+DICM preamble so pydicom can parse it.
    4. Fall back to force=True on the original if splicing also fails.
    """
    if pydicom is None:
        raise RuntimeError("pydicom not installed — pip install pydicom")

    raw = Path(path).read_bytes()

    # Case 1: already has a correct DICOM preamble.
    if raw[:4] == b"\x00\x00\x00\x00" and raw[128:132] == b"DICM":
        return _read_bytes(raw)

    # Case 2: locate the embedded file-meta start marker (0002,0001).
    offset = raw.find(_META_MARKER)
    if offset > 0:
        embedded = _PREAMBLE + raw[offset:]
        try:
            return _read_bytes(embedded)
        except Exception:
            pass  # fall through to case 3

    # Case 3: brute-force with pydicom force=True on original.
    return pydicom.dcmread(str(path), force=True)


def _read_bytes(data: bytes) -> "FileDataset":
    """Parse raw bytes (with a valid DICM preamble) into a pydicom Dataset."""
    import io as _io
    buf = _io.BytesIO(data)
    ds = pydicom.dcmread(buf, force=True)
    # Ensure transfer syntax is set so pixel_array works.
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.Dataset()
    if not hasattr(ds.file_meta, "TransferSyntaxUID"):
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return ds


def anonymize(ds: "FileDataset") -> "FileDataset":
    for tag in _PHI_TAGS:
        if tag in ds:
            try:
                setattr(ds, tag, "")
            except Exception:
                pass
    return ds


def _window(pixels: np.ndarray, ds: "FileDataset") -> np.ndarray:
    """Apply Window Center/Width if present, else min-max normalize.

    Always returns uint8 (0–255) suitable for PIL mode='L' or 'RGB'.
    """
    arr = pixels.astype(np.float32)

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    # DICOM can store these as pydicom DSfloat sequences.
    if isinstance(wc, (list, tuple, pydicom.multival.MultiValue if pydicom else list)):
        wc = float(wc[0])
    if isinstance(ww, (list, tuple, pydicom.multival.MultiValue if pydicom else list)):
        ww = float(ww[0])
    if wc is not None and ww is not None:
        try:
            wc, ww = float(wc), float(ww)
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
        except (TypeError, ValueError):
            wc = ww = None

    if wc is None or ww is None:
        lo, hi = float(arr.min()), float(arr.max())

    if hi <= lo:
        hi = lo + 1.0

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def dicom_to_png(path: str | Path, out_path: str | Path) -> Path:
    """Anonymize + window + export a single-slice DICOM as 8-bit PNG.

    Returns the output path. Multi-frame DICOMs export their middle slice.
    Raises RuntimeError if pixel data cannot be extracted after all fallbacks.
    """
    ds = anonymize(load_dicom(path))

    # Extract pixel array — pixel_array may fail on some synthetic files.
    try:
        pixels = ds.pixel_array
    except Exception:
        # Last-resort: parse raw PixelData bytes manually.
        pixels = _extract_pixels_manual(ds)

    if pixels is None:
        # Surface a precise install hint instead of a bare "Cannot extract
        # pixel data" — most often this is a compressed-transfer-syntax
        # DICOM (JPEG / JPEG-LS / JPEG2000 / RLE) and the user just needs
        # to install the right plugin.
        ts = ""
        try:
            ts = str(getattr(ds.file_meta, "TransferSyntaxUID", "") or "")
        except Exception:
            pass
        hint = dicom_handler_summary()
        raise RuntimeError(
            f"Cannot extract pixel data from {path}"
            + (f"  [transfer_syntax={ts}]" if ts else "")
            + f"  [{hint}]"
        )

    if pixels.ndim == 3:
        pixels = pixels[pixels.shape[0] // 2]  # take middle slice of multi-frame

    img_u8 = _window(pixels, ds)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(out, format="PNG")
    return out


def _extract_pixels_manual(ds: "FileDataset") -> np.ndarray | None:
    """Manual fallback: read raw PixelData bytes and reshape using image dimensions."""
    px_data = getattr(ds, "PixelData", None)
    if px_data is None:
        return None
    rows = int(ds.get("Rows") or 0)
    cols = int(ds.get("Columns") or 0)
    bits = int(ds.get("BitsAllocated") or 16)
    if rows <= 0 or cols <= 0:
        # Try to infer from data size.
        n_bytes = len(px_data)
        side = int(np.sqrt(n_bytes / (bits // 8)))
        if side * side * (bits // 8) == n_bytes:
            rows = cols = side
        else:
            return None
    dtype = np.uint16 if bits == 16 else np.uint8
    try:
        arr = np.frombuffer(bytes(px_data), dtype=dtype).reshape(rows, cols)
        return arr
    except Exception:
        return None
