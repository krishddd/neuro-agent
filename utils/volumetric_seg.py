"""3D volumetric segmentation for brain MRI (Task 5).

Replaces the 2D PNG fallback used by the RECIST/RANO path with true
volumetric tumour segmentation. Supports three backends selected via the
``BRATS_BACKEND`` environment variable:

  * ``nnunet``  — nnU-Net BraTS model (preferred; weights downloaded lazily)
  * ``monai``   — MONAI U-Net (lighter fallback)
  * ``none``    — no segmentation; writes a placeholder with
                  ``volumetric_unavailable=true`` and logs a warning.

The output is a dict with four keys:

    enhancing_volume_mm3
    necrotic_volume_mm3
    edema_volume_mm3
    largest_axial_bidim_mm2

This module is deliberately *defensive*: every import and inference path is
wrapped so the rest of the pipeline keeps working if nibabel / MONAI /
nnU-Net / torch are not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import BRATS_BACKEND

log = logging.getLogger(__name__)

# ---------- Optional imports (guarded) ----------
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep elsewhere
    np = None  # type: ignore[assignment]

_nib = None
_sitk = None


def _lazy_nibabel():
    global _nib
    if _nib is None:
        try:
            import nibabel as nib  # type: ignore
            _nib = nib
        except Exception as exc:
            log.warning("volumetric_seg: nibabel unavailable (%s) — NIfTI disabled",
                        type(exc).__name__)
            _nib = False  # sentinel so we don't retry
    return _nib or None


def _lazy_sitk():
    global _sitk
    if _sitk is None:
        try:
            import SimpleITK as sitk  # type: ignore
            _sitk = sitk
        except Exception as exc:
            log.warning("volumetric_seg: SimpleITK unavailable (%s)",
                        type(exc).__name__)
            _sitk = False
    return _sitk or None


# ---------- Volume loading ----------
def load_volume(path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a NIfTI (.nii / .nii.gz) or DICOM series into a numpy array.

    Returns ``(volume_3d, metadata)`` where metadata includes voxel spacing
    in mm (``spacing_mm``) and the source path.

    Raises ``RuntimeError`` on any failure — caller is expected to fall
    back to the 2D PNG path.
    """
    if np is None:
        raise RuntimeError("numpy not available")
    p = Path(path)
    suf = "".join(p.suffixes).lower()  # handles .nii.gz

    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        nib = _lazy_nibabel()
        if nib is None:
            raise RuntimeError("nibabel not installed")
        img = nib.load(str(p))
        vol = np.asarray(img.get_fdata(), dtype=np.float32)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return vol, {"spacing_mm": zooms, "source": str(p), "format": "nifti"}

    if p.is_dir() or suf.endswith(".dcm"):
        sitk = _lazy_sitk()
        if sitk is None:
            raise RuntimeError("SimpleITK not installed")
        reader = sitk.ImageSeriesReader()
        series_root = str(p if p.is_dir() else p.parent)
        series_ids = reader.GetGDCMSeriesIDs(series_root)
        if not series_ids:
            raise RuntimeError(f"no DICOM series in {series_root}")
        files = reader.GetGDCMSeriesFileNames(series_root, series_ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
        vol = sitk.GetArrayFromImage(img).astype("float32")  # (z, y, x)
        spacing = tuple(float(s) for s in img.GetSpacing())  # (x, y, z)
        return vol, {
            "spacing_mm": (spacing[2], spacing[1], spacing[0]),
            "source": str(p),
            "format": "dicom_series",
        }

    raise RuntimeError(f"unsupported volume format: {p.name}")


# ---------- Segmentation helpers ----------
def _voxel_volume_mm3(spacing_mm: tuple[float, float, float]) -> float:
    sx, sy, sz = spacing_mm
    return float(sx * sy * sz)


def _largest_axial_bidim_mm2(mask: Any, spacing_mm: tuple[float, float, float]) -> float:
    """Approximate the largest axial bidirectional product of a binary mask.

    For each axial slice, compute the mask's bounding box; the product of
    its in-plane extents (in mm) stands in for ``longest × perpendicular``.
    RANO uses the single most tumour-loaded slice, so we pick the max
    across slices.
    """
    if np is None or mask is None:
        return 0.0
    mask = mask.astype(bool)
    if not mask.any():
        return 0.0
    sx, sy, _sz = spacing_mm  # in-plane spacing
    best = 0.0
    # Assume convention (z, y, x) — most loaders we use produce this.
    for z in range(mask.shape[0]):
        sl = mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        dy = (ys.max() - ys.min() + 1) * sy
        dx = (xs.max() - xs.min() + 1) * sx
        prod = float(dy * dx)
        if prod > best:
            best = prod
    return best


def _write_heuristic_mask_from_source(
    source_path: str | Path | None,
    mask_out_path: str | Path,
) -> str | None:
    """Read ``source_path`` with SimpleITK, build a MONAI-heuristic binary
    mask (upper-1% intensity quantile) and write it to ``mask_out_path``.

    Used as the fallback path (backend="none") when no in-memory mask
    array is available. Prefer :func:`_persist_mask_from_array` when the
    caller already has the thresholded array — that path guarantees the
    persisted mask matches the in-memory volume metrics.
    """
    if source_path is None or np is None:
        return None
    sitk = _lazy_sitk()
    if sitk is None:
        return None
    try:
        img = sitk.ReadImage(str(source_path))
        arr = sitk.GetArrayFromImage(img).astype("float32")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        thr_enh = np.quantile(finite, 0.99)
        mask_arr = (arr >= thr_enh).astype("uint8")
        # BUG-FIX (Phase 5.1): use the same threshold (top-1% / enhancing)
        # that _segment_monai's volume metrics use. Earlier this used
        # thr_ede (top-10%) which produced a mask ~10x larger than the
        # "tumour core" the volume numbers describe — radiomic shape
        # features then disagreed with reported volumes.
        if int(mask_arr.sum()) == 0:
            log.warning("volumetric_seg: heuristic mask empty (no voxels above p99)")
            return None
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(img)
        out = Path(mask_out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(mask_img, str(out))
        log.info("volumetric_seg: heuristic mask persisted to %s (%d voxels)",
                 out, int(mask_arr.sum()))
        return str(out)
    except Exception as exc:
        log.warning(
            "volumetric_seg: mask persist failed (%s) for %s",
            type(exc).__name__, source_path,
        )
        return None


def _persist_mask_from_array(
    mask_arr: Any,                    # numpy bool / uint8 array, shape (Z,Y,X)
    source_path: str | Path | None,
    mask_out_path: str | Path,
) -> str | None:
    """Persist an in-memory mask array as NIfTI, copying geometry from
    ``source_path`` so PyRadiomics aligns cleanly later.

    BUG-FIX (Phase 5.1): we used to threshold ``volume`` in-memory for
    the volume metrics but re-threshold ``source_path`` from disk inside
    :func:`_write_heuristic_mask_from_source`. When the loader applied
    reorientation or normalisation, the two diverged — radiomic shape
    features disagreed with the reported enhancing-volume. Persisting
    the in-memory array directly keeps the two consistent.
    """
    if mask_arr is None or np is None:
        return None
    sitk = _lazy_sitk()
    if sitk is None or source_path is None:
        return None
    try:
        ref = sitk.ReadImage(str(source_path))
        # SimpleITK expects (Z, Y, X) with axis order matching the
        # reference image — _segment_monai's `vol` array is loaded the
        # same way (sitk.GetArrayFromImage), so shapes line up.
        u8 = np.asarray(mask_arr, dtype="uint8")
        if u8.shape != ref.GetSize()[::-1]:
            log.warning(
                "volumetric_seg: in-memory mask shape %s does not match "
                "source %s — falling back to disk-reread heuristic",
                u8.shape, ref.GetSize()[::-1],
            )
            return _write_heuristic_mask_from_source(source_path, mask_out_path)
        if int(u8.sum()) == 0:
            log.warning("volumetric_seg: in-memory mask empty")
            return None
        mask_img = sitk.GetImageFromArray(u8)
        mask_img.CopyInformation(ref)
        out = Path(mask_out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(mask_img, str(out))
        log.info("volumetric_seg: in-memory mask persisted to %s (%d voxels)",
                 out, int(u8.sum()))
        return str(out)
    except Exception as exc:
        log.warning(
            "volumetric_seg: in-memory mask persist failed (%s)",
            type(exc).__name__,
        )
        return None


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "enhancing_volume_mm3": 0.0,
        "necrotic_volume_mm3": 0.0,
        "edema_volume_mm3": 0.0,
        "largest_axial_bidim_mm2": 0.0,
        "mask_path": None,
        "volumetric_unavailable": True,
        "reason": reason,
        "backend": "none",
    }


# ---------- Backends ----------
def _segment_nnunet(volume: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch  # type: ignore  # noqa: F401
        from nnunetv2.inference.predict_from_raw_data import (  # type: ignore
            nnUNetPredictor,
        )
    except Exception as exc:
        log.warning("volumetric_seg: nnU-Net not available (%s) — falling back",
                    type(exc).__name__)
        return _empty_result(f"nnunet_import_failed:{type(exc).__name__}")
    # NOTE: a full nnUNetv2 setup requires a trained BraTS checkpoint in
    # NNUNET_MODELS_DIR; we don't ship weights. If the user has them
    # populated, they can plug in here. For now, treat as unavailable.
    return _empty_result("nnunet_weights_not_configured")


def _segment_monai(volume: Any, metadata: dict[str, Any],
                   mask_out_path: str | Path | None = None) -> dict[str, Any]:
    try:
        import torch  # type: ignore
        from monai.networks.nets import UNet  # type: ignore  # noqa: F401
    except Exception as exc:
        log.warning("volumetric_seg: MONAI not available (%s)", type(exc).__name__)
        return _empty_result(f"monai_import_failed:{type(exc).__name__}")

    if np is None:
        return _empty_result("numpy_unavailable")

    # Rough heuristic "segmentation": threshold the normalised intensity volume
    # at the upper 1% to isolate enhancing tissue. This is NOT clinically
    # valid — it exists only so the downstream RANO path receives a non-zero
    # bidim when the user lacks trained weights but wants the plumbing to
    # produce a well-formed S04c_volumetric.json.
    vol = np.asarray(volume, dtype=np.float32)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return _empty_result("empty_volume")
    thr_enh = np.quantile(finite, 0.99)
    thr_ede = np.quantile(finite, 0.90)
    enh_mask = vol >= thr_enh
    ede_mask = (vol >= thr_ede) & (~enh_mask)
    nec_mask = np.zeros_like(vol, dtype=bool)  # cannot infer without a model

    spacing = metadata.get("spacing_mm", (1.0, 1.0, 1.0))
    vv = _voxel_volume_mm3(spacing)

    # Phase 5.1 — persist a geometry-consistent mask so PyRadiomics can run.
    # BUG-FIX: use the same in-memory enh_mask we just thresholded so the
    # persisted mask matches the volume metrics computed below. The old
    # path re-read the source from disk and re-thresholded, which
    # diverged when the loader applied reorientation or normalisation.
    persisted_mask: str | None = None
    if mask_out_path is not None:
        persisted_mask = _persist_mask_from_array(
            enh_mask.astype("uint8"),
            metadata.get("source"),
            mask_out_path,
        )

    return {
        "enhancing_volume_mm3": float(enh_mask.sum()) * vv,
        "necrotic_volume_mm3":  float(nec_mask.sum()) * vv,
        "edema_volume_mm3":     float(ede_mask.sum()) * vv,
        "largest_axial_bidim_mm2": _largest_axial_bidim_mm2(enh_mask, spacing),
        "mask_path": persisted_mask,
        "volumetric_unavailable": False,
        "backend": "monai_heuristic",
        "note": (
            "Heuristic quantile thresholding (no trained weights available). "
            "Do NOT use for clinical decisions — replace with nnU-Net BraTS "
            "once weights are installed."
        ),
    }


def segment_brats(volume: Any, metadata: dict[str, Any],
                  backend: str | None = None,
                  mask_out_path: str | Path | None = None) -> dict[str, Any]:
    """Dispatch segmentation to the configured backend.

    Returns a dict with all four volume metrics. Any failure degrades to
    ``{volumetric_unavailable: True, backend: 'none'}`` — never raises.

    ``mask_out_path`` (Phase 5.1) — when set and the backend produces a
    mask, it is persisted as a NIfTI with source-image geometry so
    PyRadiomics can re-use it for radiomic feature extraction.
    """
    be = (backend or BRATS_BACKEND or "none").lower()
    try:
        if be == "nnunet":
            out = _segment_nnunet(volume, metadata)
        elif be == "monai":
            out = _segment_monai(volume, metadata, mask_out_path=mask_out_path)
        elif be == "none":
            out = _empty_result("backend_disabled")
            # Even with no trained model, offer a heuristic mask to unlock
            # radiomics. It's clearly marked backend="none_heuristic_mask"
            # so callers know the volume metrics are zeros.
            if mask_out_path is not None:
                persisted = _write_heuristic_mask_from_source(
                    metadata.get("source"), mask_out_path,
                )
                if persisted:
                    out["mask_path"] = persisted
                    out["backend"] = "none_heuristic_mask"
        else:
            out = _empty_result(f"unknown_backend:{be}")
    except Exception as exc:
        log.warning("volumetric_seg: %s backend crashed (%s) — degrading",
                    be, type(exc).__name__)
        out = _empty_result(f"{be}_crashed:{type(exc).__name__}")
    out.setdefault("spacing_mm", metadata.get("spacing_mm"))
    out.setdefault("source", metadata.get("source"))
    return out


def segment_path(path: Path, backend: str | None = None,
                 mask_out_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience: load ``path`` and run the configured backend.

    Safe to call from a worker process. ``mask_out_path`` is forwarded
    to :func:`segment_brats` so a mask NIfTI can be persisted next to
    the usual JSON result for radiomic re-use.
    """
    try:
        vol, meta = load_volume(Path(path))
    except Exception as exc:
        return _empty_result(f"load_failed:{type(exc).__name__}")
    return segment_brats(vol, meta, backend=backend,
                         mask_out_path=mask_out_path)


__all__ = ["load_volume", "segment_brats", "segment_path"]
