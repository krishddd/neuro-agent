"""Phase 5.1 — PyRadiomics feature extraction (Module 4).

Computes 5 radiomic features (texture/shape/intensity) from a NIfTI/DICOM
MRI volume and the nnU-Net / MONAI segmentation mask produced by Task 5:

    glcm_contrast             — texture heterogeneity
    glcm_correlation          — texture directionality
    shape_sphericity          — 0 = irregular, 1 = perfect sphere
    shape_surface_volume_ratio
    firstorder_entropy        — intensity chaos

Design notes
------------
* **Geometry-alignment trap (critical).** PyRadiomics raises
  ``ValueError: Image/Mask geometry mismatch`` when the MRI NIfTI and the
  segmentation mask don't share spacing/direction/origin. nnU-Net often
  resamples internally, so we resample the mask back onto the image grid
  with nearest-neighbour interpolation before extraction — see
  ``_align_mask_to_image``.

* **Graceful degrade.** Every failure path (missing pyradiomics,
  SimpleITK, missing mask, geometry mismatch beyond repair, extractor
  crash) yields a ``radiomics_unavailable=true`` payload rather than
  raising. The caller (``tools.mri_agent.extract_radiomics``) serialises
  that payload to ``S04d_radiomics.json`` and the Phase-4 patient-state
  builder imputes the 5 dims with population medians.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Ordered list of feature keys we emit. Keep stable — survival models and
# patient-state FEATURE_NAMES reference these by name.
RADIOMIC_FEATURE_NAMES: list[str] = [
    "glcm_contrast",
    "glcm_correlation",
    "shape_sphericity",
    "shape_surface_volume_ratio",
    "firstorder_entropy",
]

# ---------------------------------------------------------------------------
# Optional imports — each fails soft. All heavy work is lazy.
# ---------------------------------------------------------------------------
_sitk = None
_pyrad = None


def _lazy_sitk():
    global _sitk
    if _sitk is None:
        try:
            import SimpleITK as sitk  # type: ignore
            _sitk = sitk
        except Exception as exc:
            log.warning("radiomics: SimpleITK unavailable (%s)", type(exc).__name__)
            _sitk = False
    return _sitk or None


def _lazy_pyradiomics():
    global _pyrad
    if _pyrad is None:
        try:
            # PyRadiomics is chatty; quiet its logger before importing.
            import logging as _lg
            _lg.getLogger("radiomics").setLevel(_lg.WARNING)
            from radiomics import featureextractor  # type: ignore
            _pyrad = featureextractor
        except Exception as exc:
            log.warning("radiomics: pyradiomics unavailable (%s)", type(exc).__name__)
            _pyrad = False
    return _pyrad or None


# ---------------------------------------------------------------------------
# Unavailable payload (shared shape between error paths)
# ---------------------------------------------------------------------------
def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "radiomics_unavailable": True,
        "reason": reason,
        # All 5 features set to NaN-like None so callers can impute.
        **{k: None for k in RADIOMIC_FEATURE_NAMES},
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Geometry alignment (the trap the plan calls out)
# ---------------------------------------------------------------------------
def _geometry_close(a: tuple[float, ...], b: tuple[float, ...],
                    *, atol: float = 1e-6) -> bool:
    """Tolerant comparison for SimpleITK spacing / direction / origin tuples.

    NIfTI round-trips introduce float drift on the order of 1e-10..1e-7.
    Strict equality wrongly flags geometrically-identical images as
    needing a resample, which is wasteful and can introduce sub-voxel
    label dilation/erosion via the nearest-neighbour interpolator.
    """
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def _align_mask_to_image(image: Any, mask: Any) -> Any:
    """Resample ``mask`` onto ``image`` grid using nearest-neighbour.

    Returns the original ``mask`` object when spacing/direction/origin
    already match within float-drift tolerance (cheap no-op). On any
    failure, returns ``mask`` unchanged — the subsequent extractor call
    will surface the problem and we degrade gracefully from there.
    """
    sitk = _lazy_sitk()
    if sitk is None:
        return mask

    try:
        # BUG-FIX: use tolerance-based equality instead of strict tuple
        # compare. NIfTI I/O introduces ~1e-10 drift in spacing/origin
        # that previously triggered unnecessary resampling on every run.
        if (
            _geometry_close(tuple(mask.GetSpacing()),   tuple(image.GetSpacing()))
            and _geometry_close(tuple(mask.GetDirection()), tuple(image.GetDirection()))
            and _geometry_close(tuple(mask.GetOrigin()),    tuple(image.GetOrigin()))
            and mask.GetSize() == image.GetSize()
        ):
            return mask
        log.info(
            "radiomics: resampling mask to image grid "
            "(mask size=%s, image size=%s)",
            mask.GetSize(), image.GetSize(),
        )
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(image)
        r.SetInterpolator(sitk.sitkNearestNeighbor)  # preserve label ints
        r.SetDefaultPixelValue(0)
        return r.Execute(mask)
    except Exception as exc:
        log.warning(
            "radiomics: geometry alignment failed (%s) — will attempt "
            "extraction anyway and degrade on failure",
            type(exc).__name__,
        )
        return mask


def _adaptive_bin_width(image: Any, default: float = 25.0) -> float:
    """Compute a bin width appropriate to the image's intensity range.

    BUG-FIX (Phase 5.1): hard-coding ``binWidth=25`` works for raw MRI
    intensities (range ~0–5000) but degenerates to a single bin when
    the input has been normalised to [0, 1] or z-scored to ~[-3, 3].
    GLCM contrast/correlation then collapse to NaN. Aim for ~16–64
    bins of usable signal regardless of intensity scale.
    """
    sitk = _lazy_sitk()
    try:
        import numpy as _np  # type: ignore
    except Exception:
        return default
    if sitk is None:
        return default
    try:
        arr = sitk.GetArrayFromImage(image)
        finite = arr[_np.isfinite(arr)]
        if finite.size == 0:
            return default
        # Robust range — drop 1% tails so a single bright voxel doesn't
        # blow the bin count to one.
        lo = float(_np.quantile(finite, 0.01))
        hi = float(_np.quantile(finite, 0.99))
        rng = hi - lo
        if rng <= 0:
            return default
        # Target ~32 bins; floor at 1e-4 so normalised inputs still get
        # a sensible width rather than 0.
        proposed = max(rng / 32.0, 1e-4)
        # Cap at the default — we don't want fewer than ~16 bins on
        # raw-intensity MRI either.
        return min(proposed, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Feature selection — keep only the 5 we care about from PyRadiomics output
# ---------------------------------------------------------------------------
# PyRadiomics keys follow the pattern: ``original_<class>_<Name>``.
_RADIOMIC_KEY_MAP: dict[str, str] = {
    "glcm_contrast":              "original_glcm_Contrast",
    "glcm_correlation":           "original_glcm_Correlation",
    "shape_sphericity":           "original_shape_Sphericity",
    "shape_surface_volume_ratio": "original_shape_SurfaceVolumeRatio",
    "firstorder_entropy":         "original_firstorder_Entropy",
}


def _build_extractor(bin_width: float = 25.0):
    featureextractor = _lazy_pyradiomics()
    if featureextractor is None:
        return None

    # Minimal settings — we only want original image features in the 5
    # classes above, resampling turned off (we already aligned), and a
    # bin width sized to the actual intensity range (BUG-FIX #6).
    settings: dict[str, Any] = {
        "binWidth": float(bin_width),
        "resampledPixelSpacing": None,
        # Note: this interpolator is for *resampling* and is unused
        # because resampledPixelSpacing=None. Kept to silence a default-
        # values warning from PyRadiomics.
        "interpolator": "sitkBSpline",
        "verbose": False,
        "label": 1,  # extract on label==1 (tumour) — callers must remap
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # Disable all features then re-enable the three classes we use.
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    return extractor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_radiomic_features(
    image_path: str | Path,
    mask_path: str | Path | None,
    label: int = 1,
) -> dict[str, Any]:
    """Compute the 5 radiomic features from ``image_path`` and ``mask_path``.

    Parameters
    ----------
    image_path:
        Path to the source MRI volume (NIfTI .nii / .nii.gz, or a
        SimpleITK-readable DICOM series root).
    mask_path:
        Path to the binary / labelled segmentation mask in the same
        modality. If ``None`` (e.g. volumetric seg was unavailable),
        returns a ``radiomics_unavailable`` payload.
    label:
        Label value to extract on (default 1 — the whole tumour mask as
        produced by our MONAI heuristic / nnU-Net binary writer).

    Returns
    -------
    dict with either the 5 feature keys + ``radiomics_unavailable=False``
    or ``radiomics_unavailable=True`` + ``reason``.
    """
    if mask_path is None:
        return _unavailable("no_mask_available")

    image_path = Path(image_path)
    mask_path = Path(mask_path)
    if not image_path.exists():
        return _unavailable("image_not_found", image_path=str(image_path))
    if not mask_path.exists():
        return _unavailable("mask_not_found", mask_path=str(mask_path))

    sitk = _lazy_sitk()
    if sitk is None:
        return _unavailable("simpleitk_unavailable")

    try:
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
    except Exception as exc:
        return _unavailable(
            f"volume_read_failed:{type(exc).__name__}",
            error=str(exc)[:200],
        )

    # Guard against empty masks — PyRadiomics will complain with a
    # confusing "Label (1) not present in mask" otherwise.
    try:
        arr = sitk.GetArrayFromImage(mask)
        if arr.size == 0 or (arr == label).sum() == 0:
            return _unavailable(
                "empty_mask",
                label=label,
                voxels_in_label=int((arr == label).sum()) if arr.size else 0,
            )
    except Exception:
        # Non-fatal — let the extractor surface the real error.
        pass

    mask = _align_mask_to_image(image, mask)

    # BUG-FIX #7 (Phase 5.1): re-check empty mask AFTER alignment.
    # When the mask and image have very different fields-of-view, the
    # nearest-neighbour resample can produce an all-zero result even
    # though the pre-alignment mask had voxels. Catch that here so the
    # extractor doesn't surface a confusing "Label not present" error.
    try:
        aligned_arr = sitk.GetArrayFromImage(mask)
        n_label_voxels = int((aligned_arr == label).sum())
        if n_label_voxels == 0:
            return _unavailable(
                "empty_mask_after_alignment",
                label=label,
                voxels_in_label=0,
                hint="mask + image FOVs likely disjoint — check segmentation source",
            )
    except Exception:
        pass

    # BUG-FIX #6 (Phase 5.1): build the extractor with a bin-width sized
    # to the actual image intensity range. Hard-coding 25 produced a
    # single GLCM bin for normalised inputs ([0,1] or z-scored), which
    # made glcm_contrast/correlation degenerate.
    bin_width = _adaptive_bin_width(image)
    extractor = _build_extractor(bin_width=bin_width)
    if extractor is None:
        return _unavailable("pyradiomics_unavailable")

    try:
        raw = extractor.execute(image, mask, label=label)
    except Exception as exc:
        return _unavailable(
            f"extractor_failed:{type(exc).__name__}",
            error=str(exc)[:200],
        )

    # Project to our 5 canonical features; cast numpy scalars → float.
    out: dict[str, Any] = {"radiomics_unavailable": False}
    missing: list[str] = []
    for canonical, pyrad_key in _RADIOMIC_KEY_MAP.items():
        val = raw.get(pyrad_key)
        if val is None:
            missing.append(canonical)
            out[canonical] = None
            continue
        try:
            out[canonical] = float(val)
        except Exception:
            out[canonical] = None
            missing.append(canonical)

    if missing:
        out["partial"] = True
        out["missing_features"] = missing
    out["label"] = label
    out["image_path"] = str(image_path)
    out["mask_path"]  = str(mask_path)
    out["bin_width"]  = float(bin_width)   # audit trail for Bug-Fix #6
    return out


__all__ = [
    "RADIOMIC_FEATURE_NAMES",
    "extract_radiomic_features",
]
