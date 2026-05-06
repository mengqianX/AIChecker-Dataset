from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import ImageFilter

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import button_base_color, load_image


DEFAULT_PIXEL_THRESHOLD = 8
DEFAULT_CHANGE_RATIO_THRESHOLD = 0.02
DEFAULT_PROFILE_DIFF_THRESHOLD = 0.015
DEFAULT_EDGE_SHIFT_THRESHOLD_PX = 2.0

# Gaussian blur radius applied before computing change_ratio to suppress
# image-quality perturbations (invisible noise, JPEG artifacts, mild
# resize/crop blur) that scatter across individual pixels but leave the
# overall column profile and bar edge unchanged.
_ROBUST_BLUR_RADIUS = 1.5

# Minimum gradient peak height (normalized 0-1) required for the argmax-based
# edge position to be considered reliable.
_MIN_EDGE_GRADIENT = 0.01

# Minimum maximum-column-diff to count as "structural evidence".
# Real visible changes (bar fill, text timestamp, edge movement) produce at
# least one column whose row-averaged diff is clearly above noise level.
# After Gaussian blur the column average for noise rarely exceeds ~0.03;
# a threshold of 0.08 (≈ 20/255) reliably separates structural signal from
# noise/compression artifacts across a wide range of bar heights and styles.
_MAX_COL_DIFF_STRUCTURAL = 0.08

# Single-metric "clearly large" bypass thresholds (skip all other checks).
_CLEARLY_CHANGED_RATIO_FACTOR  = 5   # change_ratio_robust >= 5 × threshold
_CLEARLY_CHANGED_PROFILE_FACTOR = 3  # profile_diff         >= 3 × threshold

# Ratio-dominant fallback for subtle-but-real progress updates:
# some real cases have strong per-pixel movement but muted column-profile mean
# (e.g. thin/soft edges, anti-aliased progress fronts). Require both:
#   1) clearly elevated change_ratio_robust
#   2) non-trivial profile_diff support (not pure random noise)
_RATIO_DOMINANT_FACTOR = 2.5
_PROFILE_SUPPORT_FACTOR = 0.70


def _resolve_expected_change(payload: Dict[str, Any]) -> bool:
    raw = payload.get("expected_change")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "change", "changed"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "unchanged", "no_change"}:
        return False
    raise ValueError(f"Unsupported expected_change value: {raw!r}")


def _resolve_control_text_and_semantics(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    target_control = payload.get("target_control")
    if isinstance(target_control, dict):
        text = target_control.get("text")
        semantics = target_control.get("semantics")
    else:
        text = payload.get("text")
        semantics = payload.get("semantics")

    text_value = str(text).strip() if text is not None else ""
    semantics_value = str(semantics).strip() if semantics is not None else ""
    text_out = text_value if text_value else None
    semantics_out = semantics_value if semantics_value else None
    if text_out is None and semantics_out is None:
        semantics_out = "progress_bar"
    return text_out, semantics_out


def _estimate_progress_edge_shift_px(before_gray: np.ndarray, after_gray: np.ndarray) -> float:
    # Estimate bar-front movement by comparing strongest column gradients.
    profile_before = before_gray.mean(axis=0)
    profile_after = after_gray.mean(axis=0)

    grad_before = np.abs(np.diff(profile_before))
    grad_after = np.abs(np.diff(profile_after))
    if grad_before.size == 0 or grad_after.size == 0:
        return 0.0

    # Guard: when the profile is nearly flat (no clear bar boundary), the
    # argmax position is unreliable — a tiny noise difference between two
    # otherwise identical frames can flip it by hundreds of pixels.  Only
    # report edge shift when both profiles have a sufficiently prominent peak.
    if grad_before.max() < _MIN_EDGE_GRADIENT or grad_after.max() < _MIN_EDGE_GRADIENT:
        return 0.0

    edge_before = int(np.argmax(grad_before))
    edge_after = int(np.argmax(grad_after))
    return float(abs(edge_after - edge_before))


def _detect_progress_changed(
    *,
    change_ratio_robust: float,
    profile_diff: float,
    max_col_diff: float,
    edge_shift_px: float,
    change_ratio_threshold: float,
    profile_diff_threshold: float,
    edge_shift_threshold_px: float,
) -> Dict[str, bool]:
    # PATH 1 — clearly large single metric (unambiguous, no further checks)
    clearly_changed = (
        change_ratio_robust >= change_ratio_threshold * _CLEARLY_CHANGED_RATIO_FACTOR
        or profile_diff >= profile_diff_threshold * _CLEARLY_CHANGED_PROFILE_FACTOR
    )

    # PATH 2 — both metrics simultaneously above their thresholds.
    both_above_threshold = (
        change_ratio_robust >= change_ratio_threshold
        and profile_diff >= profile_diff_threshold
    )

    # PATH 3 — change_ratio + structural column evidence.
    change_ratio_with_structure = (
        change_ratio_robust >= change_ratio_threshold
        and max_col_diff >= _MAX_COL_DIFF_STRUCTURAL
    )

    # PATH 4 — ratio-dominant + profile support.
    ratio_dominant_with_profile_support = (
        change_ratio_robust >= change_ratio_threshold * _RATIO_DOMINANT_FACTOR
        and profile_diff >= profile_diff_threshold * _PROFILE_SUPPORT_FACTOR
    )

    # PATH 5 — edge shift + structural column evidence.
    edge_shift_triggered = (
        edge_shift_px >= edge_shift_threshold_px
        and max_col_diff >= _MAX_COL_DIFF_STRUCTURAL
    )

    detected_changed = (
        clearly_changed
        or both_above_threshold
        or change_ratio_with_structure
        or ratio_dominant_with_profile_support
        or edge_shift_triggered
    )
    return {
        "detected_changed": detected_changed,
        "clearly_changed": clearly_changed,
        "both_above_threshold": both_above_threshold,
        "change_ratio_with_structure": change_ratio_with_structure,
        "ratio_dominant_with_profile_support": ratio_dominant_with_profile_support,
        "edge_shift_triggered": edge_shift_triggered,
    }


def check_progress_change(
    payload: Dict[str, Any],
    debug_dir: Path | None = None,
) -> CheckResult:
    """
    Check whether the progress bar region changes between two screenshots.

    Expected payload fields:
    - screenshot_a: before image path
    - screenshot_b: after image path
    - bounds: [left, top, right, bottom]
    Optional:
    - expected_change: bool, default True
    - pixel_threshold / change_ratio_threshold / profile_diff_threshold / edge_shift_threshold_px
    - text / semantics / target_control
    """
    screenshot_a = payload.get("screenshot_a")
    screenshot_b = payload.get("screenshot_b") or payload.get("screenshot")
    if not screenshot_a or not screenshot_b:
        raise ValueError("Both screenshot_a and screenshot_b (or screenshot) must be provided")

    bounds = Bounds.from_sequence(payload["bounds"])
    expected_change = _resolve_expected_change(payload)
    pixel_threshold = int(payload.get("pixel_threshold", DEFAULT_PIXEL_THRESHOLD))
    change_ratio_threshold = float(
        payload.get("change_ratio_threshold", DEFAULT_CHANGE_RATIO_THRESHOLD)
    )
    profile_diff_threshold = float(
        payload.get("profile_diff_threshold", DEFAULT_PROFILE_DIFF_THRESHOLD)
    )
    edge_shift_threshold_px = float(
        payload.get("edge_shift_threshold_px", DEFAULT_EDGE_SHIFT_THRESHOLD_PX)
    )

    img_before = load_image(str(screenshot_a))
    img_after = load_image(str(screenshot_b))
    crop_before = img_before.crop(bounds.as_box()).convert("RGB")
    crop_after = img_after.crop(bounds.as_box()).convert("RGB")

    if crop_before.size != crop_after.size:
        raise ValueError(
            f"Before/after crop size mismatch: before={crop_before.size}, after={crop_after.size}"
        )

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        crop_before.save(debug_dir / "progress_crop_before.png")
        crop_after.save(debug_dir / "progress_crop_after.png")

    # ── Raw per-pixel diff (kept for diagnostics) ──────────────────────────
    before_arr = np.array(crop_before, dtype=np.int16)
    after_arr = np.array(crop_after, dtype=np.int16)
    abs_diff = np.abs(after_arr - before_arr)
    max_per_pixel = abs_diff.max(axis=2)

    change_ratio = float((max_per_pixel >= pixel_threshold).mean())
    mean_abs_pixel_diff = float(max_per_pixel.mean())

    # ── Robust per-pixel diff (Gaussian-blurred to suppress noise) ─────────
    # Randomly scattered perturbations (invisible noise, JPEG/resize artifacts)
    # average out under the blur, while real bar edge movements — which affect
    # entire consistent columns — survive.
    crop_before_blurred = crop_before.filter(ImageFilter.GaussianBlur(radius=_ROBUST_BLUR_RADIUS))
    crop_after_blurred  = crop_after.filter(ImageFilter.GaussianBlur(radius=_ROBUST_BLUR_RADIUS))
    before_blur_arr = np.array(crop_before_blurred, dtype=np.int16)
    after_blur_arr  = np.array(crop_after_blurred,  dtype=np.int16)
    abs_diff_robust = np.abs(after_blur_arr - before_blur_arr)
    max_per_pixel_robust = abs_diff_robust.max(axis=2)
    change_ratio_robust = float((max_per_pixel_robust >= pixel_threshold).mean())

    # ── Structural / profile metrics ───────────────────────────────────────
    before_gray = np.array(crop_before.convert("L"), dtype=np.float32) / 255.0
    after_gray = np.array(crop_after.convert("L"), dtype=np.float32) / 255.0

    col_diffs = np.abs(after_gray.mean(axis=0) - before_gray.mean(axis=0))
    profile_diff = float(col_diffs.mean())
    max_col_diff = float(col_diffs.max()) if col_diffs.size > 0 else 0.0
    profile_peak_ratio = max_col_diff / (profile_diff + 1e-8)  # kept for diagnostics

    edge_shift_px = _estimate_progress_edge_shift_px(before_gray, after_gray)

    # ── Detection logic (multi-path) ───────────────────────────────────────
    # Noise can raise individual metrics, so we combine independent paths
    # with structure/profile guards.
    detect_flags = _detect_progress_changed(
        change_ratio_robust=change_ratio_robust,
        profile_diff=profile_diff,
        max_col_diff=max_col_diff,
        edge_shift_px=edge_shift_px,
        change_ratio_threshold=change_ratio_threshold,
        profile_diff_threshold=profile_diff_threshold,
        edge_shift_threshold_px=edge_shift_threshold_px,
    )
    detected_changed = bool(detect_flags["detected_changed"])
    passed = detected_changed if expected_change else (not detected_changed)

    basis = (
        "progress_change: "
        f"detected_changed={detected_changed} "
        f"expected_change={expected_change} "
        f"change_ratio={change_ratio_robust:.4f}/{change_ratio_threshold:.4f} "
        f"profile_diff={profile_diff:.4f}/{profile_diff_threshold:.4f} "
        f"edge_shift_px={edge_shift_px:.2f}/{edge_shift_threshold_px:.2f}"
    )

    text, semantics = _resolve_control_text_and_semantics(payload)
    control = ControlInfo(
        bounds=bounds,
        text=text,
        semantics=semantics,
        main_color=button_base_color(crop_after),
        source="cv",
    )
    details: Dict[str, Any] = {
        "method": "progress_change_cv",
        "expected_change": expected_change,
        "detected_changed": detected_changed,
        "pixel_threshold": pixel_threshold,
        "change_ratio_threshold": change_ratio_threshold,
        "profile_diff_threshold": profile_diff_threshold,
        "edge_shift_threshold_px": edge_shift_threshold_px,
        "change_ratio": change_ratio,
        "change_ratio_robust": change_ratio_robust,
        "profile_diff": profile_diff,
        "max_col_diff": max_col_diff,
        "profile_peak_ratio": profile_peak_ratio,
        "edge_shift_px": edge_shift_px,
        "clearly_changed": bool(detect_flags["clearly_changed"]),
        "both_above_threshold": bool(detect_flags["both_above_threshold"]),
        "change_ratio_with_structure": bool(detect_flags["change_ratio_with_structure"]),
        "ratio_dominant_with_profile_support": bool(
            detect_flags["ratio_dominant_with_profile_support"]
        ),
        "edge_shift_triggered": bool(detect_flags["edge_shift_triggered"]),
        "mean_abs_pixel_diff": mean_abs_pixel_diff,
        "crop_size": crop_after.size,
    }

    return CheckResult(
        passed=passed,
        basis=basis,
        control_info=control,
        details=details,
    )
