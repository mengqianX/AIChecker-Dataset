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

# Minimum signal required in the supporting metrics for change_ratio to
# count as a "real" change.  Expressed as a fraction of each metric's own
# threshold.  Values below these fractions indicate pure image noise rather
# than structural bar movement.
# profile_diff must reach this fraction of its threshold before it can
# corroborate a change_ratio signal.  Raising to 50 % ensures that faint,
# diffuse perturbations (noise/JPEG artefacts) whose profile_diff is only
# marginally above a loose threshold cannot falsely confirm change_ratio.
_CHANGE_RATIO_CORROBORATION_PROFILE_FRAC = 0.50   # profile_diff >= 50 % of its threshold

# Minimum gradient peak height (normalized 0-1) required for the argmax-based
# edge position to be considered reliable.  When the column-mean profile is
# nearly flat (no clear bar boundary), argmax returns a noise-driven index that
# can jump by hundreds of pixels between two otherwise identical images.
_MIN_EDGE_GRADIENT = 0.01

# edge_shift_px is only meaningful when the column-mean profile has actually
# changed between the two frames.  If profile_diff is near zero, the two
# profiles are structurally identical and any argmax position difference is
# purely noise-driven — regardless of how large it appears.
# Gate: profile_diff must reach at least this fraction of its own threshold.
# Using 50 % to match the change_ratio corroboration level — both metrics
# require meaningful profile structure before they are considered reliable.
_EDGE_SHIFT_MIN_PROFILE_FRAC = 0.50   # profile_diff >= 50 % of profile_diff_threshold

# Real bar advancement concentrates column-diff changes near the moving edge;
# image-quality perturbations (noise, JPEG, mild resize) spread changes
# uniformly across all columns.  Require the peak column diff to be at least
# this many times the mean column diff before treating a near-threshold profile
# signal as "structural" (i.e. from a real bar movement rather than diffuse noise).
# NOTE: this check is bypassed when signals are clearly well above threshold
# (see _CLEARLY_CHANGED_* below), because large bar advances fill many columns
# and produce inherently diffuse — but genuine — changes.
_PROFILE_PEAK_MIN_RATIO = 4.0

# When change_ratio_robust or profile_diff exceed their threshold by this
# factor the change is considered "clearly large" and the concentration check
# is skipped entirely.  Noise/perturbation cases never reach these levels.
_CLEARLY_CHANGED_RATIO_FACTOR  = 5    # change_ratio_robust >= 5 × threshold
_CLEARLY_CHANGED_PROFILE_FACTOR = 3   # profile_diff         >= 3 × threshold


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

    # Concentration ratio: real bar movement concentrates diffs near the
    # advancing edge (high ratio); noise/compression spreads diffs uniformly
    # across all columns (ratio ≈ 1–2).
    profile_peak_ratio = max_col_diff / (profile_diff + 1e-8)
    profile_is_concentrated = profile_peak_ratio >= _PROFILE_PEAK_MIN_RATIO

    edge_shift_px = _estimate_progress_edge_shift_px(before_gray, after_gray)

    # ── Detection logic ────────────────────────────────────────────────────
    corroboration_min_profile = profile_diff_threshold * _CHANGE_RATIO_CORROBORATION_PROFILE_FRAC

    # "Clearly large" signals are unambiguous regardless of concentration.
    # Large bar advances fill many columns, producing diffuse but genuine changes
    # that would incorrectly fail a concentration test.
    clearly_changed = (
        change_ratio_robust >= change_ratio_threshold  * _CLEARLY_CHANGED_RATIO_FACTOR
        or profile_diff     >= profile_diff_threshold  * _CLEARLY_CHANGED_PROFILE_FACTOR
    )

    # Near-threshold region: require (a) concentration evidence AND (b) a
    # minimum profile_diff signal.  edge_shift is intentionally excluded here
    # because its reliability itself depends on profile structure — it cannot
    # independently vouch for change_ratio.
    change_ratio_corroborated = (
        change_ratio_robust >= change_ratio_threshold
        and profile_is_concentrated
        and profile_diff >= corroboration_min_profile
    )

    profile_diff_structural = (
        profile_diff >= profile_diff_threshold
        and profile_is_concentrated
    )

    edge_shift_min_profile = profile_diff_threshold * _EDGE_SHIFT_MIN_PROFILE_FRAC
    edge_shift_corroborated = (
        edge_shift_px >= edge_shift_threshold_px
        and profile_diff >= edge_shift_min_profile
        and profile_is_concentrated
    )

    detected_changed = (
        clearly_changed
        or change_ratio_corroborated
        or profile_diff_structural
        or edge_shift_corroborated
    )
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
        "profile_peak_ratio": profile_peak_ratio,
        "profile_is_concentrated": profile_is_concentrated,
        "clearly_changed": clearly_changed,
        "change_ratio_corroborated": change_ratio_corroborated,
        "profile_diff_structural": profile_diff_structural,
        "edge_shift_corroborated": edge_shift_corroborated,
        "profile_diff": profile_diff,
        "edge_shift_px": edge_shift_px,
        "mean_abs_pixel_diff": mean_abs_pixel_diff,
        "crop_size": crop_after.size,
    }

    return CheckResult(
        passed=passed,
        basis=basis,
        control_info=control,
        details=details,
    )
