from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import button_base_color, load_image


DEFAULT_PIXEL_THRESHOLD = 8
DEFAULT_CHANGE_RATIO_THRESHOLD = 0.02
DEFAULT_PROFILE_DIFF_THRESHOLD = 0.015
DEFAULT_EDGE_SHIFT_THRESHOLD_PX = 2.0


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

    before_arr = np.array(crop_before, dtype=np.int16)
    after_arr = np.array(crop_after, dtype=np.int16)
    abs_diff = np.abs(after_arr - before_arr)
    max_per_pixel = abs_diff.max(axis=2)

    change_ratio = float((max_per_pixel >= pixel_threshold).mean())
    mean_abs_pixel_diff = float(max_per_pixel.mean())

    before_gray = np.array(crop_before.convert("L"), dtype=np.float32) / 255.0
    after_gray = np.array(crop_after.convert("L"), dtype=np.float32) / 255.0
    profile_diff = float(np.mean(np.abs(after_gray.mean(axis=0) - before_gray.mean(axis=0))))
    edge_shift_px = _estimate_progress_edge_shift_px(before_gray, after_gray)

    detected_changed = (
        change_ratio >= change_ratio_threshold
        or profile_diff >= profile_diff_threshold
        or edge_shift_px >= edge_shift_threshold_px
    )
    passed = detected_changed if expected_change else (not detected_changed)

    basis = (
        "progress_change: "
        f"detected_changed={detected_changed} "
        f"expected_change={expected_change} "
        f"change_ratio={change_ratio:.4f}/{change_ratio_threshold:.4f} "
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
