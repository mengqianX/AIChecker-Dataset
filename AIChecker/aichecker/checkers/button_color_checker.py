from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ..models import Bounds, CheckResult, ControlInfo
from ..utils import button_base_color, dominant_button_color, dominant_color, load_image, mean_color, parse_color
from PIL import Image


DEFAULT_TOLERANCE = 20  # max per-channel delta allowed
DEFAULT_PIXEL_THRESHOLD = 5
DEFAULT_RATIO_THRESHOLD = 0.08
AUTO_COLOR_CHANGE_KEYWORDS = ("auto_color_change", "auto_color_diff", "auto_color")


def _resolve_expected_color(payload: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    """
    Resolve expected color from payload.
    
    Returns:
        Tuple[int, int, int]: RGB color tuple if a valid color is specified
        None: If expected_color is None or matches AUTO_COLOR_CHANGE_KEYWORDS (signals auto color change detection)
    
    Raises:
        KeyError: If no expected_color/color/expected key is found in payload
    """
    for key in ("expected_color", "color", "expected"):
        if key in payload:
            if payload[key] is None or payload[key] in AUTO_COLOR_CHANGE_KEYWORDS:
                return None  # Signal to use auto color change detection
            return parse_color(payload[key])
    raise KeyError("Payload must include 'expected_color' (or 'color')")


def _resolve_screenshot(payload: Dict[str, Any]) -> str:
    for key in ("screenshot_b", "screenshot", "image", "target_screenshot"):
        if payload.get(key):
            return str(payload[key])
    raise KeyError("Payload must include 'screenshot_b' (or 'screenshot')")


def check_button_color(
    payload: Dict[str, Any],
    debug_dir: Path | None = None,
) -> CheckResult:
    """
    Validate that a button/control region matches an expected color.

    - Crops the region defined by `bounds` from the target screenshot.
    - Computes mean and dominant colors.
    - Compares mean color to expected with per-channel tolerance.
    """
    bounds = Bounds.from_sequence(payload["bounds"])
    expected_color = _resolve_expected_color(payload)
    tolerance = int(payload.get("tolerance") or payload.get("color_tolerance") or DEFAULT_TOLERANCE)
    screenshot_path = _resolve_screenshot(payload)

    img_after = load_image(screenshot_path)
    l, t, r, b = bounds.as_box()
    w, h = img_after.size
    if l < 0 or t < 0 or r > w or b > h:
        raise ValueError(f"Bounds out of image: bounds={bounds.as_box()} image_size={(w, h)}")

    crop_after = img_after.crop(bounds.as_box())
    before_color = None
    img_before: Optional[Image.Image] = None
    if payload.get("screenshot_a"):
        img_before = load_image(payload["screenshot_a"])
        crop_before = img_before.crop(bounds.as_box())
        # 更鲁棒的“按钮底色”估计（中心区域 + 量化聚类）
        before_color = button_base_color(crop_before)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        crop_after.save(debug_dir / "button_crop_after.png")
        if img_before:
            img_before.crop(bounds.as_box()).save(debug_dir / "button_crop_before.png")

    # 更鲁棒的“按钮底色”估计（中心区域 + 量化聚类）
    dom_color = button_base_color(crop_after)

    # =======================================================
    # MODE 2: Auto "color change" detection
    # expected_color is None → user wants to check if color changed
    # =======================================================
    if expected_color is None:
        if before_color is None:
            raise ValueError("expected_color=None but no screenshot_a provided for comparison.")
        
        # Compare before vs after (multi-signal)
        dom_before = dominant_button_color(crop_before)
        dom_after = dominant_button_color(crop_after)
        mean_before = mean_color(crop_before)
        mean_after = mean_color(crop_after)

        base_diff = tuple(abs(dom_color[i] - before_color[i]) for i in range(3))
        dom_diff = tuple(abs(dom_after[i] - dom_before[i]) for i in range(3))
        mean_diff = tuple(abs(mean_after[i] - mean_before[i]) for i in range(3))

        base_max_diff = max(base_diff)
        dom_max_diff = max(dom_diff)
        mean_max_diff = max(mean_diff)

        pixel_threshold = int(payload.get("pixel_threshold") or DEFAULT_PIXEL_THRESHOLD)
        ratio_threshold = float(payload.get("ratio_threshold") or DEFAULT_RATIO_THRESHOLD)

        before_arr = np.array(crop_before, dtype=np.int16)
        after_arr = np.array(crop_after, dtype=np.int16)
        max_per_pixel = np.abs(after_arr - before_arr).max(axis=2)
        change_ratio = float((max_per_pixel >= pixel_threshold).mean())

        passed = (
            base_max_diff > tolerance
            or dom_max_diff > tolerance
            or mean_max_diff > tolerance
            or change_ratio >= ratio_threshold
        )

        basis = (
            f"auto_color_change: "
            f"base_max_diff={base_max_diff} "
            f"dom_max_diff={dom_max_diff} "
            f"mean_max_diff={mean_max_diff} "
            f"change_ratio={change_ratio:.3f} "
            f"tolerance={tolerance} "
            f"pixel_threshold={pixel_threshold} "
            f"ratio_threshold={ratio_threshold}"
        )
        channel_diff = base_diff
        max_diff = base_max_diff
        distance = sum(d * d for d in base_diff) ** 0.5

    else:
        channel_diff = tuple(abs(dom_color[i] - expected_color[i]) for i in range(3))
        max_diff = max(channel_diff)
        distance = sum(d * d for d in channel_diff) ** 0.5
        passed = max_diff <= tolerance

        basis = (
            f"button_color_dominant: dominant_color={dom_color} expected={expected_color} "
            f"max_diff={max_diff} tolerance={tolerance}"
        )
        if before_color:
            basis += f" before_color={before_color}"

    control = ControlInfo(
        bounds=bounds,
        main_color=dom_color,
        source="cv",
    )
    
    details: Dict[str, Any] = {
        "method": "button_color_dominant" if expected_color else "auto_color_change",
        "expected_color": expected_color if expected_color else "N/A",
        "before_color": before_color if before_color else "N/A",
        "dominant_color": dom_color,
        "channel_diff": channel_diff,
        "max_diff": max_diff,
        "distance": distance,
        "tolerance": tolerance,
        "crop_size": crop_after.size,
    }
    if expected_color is None:
        details.update(
            {
                "base_diff": base_diff,
                "dom_before": dom_before,
                "dom_after": dom_after,
                "dom_diff": dom_diff,
                "mean_before": mean_before,
                "mean_after": mean_after,
                "mean_diff": mean_diff,
                "base_max_diff": base_max_diff,
                "dom_max_diff": dom_max_diff,
                "mean_max_diff": mean_max_diff,
                "pixel_threshold": pixel_threshold,
                "ratio_threshold": ratio_threshold,
                "change_ratio": change_ratio,
            }
        )

    return CheckResult(passed=passed, basis=basis, control_info=control, details=details)