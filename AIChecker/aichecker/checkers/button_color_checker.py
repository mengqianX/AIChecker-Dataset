from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import (
    button_base_color,
    diff_structure_score,
    load_image,
    parse_color,
)
from PIL import Image


DEFAULT_TOLERANCE = 20  # max per-channel delta allowed
DEFAULT_STRUCTURE_THRESHOLD = 0.25  # composite score threshold for auto_color_change
DEFAULT_PIXEL_DIFF_THRESHOLD = 20   # τ for binary diff mask in diff_structure_score
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
    # MODE 2: Auto "color change" detection (single principle)
    # =======================================================
    # The classical multi-signal approach (mean/dominant/coverage) cannot
    # distinguish "button responded to click" from "background frame drifted
    # under a transparent overlay button" — both produce large pixel-level
    # changes.  Instead we compute a single structural descriptor of the
    # diff `b - a` and threshold it.  See utils.diff_structure_score for the
    # full rationale; the three sub-metrics are:
    #   concentration : largest connected change blob / total changed pixels
    #   coherence     : how aligned the per-pixel RGB deltas are
    #   centrality    : whether the change concentrates at the centre or rings
    # All three are simultaneously high only when a real, localised, coherent
    # state transition happens (independent of which colour it transitions to).
    if expected_color is None:
        if before_color is None:
            raise ValueError("expected_color=None but no screenshot_a provided for comparison.")

        pixel_diff_threshold = int(
            payload.get("pixel_diff_threshold")
            or payload.get("pixel_threshold")
            or DEFAULT_PIXEL_DIFF_THRESHOLD
        )
        score_threshold = float(
            payload.get("structure_threshold")
            or payload.get("score_threshold")
            or DEFAULT_STRUCTURE_THRESHOLD
        )

        metrics = diff_structure_score(
            crop_before,
            crop_after,
            pixel_diff_threshold=pixel_diff_threshold,
        )
        score = metrics["score"]
        passed = score > score_threshold

        basis = (
            f"auto_color_change(structure): "
            f"score={score:.3f} threshold={score_threshold:.2f} "
            f"concentration={metrics['concentration']:.3f} "
            f"coherence={metrics['coherence']:.3f} "
            f"centrality={metrics['centrality']:.3f} "
            f"coverage={metrics['coverage']:.3f}"
        )
        # Keep these for backward-compatible report fields.
        channel_diff = tuple(abs(dom_color[i] - before_color[i]) for i in range(3))
        max_diff = max(channel_diff)
        distance = sum(d * d for d in channel_diff) ** 0.5

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
                "structure_score": metrics["score"],
                "concentration": metrics["concentration"],
                "coherence": metrics["coherence"],
                "centrality": metrics["centrality"],
                "coverage": metrics["coverage"],
                "n_changed": metrics["n_changed"],
                "pixel_diff_threshold": pixel_diff_threshold,
                "score_threshold": score_threshold,
            }
        )

    return CheckResult(passed=passed, basis=basis, control_info=control, details=details)