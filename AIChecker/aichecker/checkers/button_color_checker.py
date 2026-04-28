from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import (
    button_base_color,
    diff_structure_score,
    load_image,
    outline_iou_and_coverage,
    parse_color,
)
from PIL import Image


DEFAULT_TOLERANCE = 20  # max per-channel delta allowed
DEFAULT_STRUCTURE_THRESHOLD = 0.25  # composite score threshold for auto_color_change
DEFAULT_PIXEL_DIFF_THRESHOLD = 10   # τ for binary diff mask; low enough to catch
                                    # subtle but uniform "whole-button darkens" activations
                                    # (e.g. semi-transparent button pressed state where
                                    # every pixel shifts by ~8 RGB).
DEFAULT_MASK_DILATE_RADIUS = 2      # bridge gaps between character glyphs before
                                    # measuring concentration, so text-style activations
                                    # ("Qwen3-Max" turning blue->black, made of ~30
                                    # disconnected character blobs) register as one
                                    # coherent text-line structure instead of dozens
                                    # of unrelated specks.
# Outline-IoU veto parameters: "icon whose ink outline didn't move" is conclusive
# evidence of no real activation, regardless of any structural noise we measure.
DEFAULT_OUTLINE_IOU_VETO = 0.80         # outline_iou >= this => veto fires
DEFAULT_OUTLINE_VETO_MIN_COV = 0.01     # below this we don't trust outline detection
DEFAULT_OUTLINE_VETO_MAX_COV = 0.55     # above this it's a labelled solid button, not an overlay icon
DEFAULT_OUTLINE_MAX_SHIFT = 4           # search [-4, +4] pixel translations when computing
                                        # outline IoU, so that small layout jitter / video
                                        # overlay drift / capture-time animation doesn't
                                        # destroy the static-icon detection signal.
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
    # MODE 2: Auto "color change" detection (two complementary signals)
    # =======================================================
    # We combine two independent diagnostics, each addressing a different
    # failure mode of the naive bbox-colour-diff approach:
    #
    #  (1) STRUCTURE SCORE  (utils.diff_structure_score)
    #      Asks: is the change between a and b localised (single connected
    #      blob), coherent (RGB deltas point in one direction), and centred
    #      in the bbox?  Real button activations satisfy all three; video
    #      background drift around a static button does not.
    #
    #  (2) OUTLINE-IoU VETO  (utils.outline_iou_and_coverage)
    #      For overlay-style icons (white outline on dynamic background),
    #      the structural signal can still be fooled when the surrounding
    #      video drifts coherently.  The geometric signature of activation
    #      is unmistakable though: a deliberate ink pattern (near-pure-white
    #      or near-pure-black pixels) shifts position or disappears.  When
    #      the bbox contains a small overlay icon whose ink mask is almost
    #      similar between a and b (IoU ≥ 0.80), we know the icon's geometry
    #      didn't change — vetoing any positive structural signal.  The 0.80
    #      threshold (rather than 0.95) is deliberately permissive: it allows
    #      some edge-pixel drift caused by a video background bleeding through
    #      a semi-transparent ink layer, which would otherwise let a stable
    #      overlay icon slip past the veto.
    #
    # The veto is restricted to outline_coverage ∈ [0.01, 0.55] because:
    #   - Below 1%: outline detection unreliable, can't trust IoU.
    #   - Above 55%: bbox is dominated by a static label/icon (solid buttons
    #     with white text "关注" etc.); legitimate activation would shift
    #     the button's body colour while the label persists, which would
    #     spuriously fire the veto.  Coverage in this regime correlates with
    #     "real activation candidates have outline_iou ≈ 0", so the iou
    #     threshold filters them out anyway.
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
        outline_iou_veto = float(
            payload.get("outline_iou_veto") or DEFAULT_OUTLINE_IOU_VETO
        )
        outline_veto_min_cov = float(
            payload.get("outline_veto_min_coverage") or DEFAULT_OUTLINE_VETO_MIN_COV
        )
        outline_veto_max_cov = float(
            payload.get("outline_veto_max_coverage") or DEFAULT_OUTLINE_VETO_MAX_COV
        )
        _mdr = payload.get("mask_dilate_radius")
        mask_dilate_radius = int(_mdr) if _mdr is not None else DEFAULT_MASK_DILATE_RADIUS
        _oms = payload.get("outline_max_shift")
        outline_max_shift = int(_oms) if _oms is not None else DEFAULT_OUTLINE_MAX_SHIFT

        metrics = diff_structure_score(
            crop_before,
            crop_after,
            pixel_diff_threshold=pixel_diff_threshold,
            mask_dilate_radius=mask_dilate_radius,
        )
        outline = outline_iou_and_coverage(crop_before, crop_after, max_shift=outline_max_shift)

        score = metrics["score"]
        is_overlay_icon = outline_veto_min_cov <= outline["coverage"] <= outline_veto_max_cov
        outline_veto_active = is_overlay_icon and outline["iou"] >= outline_iou_veto
        passed = (score > score_threshold) and not outline_veto_active

        basis = (
            f"auto_color_change(structure+outline): "
            f"score={score:.3f}>{score_threshold:.2f}={'Y' if score > score_threshold else 'N'} "
            f"outline_iou={outline['iou']:.3f} coverage={outline['coverage']:.3f} "
            f"veto={'fired' if outline_veto_active else 'idle'} "
            f"[concentration={metrics['concentration']:.3f} "
            f"coherence={metrics['coherence']:.3f} "
            f"centrality={metrics['centrality']:.3f}]"
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
                "mask_dilate_radius": mask_dilate_radius,
                "score_threshold": score_threshold,
                "outline_ink_color": outline["ink_color"],
                "outline_count_a": outline["count_a"],
                "outline_count_b": outline["count_b"],
                "outline_coverage": outline["coverage"],
                "outline_iou": outline["iou"],
                "outline_shift": outline.get("shift", (0, 0)),
                "outline_max_shift": outline_max_shift,
                "outline_veto_active": outline_veto_active,
                "outline_iou_veto": outline_iou_veto,
                "outline_veto_min_coverage": outline_veto_min_cov,
                "outline_veto_max_coverage": outline_veto_max_cov,
            }
        )

    return CheckResult(passed=passed, basis=basis, control_info=control, details=details)