from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ..models import Bounds, CheckResult, ControlInfo
from ..utils import (
    button_base_color,
    diff_structure_score, 
    load_image,
    outline_iou_and_coverage,
    parse_color,
)
from PIL import Image, ImageFilter


DEFAULT_TOLERANCE = 20  # max per-channel delta allowed
DEFAULT_AUTO_COLOR_MODE = "hybrid"
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
DEFAULT_OUTLINE_HIGH_COV_IOU_VETO = 0.95    # in heavy blur/compression, unchanged controls often
                                             # keep very high outline IoU even with large coverage
DEFAULT_OUTLINE_HIGH_COV_MIN = 0.70          # enable a conservative static-control veto in that regime
DEFAULT_OUTLINE_HIGH_COV_MAX_COLOR_DELTA = 6  # only veto if dominant/base colour drift is tiny
DEFAULT_OUTLINE_HIGH_COV_MAX_COHERENCE = 0.75  # avoid vetoing genuine coherent activations
DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_SCORE = 0.80
DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_COHERENCE = 0.90
DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_CENTRALITY = 0.55
DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_COLOR_DELTA = 10
DEFAULT_OVERLAY_GATE_MIN_COVERAGE = 0.01
DEFAULT_OVERLAY_GATE_MAX_COVERAGE = 0.45
DEFAULT_OVERLAY_GATE_MIN_STABLE_RATIO = 0.45
DEFAULT_OVERLAY_GATE_MIN_STABLE_PIXELS = 10
DEFAULT_OVERLAY_GATE_TARGET_COVERAGE = 0.12
DEFAULT_OVERLAY_GATE_MIN_SCORE = 0.55
DEFAULT_OVERLAY_IOU_PENALTY_WEIGHT = 0.22
DEFAULT_OVERLAY_IOU_HARD_VETO = 0.90
DEFAULT_OVERLAY_HARD_VETO_MAX_CENTRALITY = 0.55
DEFAULT_OVERLAY_BG_DRIFT_MIN_SCORE = 0.70
DEFAULT_OVERLAY_BG_DRIFT_MIN_COHERENCE = 0.84
DEFAULT_OVERLAY_BG_DRIFT_MIN_CENTRALITY = 0.40
DEFAULT_OVERLAY_BG_DRIFT_MAX_CENTRALITY = 0.58
DEFAULT_OVERLAY_BG_DRIFT_MAX_COVERAGE = 0.20
DEFAULT_OVERLAY_BG_DRIFT_MIN_IOU = 0.35
DEFAULT_OVERLAY_BG_DRIFT_MAX_IOU = 0.74
DEFAULT_OUTLINE_MAX_SHIFT = 4           # search [-4, +4] pixel translations when computing
                                        # outline IoU, so that small layout jitter / video
                                        # overlay drift / capture-time animation doesn't
                                        # destroy the static-icon detection signal.
DEFAULT_SEGMENTATION_SEED_RATIO = 0.25
DEFAULT_SEGMENTATION_COLOR_TOLERANCE = 32
DEFAULT_SEGMENTATION_MIN_COVERAGE = 0.05
DEFAULT_SEGMENTATION_MAX_COVERAGE = 1.00
DEFAULT_SEGMENTATION_MIN_IOU = 0.20
DEFAULT_SEGMENTATION_COLOR_DELTA_THRESHOLD = 8.0
DEFAULT_SEGMENTATION_KERNEL = 5
AUTO_COLOR_CHANGE_KEYWORDS = ("auto_color_change", "auto_color_diff", "auto_color")
AUTO_COLOR_MODES = ("hybrid", "pure_segmentation")


def _binary_mask_morphology(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Close + open a binary mask with PIL Max/Min filters.
    """
    if kernel_size <= 1:
        return mask
    # PIL filters require odd sizes >= 3.
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    # close: dilation -> erosion, then open: erosion -> dilation.
    img = img.filter(ImageFilter.MaxFilter(size=kernel_size))
    img = img.filter(ImageFilter.MinFilter(size=kernel_size))
    img = img.filter(ImageFilter.MinFilter(size=kernel_size))
    img = img.filter(ImageFilter.MaxFilter(size=kernel_size))
    return np.array(img) > 0


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Return largest 8-connected component from a binary mask.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_pixels: list[tuple[int, int]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                y0 = max(0, cy - 1)
                y1 = min(h, cy + 2)
                x0 = max(0, cx - 1)
                x1 = min(w, cx + 2)
                for ny in range(y0, y1):
                    for nx in range(x0, x1):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels

    out = np.zeros_like(mask, dtype=bool)
    for py, px in best_pixels:
        out[py, px] = True
    return out


def _segment_button_body(
    crop: Image.Image,
    *,
    seed_ratio: float,
    color_tolerance: int,
    kernel_size: int,
    anchor_colors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Segment central button body by color-similarity to center seed region.
    """
    arr = np.array(crop.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    cy0 = int(h * (0.5 - seed_ratio / 2))
    cy1 = int(h * (0.5 + seed_ratio / 2))
    cx0 = int(w * (0.5 - seed_ratio / 2))
    cx1 = int(w * (0.5 + seed_ratio / 2))
    cy0, cy1 = max(0, cy0), max(cy0 + 1, min(h, cy1))
    cx0, cx1 = max(0, cx0), max(cx0 + 1, min(w, cx1))
    center_patch = arr[cy0:cy1, cx0:cx1]
    if anchor_colors is None:
        anchors = np.median(center_patch.reshape(-1, 3), axis=0).reshape(1, 3)
    else:
        anchors = anchor_colors

    max_delta = np.full((h, w), 1e9, dtype=np.float32)
    for anchor in anchors:
        dist = np.max(np.abs(arr - anchor[None, None, :]), axis=2).astype(np.float32)
        max_delta = np.minimum(max_delta, dist)
    mask = max_delta <= color_tolerance
    mask = _binary_mask_morphology(mask, kernel_size=kernel_size)
    if not mask.any():
        return mask
    mask = _largest_connected_component(mask)
    return mask


def _mean_color_in_mask(crop: Image.Image, mask: np.ndarray) -> Tuple[float, float, float]:
    arr = np.array(crop.convert("RGB"), dtype=np.float32)
    if not mask.any():
        return (0.0, 0.0, 0.0)
    sel = arr[mask]
    means = sel.mean(axis=0)
    return (float(means[0]), float(means[1]), float(means[2]))


def _segmentation_activation_score(
    crop_before: Image.Image,
    crop_after: Image.Image,
    *,
    seed_ratio: float,
    color_tolerance: int,
    kernel_size: int,
) -> Dict[str, Any]:
    # Use both frame seeds as anchors so segmentation remains stable
    # even when button color changes significantly between states.
    seed_before = np.array(
        _mean_color_in_mask(
            crop_before,
            _segment_button_body(
                crop_before,
                seed_ratio=seed_ratio,
                color_tolerance=max(8, color_tolerance // 2),
                kernel_size=1,
            ),
        ),
        dtype=np.float32,
    )
    seed_after = np.array(
        _mean_color_in_mask(
            crop_after,
            _segment_button_body(
                crop_after,
                seed_ratio=seed_ratio,
                color_tolerance=max(8, color_tolerance // 2),
                kernel_size=1,
            ),
        ),
        dtype=np.float32,
    )
    anchors = np.stack([seed_before, seed_after], axis=0)

    mask_before = _segment_button_body(
        crop_before,
        seed_ratio=seed_ratio,
        color_tolerance=color_tolerance,
        kernel_size=kernel_size,
        anchor_colors=anchors,
    )
    mask_after = _segment_button_body(
        crop_after,
        seed_ratio=seed_ratio,
        color_tolerance=color_tolerance,
        kernel_size=kernel_size,
        anchor_colors=anchors,
    )
    h, w = mask_before.shape
    cy, cx = h // 2, w // 2

    union = mask_before | mask_after
    inter = mask_before & mask_after
    coverage_before = float(mask_before.mean())
    coverage_after = float(mask_after.mean())
    iou = float(inter.sum() / max(1, union.sum()))

    mean_before = _mean_color_in_mask(crop_before, mask_before)
    mean_after = _mean_color_in_mask(crop_after, mask_after)
    color_delta = float(
        np.sqrt(sum((mean_after[i] - mean_before[i]) ** 2 for i in range(3)))
    )

    return {
        "mask_before_coverage": coverage_before,
        "mask_after_coverage": coverage_after,
        "mask_iou": iou,
        "center_hit_before": bool(mask_before[cy, cx]),
        "center_hit_after": bool(mask_after[cy, cx]),
        "mask_mean_color_before": mean_before,
        "mask_mean_color_after": mean_after,
        "mask_color_delta": color_delta,
    }


def _resolve_auto_color_mode(payload: Dict[str, Any]) -> str:
    mode = str(payload.get("auto_color_mode", DEFAULT_AUTO_COLOR_MODE)).strip().lower()
    if mode not in AUTO_COLOR_MODES:
        valid = ", ".join(AUTO_COLOR_MODES)
        raise ValueError(f"Invalid auto_color_mode={mode!r}. Valid values: {valid}")
    return mode


def _luma_diff_stats(before: Image.Image, after: Image.Image) -> Dict[str, float]:
    """
    Perceptual luminance diff stats used for "human-invisible noise" gating.
    """
    a = np.array(before.convert("RGB"), dtype=np.float32)
    b_img = after if after.size == before.size else after.resize(before.size)
    b = np.array(b_img.convert("RGB"), dtype=np.float32)
    da = np.abs(
        (0.299 * (b[:, :, 0] - a[:, :, 0]))
        + (0.587 * (b[:, :, 1] - a[:, :, 1]))
        + (0.114 * (b[:, :, 2] - a[:, :, 2]))
    )
    return {
        "mean": float(np.mean(da)),
        "p95": float(np.percentile(da, 95)),
    }


def _overlay_applicability(
    outline: Dict[str, Any],
    *,
    min_coverage: float,
    max_coverage: float,
    min_stable_ratio: float,
    min_stable_pixels: int,
    target_coverage: float,
    min_score: float,
) -> Dict[str, float | bool]:
    """
    Decide whether overlay-outline IoU logic should participate in voting.
    """
    coverage = float(outline.get("coverage", 0.0))
    count_a = int(outline.get("count_a", 0))
    count_b = int(outline.get("count_b", 0))
    stable_pixels = min(count_a, count_b)
    stable_ratio = stable_pixels / max(1, max(count_a, count_b))
    coverage_span = max(1e-6, max_coverage - min_coverage)
    coverage_delta = abs(coverage - target_coverage)
    coverage_score = max(0.0, 1.0 - min(1.0, coverage_delta / (0.5 * coverage_span)))
    stable_ratio_score = min(1.0, stable_ratio / max(min_stable_ratio, 1e-6))
    stable_pixels_score = min(1.0, stable_pixels / max(1, min_stable_pixels))
    score = 0.45 * coverage_score + 0.35 * stable_ratio_score + 0.20 * stable_pixels_score
    applicable = (
        min_coverage <= coverage <= max_coverage
        and stable_ratio >= min_stable_ratio
        and stable_pixels >= min_stable_pixels
        and score >= min_score
    )
    return {
        "applicable": applicable,
        "score": float(score),
        "stable_ratio": float(stable_ratio),
        "stable_pixels": float(stable_pixels),
        "coverage_score": float(coverage_score),
    }


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
    crop_before: Optional[Image.Image] = None
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
    metrics: Dict[str, Any] = {}
    outline: Dict[str, Any] = {}
    pixel_diff_threshold = DEFAULT_PIXEL_DIFF_THRESHOLD
    score_threshold = DEFAULT_STRUCTURE_THRESHOLD
    mask_dilate_radius = DEFAULT_MASK_DILATE_RADIUS
    outline_max_shift = DEFAULT_OUTLINE_MAX_SHIFT
    outline_veto_active = False
    outline_iou_veto = DEFAULT_OUTLINE_IOU_VETO
    outline_veto_min_cov = DEFAULT_OUTLINE_VETO_MIN_COV
    outline_veto_max_cov = DEFAULT_OUTLINE_VETO_MAX_COV
    outline_high_cov_iou_veto = DEFAULT_OUTLINE_HIGH_COV_IOU_VETO
    outline_high_cov_min = DEFAULT_OUTLINE_HIGH_COV_MIN
    outline_high_cov_max_color_delta = DEFAULT_OUTLINE_HIGH_COV_MAX_COLOR_DELTA
    outline_high_cov_max_coherence = DEFAULT_OUTLINE_HIGH_COV_MAX_COHERENCE
    static_low_centrality_max = 0.35
    perceptual_low_score_max = 0.40
    perceptual_jnd_p95_max = 8.0
    perceptual_jnd_mean_max = 2.2
    tiny_color_low_score_max = 0.40
    tiny_color_max_delta = 2
    near_static_mid_outline_cov_min = 0.35
    near_static_mid_outline_cov_max = 0.55
    near_static_mid_outline_iou_min = 0.75
    near_static_mid_coverage_min = 0.95
    near_static_mid_coherence_min = 0.95
    near_static_mid_centrality_min = 0.40
    near_static_mid_centrality_max = 0.60
    near_static_mid_max_color_delta = 6
    overlay_strong_activation_min_score = DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_SCORE
    overlay_strong_activation_min_coherence = DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_COHERENCE
    overlay_strong_activation_min_centrality = DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_CENTRALITY
    overlay_strong_activation_min_color_delta = DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_COLOR_DELTA
    overlay_gate_min_coverage = DEFAULT_OVERLAY_GATE_MIN_COVERAGE
    overlay_gate_max_coverage = DEFAULT_OVERLAY_GATE_MAX_COVERAGE
    overlay_gate_min_stable_ratio = DEFAULT_OVERLAY_GATE_MIN_STABLE_RATIO
    overlay_gate_min_stable_pixels = DEFAULT_OVERLAY_GATE_MIN_STABLE_PIXELS
    overlay_gate_target_coverage = DEFAULT_OVERLAY_GATE_TARGET_COVERAGE
    overlay_gate_min_score = DEFAULT_OVERLAY_GATE_MIN_SCORE
    overlay_iou_penalty_weight = DEFAULT_OVERLAY_IOU_PENALTY_WEIGHT
    overlay_iou_hard_veto = DEFAULT_OVERLAY_IOU_HARD_VETO
    overlay_hard_veto_max_centrality = DEFAULT_OVERLAY_HARD_VETO_MAX_CENTRALITY
    overlay_bg_drift_min_score = DEFAULT_OVERLAY_BG_DRIFT_MIN_SCORE
    overlay_bg_drift_min_coherence = DEFAULT_OVERLAY_BG_DRIFT_MIN_COHERENCE
    overlay_bg_drift_min_centrality = DEFAULT_OVERLAY_BG_DRIFT_MIN_CENTRALITY
    overlay_bg_drift_max_centrality = DEFAULT_OVERLAY_BG_DRIFT_MAX_CENTRALITY
    overlay_bg_drift_max_coverage = DEFAULT_OVERLAY_BG_DRIFT_MAX_COVERAGE
    overlay_bg_drift_min_iou = DEFAULT_OVERLAY_BG_DRIFT_MIN_IOU
    overlay_bg_drift_max_iou = DEFAULT_OVERLAY_BG_DRIFT_MAX_IOU
    overlay_gate: Dict[str, float | bool] = {"applicable": False, "score": 0.0}
    overlay_iou_penalty = 0.0
    auto_color_mode = DEFAULT_AUTO_COLOR_MODE
    details_seg: Dict[str, Any] = {}
    luma_stats: Dict[str, float] = {"mean": 0.0, "p95": 0.0}
    veto_reasons: list[str] = []

    if expected_color is None:
        auto_color_mode = _resolve_auto_color_mode(payload)
        if before_color is None:
            raise ValueError("expected_color=None but no screenshot_a provided for comparison.")
        if crop_before is None:
            raise ValueError("expected_color=None but screenshot_a crop is unavailable.")

        if auto_color_mode == "pure_segmentation":
            seg_seed_ratio = float(
                payload.get("seg_seed_ratio") or DEFAULT_SEGMENTATION_SEED_RATIO
            )
            seg_color_tolerance = int(
                payload.get("seg_color_tolerance") or DEFAULT_SEGMENTATION_COLOR_TOLERANCE
            )
            seg_min_coverage = float(
                payload.get("seg_min_coverage") or DEFAULT_SEGMENTATION_MIN_COVERAGE
            )
            seg_max_coverage = float(
                payload.get("seg_max_coverage") or DEFAULT_SEGMENTATION_MAX_COVERAGE
            )
            seg_min_iou = float(payload.get("seg_min_iou") or DEFAULT_SEGMENTATION_MIN_IOU)
            seg_color_delta_threshold = float(
                payload.get("seg_color_delta_threshold")
                or DEFAULT_SEGMENTATION_COLOR_DELTA_THRESHOLD
            )
            seg_kernel_size = int(payload.get("seg_kernel_size") or DEFAULT_SEGMENTATION_KERNEL)

            seg_metrics = _segmentation_activation_score(
                crop_before,
                crop_after,
                seed_ratio=seg_seed_ratio,
                color_tolerance=seg_color_tolerance,
                kernel_size=seg_kernel_size,
            )
            cov_before = seg_metrics["mask_before_coverage"]
            cov_after = seg_metrics["mask_after_coverage"]
            cov_ok = (
                seg_min_coverage <= cov_before <= seg_max_coverage
                and seg_min_coverage <= cov_after <= seg_max_coverage
            )
            center_ok = seg_metrics["center_hit_before"] and seg_metrics["center_hit_after"]
            iou_ok = seg_metrics["mask_iou"] >= seg_min_iou
            delta_ok = seg_metrics["mask_color_delta"] >= seg_color_delta_threshold

            passed = cov_ok and center_ok and iou_ok and delta_ok
            basis = (
                "auto_color_change(pure_segmentation): "
                f"delta={seg_metrics['mask_color_delta']:.3f}>={seg_color_delta_threshold:.2f}"
                f"={'Y' if delta_ok else 'N'} "
                f"iou={seg_metrics['mask_iou']:.3f}>={seg_min_iou:.2f}={'Y' if iou_ok else 'N'} "
                f"cov_a={cov_before:.3f} cov_b={cov_after:.3f} "
                f"center_ok={'Y' if center_ok else 'N'}"
            )
            metrics = {
                "score": 0.0,
                "concentration": 0.0,
                "coherence": 0.0,
                "centrality": 0.0,
                "coverage": 0.0,
                "n_changed": 0,
            }
            outline = {
                "ink_color": "N/A",
                "count_a": 0,
                "count_b": 0,
                "coverage": 0.0,
                "iou": 0.0,
                "shift": (0, 0),
            }
            luma_stats = _luma_diff_stats(crop_before, crop_after)
            veto_reasons = []
            details_seg = {
                "seg_seed_ratio": seg_seed_ratio,
                "seg_color_tolerance": seg_color_tolerance,
                "seg_min_coverage": seg_min_coverage,
                "seg_max_coverage": seg_max_coverage,
                "seg_min_iou": seg_min_iou,
                "seg_color_delta_threshold": seg_color_delta_threshold,
                "seg_kernel_size": seg_kernel_size,
                **seg_metrics,
                "seg_cov_ok": cov_ok,
                "seg_center_ok": center_ok,
                "seg_iou_ok": iou_ok,
                "seg_delta_ok": delta_ok,
            }
        else:
            details_seg = {}
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
            outline_veto_min_cov = float(payload.get("outline_veto_min_coverage") or DEFAULT_OUTLINE_VETO_MIN_COV)
            outline_veto_max_cov = float(payload.get("outline_veto_max_coverage") or DEFAULT_OUTLINE_VETO_MAX_COV)
            _mdr = payload.get("mask_dilate_radius")
            mask_dilate_radius = int(_mdr) if _mdr is not None else DEFAULT_MASK_DILATE_RADIUS
            _oms = payload.get("outline_max_shift")
            outline_max_shift = int(_oms) if _oms is not None else DEFAULT_OUTLINE_MAX_SHIFT
            outline_high_cov_iou_veto = float(
                payload.get("outline_high_cov_iou_veto") or DEFAULT_OUTLINE_HIGH_COV_IOU_VETO
            )
            outline_high_cov_min = float(
                payload.get("outline_high_cov_min") or DEFAULT_OUTLINE_HIGH_COV_MIN
            )
            outline_high_cov_max_color_delta = int(
                payload.get("outline_high_cov_max_color_delta") or DEFAULT_OUTLINE_HIGH_COV_MAX_COLOR_DELTA
            )
            outline_high_cov_max_coherence = float(
                payload.get("outline_high_cov_max_coherence") or DEFAULT_OUTLINE_HIGH_COV_MAX_COHERENCE
            )
            static_low_centrality_max = float(payload.get("static_low_centrality_max", 0.35))
            perceptual_low_score_max = float(payload.get("perceptual_low_score_max", 0.40))
            perceptual_jnd_p95_max = float(payload.get("perceptual_jnd_p95_max", 8.0))
            perceptual_jnd_mean_max = float(payload.get("perceptual_jnd_mean_max", 2.2))
            tiny_color_low_score_max = float(payload.get("tiny_color_low_score_max", 0.40))
            tiny_color_max_delta = int(payload.get("tiny_color_max_delta", 2))
            near_static_mid_outline_cov_min = float(payload.get("near_static_mid_outline_cov_min", 0.35))
            near_static_mid_outline_cov_max = float(payload.get("near_static_mid_outline_cov_max", 0.55))
            near_static_mid_outline_iou_min = float(payload.get("near_static_mid_outline_iou_min", 0.75))
            near_static_mid_coverage_min = float(payload.get("near_static_mid_coverage_min", 0.95))
            near_static_mid_coherence_min = float(payload.get("near_static_mid_coherence_min", 0.95))
            near_static_mid_centrality_min = float(payload.get("near_static_mid_centrality_min", 0.40))
            near_static_mid_centrality_max = float(payload.get("near_static_mid_centrality_max", 0.60))
            near_static_mid_max_color_delta = int(payload.get("near_static_mid_max_color_delta", 6))
            overlay_strong_activation_min_score = float(
                payload.get(
                    "overlay_strong_activation_min_score",
                    DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_SCORE,
                )
            )
            overlay_strong_activation_min_coherence = float(
                payload.get(
                    "overlay_strong_activation_min_coherence",
                    DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_COHERENCE,
                )
            )
            overlay_strong_activation_min_centrality = float(
                payload.get(
                    "overlay_strong_activation_min_centrality",
                    DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_CENTRALITY,
                )
            )
            overlay_strong_activation_min_color_delta = int(
                payload.get(
                    "overlay_strong_activation_min_color_delta",
                    DEFAULT_OVERLAY_STRONG_ACTIVATION_MIN_COLOR_DELTA,
                )
            )
            overlay_gate_min_coverage = float(
                payload.get("overlay_gate_min_coverage", DEFAULT_OVERLAY_GATE_MIN_COVERAGE)
            )
            overlay_gate_max_coverage = float(
                payload.get("overlay_gate_max_coverage", DEFAULT_OVERLAY_GATE_MAX_COVERAGE)
            )
            overlay_gate_min_stable_ratio = float(
                payload.get(
                    "overlay_gate_min_stable_ratio",
                    DEFAULT_OVERLAY_GATE_MIN_STABLE_RATIO,
                )
            )
            overlay_gate_min_stable_pixels = int(
                payload.get(
                    "overlay_gate_min_stable_pixels",
                    DEFAULT_OVERLAY_GATE_MIN_STABLE_PIXELS,
                )
            )
            overlay_gate_target_coverage = float(
                payload.get(
                    "overlay_gate_target_coverage",
                    DEFAULT_OVERLAY_GATE_TARGET_COVERAGE,
                )
            )
            overlay_gate_min_score = float(
                payload.get("overlay_gate_min_score", DEFAULT_OVERLAY_GATE_MIN_SCORE)
            )
            overlay_iou_penalty_weight = float(
                payload.get(
                    "overlay_iou_penalty_weight",
                    DEFAULT_OVERLAY_IOU_PENALTY_WEIGHT,
                )
            )
            overlay_iou_hard_veto = float(
                payload.get("overlay_iou_hard_veto", DEFAULT_OVERLAY_IOU_HARD_VETO)
            )
            overlay_hard_veto_max_centrality = float(
                payload.get(
                    "overlay_hard_veto_max_centrality",
                    DEFAULT_OVERLAY_HARD_VETO_MAX_CENTRALITY,
                )
            )
            overlay_bg_drift_min_score = float(
                payload.get(
                    "overlay_bg_drift_min_score",
                    DEFAULT_OVERLAY_BG_DRIFT_MIN_SCORE,
                )
            )
            overlay_bg_drift_min_coherence = float(
                payload.get(
                    "overlay_bg_drift_min_coherence",
                    DEFAULT_OVERLAY_BG_DRIFT_MIN_COHERENCE,
                )
            )
            overlay_bg_drift_min_centrality = float(
                payload.get(
                    "overlay_bg_drift_min_centrality",
                    DEFAULT_OVERLAY_BG_DRIFT_MIN_CENTRALITY,
                )
            )
            overlay_bg_drift_max_centrality = float(
                payload.get(
                    "overlay_bg_drift_max_centrality",
                    DEFAULT_OVERLAY_BG_DRIFT_MAX_CENTRALITY,
                )
            )
            overlay_bg_drift_max_coverage = float(
                payload.get(
                    "overlay_bg_drift_max_coverage",
                    DEFAULT_OVERLAY_BG_DRIFT_MAX_COVERAGE,
                )
            )
            overlay_bg_drift_min_iou = float(
                payload.get(
                    "overlay_bg_drift_min_iou",
                    DEFAULT_OVERLAY_BG_DRIFT_MIN_IOU,
                )
            )
            overlay_bg_drift_max_iou = float(
                payload.get(
                    "overlay_bg_drift_max_iou",
                    DEFAULT_OVERLAY_BG_DRIFT_MAX_IOU,
                )
            )
            metrics = diff_structure_score(
                crop_before,
                crop_after,
                pixel_diff_threshold=pixel_diff_threshold,
                mask_dilate_radius=mask_dilate_radius,
            )
            outline = outline_iou_and_coverage(crop_before, crop_after, max_shift=outline_max_shift)
            luma_stats = _luma_diff_stats(crop_before, crop_after)
            base_color_delta = max(abs(dom_color[i] - before_color[i]) for i in range(3))

            score = metrics["score"]
            overlay_gate = _overlay_applicability(
                outline,
                min_coverage=overlay_gate_min_coverage,
                max_coverage=overlay_gate_max_coverage,
                min_stable_ratio=overlay_gate_min_stable_ratio,
                min_stable_pixels=overlay_gate_min_stable_pixels,
                target_coverage=overlay_gate_target_coverage,
                min_score=overlay_gate_min_score,
            )
            is_overlay_icon = bool(overlay_gate["applicable"])
            is_overlay_strong_activation = (
                score >= overlay_strong_activation_min_score
                and metrics["coherence"] >= overlay_strong_activation_min_coherence
                and metrics["centrality"] >= overlay_strong_activation_min_centrality
                and base_color_delta >= overlay_strong_activation_min_color_delta
            )
            overlay_iou_excess = max(0.0, outline["iou"] - outline_iou_veto) / max(
                1e-6, 1.0 - outline_iou_veto
            )
            overlay_iou_penalty = (
                overlay_iou_penalty_weight
                * float(overlay_gate["score"])
                * overlay_iou_excess
                if is_overlay_icon and outline["iou"] >= outline_iou_veto and not is_overlay_strong_activation
                else 0.0
            )
            effective_score = max(0.0, score - overlay_iou_penalty)
            overlay_iou_hard_veto_fired = (
                is_overlay_icon
                and outline["iou"] >= overlay_iou_hard_veto
                and metrics["centrality"] <= overlay_hard_veto_max_centrality
                and not is_overlay_strong_activation
            )
            is_high_cov_static = (
                outline["coverage"] >= outline_high_cov_min
                and outline["iou"] >= outline_high_cov_iou_veto
                and base_color_delta <= outline_high_cov_max_color_delta
                and metrics["coherence"] <= outline_high_cov_max_coherence
            )
            is_static_high_iou_low_centrality = (
                outline["coverage"] >= 0.60
                and outline["iou"] >= outline_iou_veto
                and metrics["centrality"] <= static_low_centrality_max
            )
            is_perceptual_low_score_noise = (
                score <= perceptual_low_score_max
                and luma_stats["p95"] <= perceptual_jnd_p95_max
                and luma_stats["mean"] <= perceptual_jnd_mean_max
            )
            is_tiny_color_low_score_noise = (
                score <= tiny_color_low_score_max and base_color_delta <= tiny_color_max_delta
            )
            is_overlay_bg_drift_core = (
                is_overlay_icon
                and score >= overlay_bg_drift_min_score
                and metrics["coherence"] >= overlay_bg_drift_min_coherence
                and overlay_bg_drift_min_centrality <= metrics["centrality"] <= overlay_bg_drift_max_centrality
                and outline["coverage"] <= overlay_bg_drift_max_coverage
                and overlay_bg_drift_min_iou <= outline["iou"] <= overlay_bg_drift_max_iou
                and not is_overlay_strong_activation
            )
            is_overlay_bg_drift_mid_outline = (
                metrics["coverage"] >= near_static_mid_coverage_min
                and metrics["coherence"] >= near_static_mid_coherence_min
                and near_static_mid_centrality_min <= metrics["centrality"] <= near_static_mid_centrality_max
                and near_static_mid_outline_cov_min <= outline["coverage"] <= near_static_mid_outline_cov_max
                and outline["iou"] >= near_static_mid_outline_iou_min
                and base_color_delta <= near_static_mid_max_color_delta
                and not is_overlay_strong_activation
            )
            is_overlay_bg_drift = is_overlay_bg_drift_core or is_overlay_bg_drift_mid_outline

            outline_veto_active = (
                overlay_iou_hard_veto_fired
                or is_high_cov_static
                or is_static_high_iou_low_centrality
                or is_perceptual_low_score_noise
                or is_tiny_color_low_score_noise
                or is_overlay_bg_drift
            )
            if overlay_iou_penalty > 0:
                veto_reasons.append("overlay_icon_iou_soft_penalty")
            if overlay_iou_hard_veto_fired:
                veto_reasons.append("overlay_icon_iou_hard_veto")
            if is_overlay_icon and outline["iou"] >= outline_iou_veto and is_overlay_strong_activation:
                veto_reasons.append("overlay_icon_iou_waived_strong_activation")
            if is_high_cov_static:
                veto_reasons.append("high_cov_static")
            if is_static_high_iou_low_centrality:
                veto_reasons.append("static_high_iou_low_centrality")
            if is_perceptual_low_score_noise:
                veto_reasons.append("perceptual_low_score_noise")
            if is_tiny_color_low_score_noise:
                veto_reasons.append("tiny_color_low_score_noise")
            if is_overlay_bg_drift_core:
                veto_reasons.append("overlay_bg_drift_veto_core")
            if is_overlay_bg_drift_mid_outline:
                veto_reasons.append("overlay_bg_drift_veto_mid_outline")
            passed = (effective_score > score_threshold) and not outline_veto_active

            basis = (
                f"auto_color_change(structure+outline): "
                f"score={score:.3f} effective={effective_score:.3f}>{score_threshold:.2f}"
                f"={'Y' if effective_score > score_threshold else 'N'} "
                f"outline_iou={outline['iou']:.3f} coverage={outline['coverage']:.3f} "
                f"overlay_gate={int(bool(overlay_gate['applicable']))}:{float(overlay_gate['score']):.3f} "
                f"overlay_penalty={overlay_iou_penalty:.3f} "
                f"veto={'fired' if outline_veto_active else 'idle'}"
                f"{'[' + ','.join(veto_reasons) + ']' if veto_reasons else ''} "
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
                "auto_color_mode": auto_color_mode,
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
                "outline_high_cov_iou_veto": outline_high_cov_iou_veto,
                "outline_high_cov_min": outline_high_cov_min,
                "outline_high_cov_max_color_delta": outline_high_cov_max_color_delta,
                "outline_high_cov_max_coherence": outline_high_cov_max_coherence,
                "luma_diff_mean": luma_stats["mean"],
                "luma_diff_p95": luma_stats["p95"],
                "static_low_centrality_max": static_low_centrality_max,
                "perceptual_low_score_max": perceptual_low_score_max,
                "perceptual_jnd_p95_max": perceptual_jnd_p95_max,
                "perceptual_jnd_mean_max": perceptual_jnd_mean_max,
                "tiny_color_low_score_max": tiny_color_low_score_max,
                "tiny_color_max_delta": tiny_color_max_delta,
                "near_static_mid_outline_cov_min": near_static_mid_outline_cov_min,
                "near_static_mid_outline_cov_max": near_static_mid_outline_cov_max,
                "near_static_mid_outline_iou_min": near_static_mid_outline_iou_min,
                "near_static_mid_coverage_min": near_static_mid_coverage_min,
                "near_static_mid_coherence_min": near_static_mid_coherence_min,
                "near_static_mid_centrality_min": near_static_mid_centrality_min,
                "near_static_mid_centrality_max": near_static_mid_centrality_max,
                "near_static_mid_max_color_delta": near_static_mid_max_color_delta,
                "overlay_strong_activation_min_score": overlay_strong_activation_min_score,
                "overlay_strong_activation_min_coherence": overlay_strong_activation_min_coherence,
                "overlay_strong_activation_min_centrality": overlay_strong_activation_min_centrality,
                "overlay_strong_activation_min_color_delta": overlay_strong_activation_min_color_delta,
                "overlay_gate_applicable": bool(overlay_gate.get("applicable", False)),
                "overlay_gate_score": float(overlay_gate.get("score", 0.0)),
                "overlay_gate_stable_ratio": float(overlay_gate.get("stable_ratio", 0.0)),
                "overlay_gate_stable_pixels": float(overlay_gate.get("stable_pixels", 0.0)),
                "overlay_gate_coverage_score": float(overlay_gate.get("coverage_score", 0.0)),
                "overlay_gate_min_coverage": overlay_gate_min_coverage,
                "overlay_gate_max_coverage": overlay_gate_max_coverage,
                "overlay_gate_min_stable_ratio": overlay_gate_min_stable_ratio,
                "overlay_gate_min_stable_pixels": overlay_gate_min_stable_pixels,
                "overlay_gate_target_coverage": overlay_gate_target_coverage,
                "overlay_gate_min_score": overlay_gate_min_score,
                "overlay_iou_penalty_weight": overlay_iou_penalty_weight,
                "overlay_iou_penalty": overlay_iou_penalty,
                "overlay_iou_hard_veto": overlay_iou_hard_veto,
                "overlay_hard_veto_max_centrality": overlay_hard_veto_max_centrality,
                "overlay_bg_drift_min_score": overlay_bg_drift_min_score,
                "overlay_bg_drift_min_coherence": overlay_bg_drift_min_coherence,
                "overlay_bg_drift_min_centrality": overlay_bg_drift_min_centrality,
                "overlay_bg_drift_max_centrality": overlay_bg_drift_max_centrality,
                "overlay_bg_drift_max_coverage": overlay_bg_drift_max_coverage,
                "overlay_bg_drift_min_iou": overlay_bg_drift_min_iou,
                "overlay_bg_drift_max_iou": overlay_bg_drift_max_iou,
                "veto_reasons": veto_reasons,
            }
        )
        if auto_color_mode == "pure_segmentation":
            details.update(details_seg)

    return CheckResult(passed=passed, basis=basis, control_info=control, details=details)