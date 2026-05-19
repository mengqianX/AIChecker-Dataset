from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import load_image

# 默认相似度阈值（用于模板匹配主阈值；测试样例已统一改为依赖此默认值）
DEFAULT_SIMILARITY_THRESHOLD = 0.8
# 默认缩放范围（用于多尺度匹配，放宽以覆盖不同 DPI/分辨率）
DEFAULT_SCALE_MIN = 0.3
DEFAULT_SCALE_MAX = 2.0
DEFAULT_SCALE_STEP = 0.1
# 默认匹配方法
DEFAULT_MATCH_METHOD = cv2.TM_CCOEFF_NORMED
TIMING_ENV_VERBOSE = "AICHECKER_IMAGE_MATCH_TIMING_VERBOSE"
TIMING_ENV_CSV_PATH = "AICHECKER_IMAGE_MATCH_TIMING_CSV"


def _resolve_template_image(payload: Dict[str, Any]) -> str:
    """解析模板图路径。模板图 = 要寻找的图案（小图，如图标、裁块）。"""
    for key in ("template_image", "template", "small_image"):
        if payload.get(key):
            return str(payload[key])
    raise KeyError("Payload must include 'template_image'")


def _resolve_target_image(payload: Dict[str, Any]) -> str:
    """解析目标图路径。目标图 = 被搜索的整张图（大图，如整张截图）。"""
    for key in ("target_image", "target", "large_image", "screenshot"):
        if payload.get(key):
            return str(payload[key])
    raise KeyError("Payload must include 'target_image'")


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _resolve_timing_csv_path(payload: Dict[str, Any]) -> Path | None:
    payload_path = str(payload.get("timing_csv_path", "")).strip()
    env_path = str(os.getenv(TIMING_ENV_CSV_PATH, "")).strip()
    raw = payload_path or env_path
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[3] / p).resolve()


def _maybe_print_timing(payload: Dict[str, Any], line: str) -> None:
    payload_verbose = _normalize_bool(payload.get("timing_verbose", False))
    env_verbose = _normalize_bool(os.getenv(TIMING_ENV_VERBOSE, ""))
    if payload_verbose or env_verbose:
        print(line)


def _append_timing_csv(
    payload: Dict[str, Any],
    result: CheckResult,
    timing: Dict[str, Any],
) -> None:
    csv_path = _resolve_timing_csv_path(payload)
    if csv_path is None:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "run_at",
        "app",
        "case_id",
        "backend_used",
        "passed",
        "target_image",
        "template_image",
        "target_size",
        "template_size",
        "target_megapixels",
        "target_file_size_mb",
        "similarity",
        "total_sec",
        "feature_sec",
        "template_sec",
        "template_calls",
        "write_result_sec",
        "threshold_feature",
        "threshold_template",
        "basis",
    ]
    row = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app": str(payload.get("timing_app", payload.get("app", ""))),
        "case_id": str(payload.get("timing_case_id", payload.get("case_id", ""))),
        "backend_used": str(result.details.get("backend_used", "")),
        "passed": bool(result.passed),
        "target_image": str(payload.get("target_image", payload.get("target", ""))),
        "template_image": str(payload.get("template_image", payload.get("template", ""))),
        "target_size": str(timing.get("target_size", "")),
        "template_size": str(timing.get("template_size", "")),
        "target_megapixels": timing.get("target_megapixels", ""),
        "target_file_size_mb": timing.get("target_file_size_mb", ""),
        "similarity": result.details.get("similarity", ""),
        "total_sec": timing.get("total_sec", ""),
        "feature_sec": timing.get("feature_sec", ""),
        "template_sec": timing.get("template_sec", ""),
        "template_calls": timing.get("template_calls", ""),
        "write_result_sec": timing.get("write_result_sec", ""),
        "threshold_feature": timing.get("feature_similarity_threshold", ""),
        "threshold_template": timing.get("template_similarity_threshold", ""),
        "basis": str(result.basis).replace("\n", " "),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """将PIL Image转换为OpenCV格式（BGR）"""
    # PIL是RGB，OpenCV是BGR
    rgb_array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def _multi_scale_template_match(
    template: np.ndarray,
    target: np.ndarray,
    scale_min: float = DEFAULT_SCALE_MIN,
    scale_max: float = DEFAULT_SCALE_MAX,
    scale_step: float = DEFAULT_SCALE_STEP,
    method: int = DEFAULT_MATCH_METHOD,
) -> Tuple[float, Tuple[int, int, int, int], float]:
    """
    多尺度模板匹配
    
    Args:
        template: 模板图片（小图A），OpenCV格式（BGR）
        target: 目标图片（大图B），OpenCV格式（BGR）
        scale_min: 最小缩放比例
        scale_max: 最大缩放比例
        scale_step: 缩放步长
        method: OpenCV模板匹配方法
    
    Returns:
        Tuple[相似度, (left, top, right, bottom), 最佳缩放比例]
    """
    template_h, template_w = template.shape[:2]
    target_h, target_w = target.shape[:2]
    
    # ----- 原逻辑与 bug 说明 -----
    # 目标：在多个缩放比例(scale)下做模板匹配，取「相似度最高」的那次结果。
    # 流程：for scale in [0.3, 0.35, ..., 3.0]:
    #         把模板缩放到 scale 倍 → cv2.matchTemplate(...) 得到整图每个位置的匹配值
    #         minMaxLoc 得到这次 scale 下的最佳值 match_val 和位置
    #         若 match_val 比当前记录的 best_match_val 更大，就更新 best_*。
    # OpenCV 两种约定：TM_CCOEFF_NORMED 等是「值越大越像」；TM_SQDIFF_NORMED 是「值越小越像」。
    # 下面已把 SQDIFF 转成 (1-min_val)，所以统一成「match_val 越大越像」。
    # 因此要在所有 scale 里找「最大的 match_val」→ 初始值必须让「第一次得到的 match_val 能胜出」，
    # 即 best_match_val 初始应为「比任何实际 match_val 都小」→ 用 -np.inf。
    # 原 bug：对 CCOEFF 用了 best_match_val = np.inf，导致 match_val > np.inf 永远为 False，从不更新。
    best_match_val = -np.inf
    best_location = None
    best_scale = 1.0
    best_size = (template_w, template_h)
    
    # 计算合理的缩放范围
    # 确保缩放后的模板不会大于目标图片
    max_scale_w = target_w / template_w
    max_scale_h = target_h / template_h
    actual_scale_max = min(scale_max, max_scale_w, max_scale_h)
    actual_scale_min = max(scale_min, 0.1)  # 至少缩小到10%
    
    scales = np.arange(actual_scale_min, actual_scale_max + scale_step, scale_step, dtype=np.float32)

    # Deduplicate integer sizes: multiple scales can map to same (w,h).
    seen_sizes: set[Tuple[int, int]] = set()

    for scale_f in scales:
        scale = float(scale_f)
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)

        if scaled_w < 1 or scaled_h < 1 or scaled_w > target_w or scaled_h > target_h:
            continue
        size_key = (scaled_w, scaled_h)
        if size_key in seen_sizes:
            continue
        seen_sizes.add(size_key)

        scaled_template = cv2.resize(
            template,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA,
        )

        result = cv2.matchTemplate(target, scaled_template, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            # 对于SQDIFF方法，值越小越好
            # TM_SQDIFF_NORMED的值在0-1之间，值越小越好，转换为相似度：1.0 - min_val
            # TM_SQDIFF的值可能很大，值越小越好，但难以归一化，建议使用TM_SQDIFF_NORMED
            if method == cv2.TM_SQDIFF_NORMED:
                match_val = 1.0 - min_val  # 转换为相似度（0-1，越大越好）
            else:
                # TM_SQDIFF: 值越小越好，但值可能很大，使用负值以便统一比较
                # 注意：这种情况下相似度可能为负，建议使用TM_SQDIFF_NORMED
                match_val = -min_val
            match_loc = min_loc
        else:
            # 对于其他方法（CCOEFF, CCORR等），值越大越好
            match_val = max_val
            match_loc = max_loc
        
        # 更新最佳匹配（统一使用"越大越好"的比较方式）
        if match_val > best_match_val:
            best_match_val = match_val
            best_location = match_loc
            best_scale = scale
            best_size = (scaled_w, scaled_h)
    
    if best_location is None:
        return 0.0, (0, 0, 0, 0), 1.0
    
    # 计算bounds
    left = best_location[0]
    top = best_location[1]
    right = left + best_size[0]
    bottom = top + best_size[1]
    
    # 确保bounds在目标图片范围内
    left = max(0, min(left, target_w))
    top = max(0, min(top, target_h))
    right = max(left, min(right, target_w))
    bottom = max(top, min(bottom, target_h))
    
    return best_match_val, (left, top, right, bottom), best_scale


def _expand_bounds(
    bounds: Tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand bbox with symmetric padding and clip to image."""
    left, top, right, bottom = bounds
    bw = max(right - left, 1)
    bh = max(bottom - top, 1)
    pad_x = int(round(bw * max(0.0, padding_ratio)))
    pad_y = int(round(bh * max(0.0, padding_ratio)))
    nl = max(0, left - pad_x)
    nt = max(0, top - pad_y)
    nr = min(image_w, right + pad_x)
    nb = min(image_h, bottom + pad_y)
    if nr <= nl or nb <= nt:
        return bounds
    return nl, nt, nr, nb


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox_union(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if not boxes:
        return (0, 0, 0, 0)
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    return (left, top, right, bottom)


def _decompose_template_components(
    template: np.ndarray,
    min_area_ratio: float = 0.01,
    max_parts: int = 8,
) -> List[Dict[str, Any]]:
    """
    Heuristic template decomposition for multi-control strips.
    Returns parts with local boxes in template space.
    """
    h, w = template.shape[:2]
    if h <= 0 or w <= 0:
        return []

    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Highlight non-background foreground for white/light UI strips.
    fg_mask = (gray < 245).astype(np.uint8) * 255

    # Merge icon + label into one "control island", while preserving gaps.
    kx = max(3, int(round(w * 0.06)))
    ky = max(3, int(round(h * 0.30)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    merged = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    area_th = max(40, int(round(w * h * max(0.0, min(min_area_ratio, 0.2)))))

    parts: List[Dict[str, Any]] = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < area_th or bw < 6 or bh < 6:
            continue
        left = max(0, int(x) - 2)
        top = max(0, int(y) - 2)
        right = min(w, int(x + bw) + 2)
        bottom = min(h, int(y + bh) + 2)
        if right <= left or bottom <= top:
            continue
        crop = template[top:bottom, left:right]
        if crop.size == 0:
            continue
        parts.append(
            {
                "box": (left, top, right, bottom),
                "crop": crop,
                "area_ratio": float((right - left) * (bottom - top)) / float(max(w * h, 1)),
            }
        )

    parts.sort(key=lambda p: (p["box"][0], p["box"][1]))
    return parts[:max_parts]


def _compute_atomic_geometry_score(
    template_boxes: List[Tuple[int, int, int, int]],
    target_boxes: List[Tuple[int, int, int, int]],
) -> float:
    """
    Geometry consistency score in [0, 1]:
    order consistency + Y alignment + normalized spacing consistency.
    """
    if len(template_boxes) < 2 or len(target_boxes) < 2 or len(template_boxes) != len(target_boxes):
        return 0.0

    # 1) order consistency
    pair_total = 0
    pair_good = 0
    for i in range(len(template_boxes)):
        for j in range(i + 1, len(template_boxes)):
            tx_i = (template_boxes[i][0] + template_boxes[i][2]) * 0.5
            tx_j = (template_boxes[j][0] + template_boxes[j][2]) * 0.5
            gx_i = (target_boxes[i][0] + target_boxes[i][2]) * 0.5
            gx_j = (target_boxes[j][0] + target_boxes[j][2]) * 0.5
            pair_total += 1
            if (tx_i <= tx_j and gx_i <= gx_j) or (tx_i >= tx_j and gx_i >= gx_j):
                pair_good += 1
    order_score = float(pair_good) / float(max(pair_total, 1))

    # 2) Y alignment in target (for horizontal control rows)
    ys = [((b[1] + b[3]) * 0.5) for b in target_boxes]
    th = max(max(b[3] for b in target_boxes) - min(b[1] for b in target_boxes), 1.0)
    y_std_norm = float(np.std(np.array(ys, dtype=np.float32))) / float(th)
    y_score = _clamp01(1.0 - y_std_norm * 1.5)

    # 3) spacing ratio consistency
    t_centers = [((b[0] + b[2]) * 0.5) for b in template_boxes]
    g_centers = [((b[0] + b[2]) * 0.5) for b in target_boxes]
    t_gaps = [t_centers[i + 1] - t_centers[i] for i in range(len(t_centers) - 1)]
    g_gaps = [g_centers[i + 1] - g_centers[i] for i in range(len(g_centers) - 1)]
    if not t_gaps or not g_gaps:
        gap_score = 0.0
    else:
        t_span = max(sum(abs(v) for v in t_gaps), 1e-6)
        g_span = max(sum(abs(v) for v in g_gaps), 1e-6)
        t_norm = [v / t_span for v in t_gaps]
        g_norm = [v / g_span for v in g_gaps]
        gap_err = float(np.mean(np.abs(np.array(t_norm) - np.array(g_norm))))
        gap_score = _clamp01(1.0 - gap_err * 3.0)

    return _clamp01(0.4 * order_score + 0.3 * y_score + 0.3 * gap_score)


def _should_auto_apply_atomic(
    parts: List[Dict[str, Any]],
    template_shape: Tuple[int, int, int],
    payload: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether atomic mode should be applied in auto mode.
    """
    h, w = template_shape[:2]
    parts_count = len(parts)
    min_parts = int(payload.get("auto_decompose_min_parts", 2))
    auto_trigger_parts = int(payload.get("auto_decompose_auto_trigger_parts", 3))
    if parts_count < min_parts:
        return False, {
            "reason": "insufficient_parts",
            "parts_count": parts_count,
            "min_parts": min_parts,
        }

    aspect = float(w) / float(max(h, 1))
    y_centers = [((p["box"][1] + p["box"][3]) * 0.5) for p in parts]
    y_std_norm = (
        float(np.std(np.array(y_centers, dtype=np.float32))) / float(max(h, 1))
        if y_centers
        else 1.0
    )
    horizontal_alignment = _clamp01(1.0 - y_std_norm * 8.0)

    # Fast path: 3+ islands is usually a composite control strip.
    if parts_count >= auto_trigger_parts:
        return True, {
            "reason": "parts_count_trigger",
            "parts_count": parts_count,
            "aspect": aspect,
            "horizontal_alignment": horizontal_alignment,
        }

    # Two-part case: be stricter to avoid false positives on generic widgets.
    auto_two_part_min_aspect = float(payload.get("auto_decompose_auto_two_part_min_aspect", 2.6))
    auto_two_part_min_alignment = float(payload.get("auto_decompose_auto_two_part_min_alignment", 0.65))
    should_apply = (
        parts_count == 2
        and aspect >= auto_two_part_min_aspect
        and horizontal_alignment >= auto_two_part_min_alignment
    )
    return should_apply, {
        "reason": "two_part_rule" if should_apply else "auto_rule_not_satisfied",
        "parts_count": parts_count,
        "aspect": aspect,
        "horizontal_alignment": horizontal_alignment,
        "constraints": {
            "auto_two_part_min_aspect": auto_two_part_min_aspect,
            "auto_two_part_min_alignment": auto_two_part_min_alignment,
        },
    }


def _run_atomic_template_match(
    template_cv: np.ndarray,
    target_cv: np.ndarray,
    similarity_threshold: float,
    scale_min: float,
    scale_max: float,
    scale_step: float,
    match_method: int,
    payload: Dict[str, Any],
    coarse_bounds: Tuple[int, int, int, int] | None = None,
) -> Dict[str, Any]:
    """
    Atomic decomposition + per-part matching + score fusion.
    """
    # Controls:
    # - False / "off": disable
    # - True  / "on": force enable
    # - "auto" (default): auto-detect whether decomposition is needed
    auto_mode_raw = payload.get("auto_decompose_template", "auto")
    auto_mode = str(auto_mode_raw).strip().lower()
    if isinstance(auto_mode_raw, bool):
        auto_mode = "on" if auto_mode_raw else "off"
    if auto_mode in ("0", "false", "off", "no"):
        return {"enabled": False, "applied": False, "mode": "off"}

    min_parts = int(payload.get("auto_decompose_min_parts", 2))
    max_parts = int(payload.get("auto_decompose_max_parts", 8))
    min_area_ratio = float(payload.get("auto_decompose_min_area_ratio", 0.01))
    part_threshold = float(
        payload.get("atomic_part_similarity_threshold", max(0.30, similarity_threshold * 0.65))
    )
    w_hit = float(payload.get("atomic_weight_hit", 0.4))
    w_sim = float(payload.get("atomic_weight_similarity", 0.4))
    w_geo = float(payload.get("atomic_weight_geometry", 0.2))
    weight_sum = max(w_hit + w_sim + w_geo, 1e-6)
    w_hit, w_sim, w_geo = w_hit / weight_sum, w_sim / weight_sum, w_geo / weight_sum

    parts = _decompose_template_components(
        template_cv,
        min_area_ratio=min_area_ratio,
        max_parts=max_parts,
    )
    if len(parts) < min_parts:
        return {
            "enabled": True,
            "applied": False,
            "reason": "insufficient_parts",
            "parts_count": len(parts),
            "min_parts": min_parts,
            "mode": auto_mode,
        }

    auto_decision = {"mode": auto_mode}
    if auto_mode == "auto":
        should_apply, decision_meta = _should_auto_apply_atomic(parts, template_cv.shape, payload)
        auto_decision.update(decision_meta)
        if not should_apply:
            return {
                "enabled": True,
                "applied": False,
                "mode": "auto",
                "parts_count": len(parts),
                "auto_decision": auto_decision,
            }
    else:
        auto_decision["reason"] = "forced_on"

    def _run_once(
        search_target: np.ndarray,
        target_offset_xy: Tuple[int, int] = (0, 0),
    ) -> Dict[str, Any]:
        part_results: List[Dict[str, Any]] = []
        hit_boxes_target: List[Tuple[int, int, int, int]] = []
        hit_boxes_template: List[Tuple[int, int, int, int]] = []
        hit_scales: List[float] = []
        sim_values: List[float] = []
        hit_count = 0
        ox, oy = int(target_offset_xy[0]), int(target_offset_xy[1])

        for idx, part in enumerate(parts):
            sim, box_local, scale = _multi_scale_template_match(
                part["crop"],
                search_target,
                scale_min=scale_min,
                scale_max=scale_max,
                scale_step=scale_step,
                method=match_method,
            )
            box_global = (
                int(box_local[0]) + ox,
                int(box_local[1]) + oy,
                int(box_local[2]) + ox,
                int(box_local[3]) + oy,
            )
            sim_norm = _clamp01(sim)
            sim_values.append(sim_norm)
            passed = sim >= part_threshold
            if passed:
                hit_count += 1
                hit_boxes_target.append(box_global)
                hit_boxes_template.append(part["box"])
                hit_scales.append(scale)
            part_results.append(
                {
                    "index": idx,
                    "template_box": part["box"],
                    "target_box": box_global,
                    "similarity": sim,
                    "passed": passed,
                    "scale": scale,
                    "area_ratio": part["area_ratio"],
                }
            )

        hit_ratio = float(hit_count) / float(max(len(parts), 1))
        mean_similarity = float(np.mean(sim_values)) if sim_values else 0.0
        geometry_score = _compute_atomic_geometry_score(hit_boxes_template, hit_boxes_target)
        final_similarity = _clamp01(w_hit * hit_ratio + w_sim * mean_similarity + w_geo * geometry_score)
        final_bounds = _bbox_union(hit_boxes_target) if hit_boxes_target else (0, 0, 0, 0)
        final_scale = float(np.mean(hit_scales)) if hit_scales else 1.0
        return {
            "part_results": part_results,
            "hit_count": hit_count,
            "hit_ratio": hit_ratio,
            "mean_similarity": mean_similarity,
            "geometry_score": geometry_score,
            "final_similarity": final_similarity,
            "final_bounds": final_bounds,
            "final_scale": final_scale,
        }

    roi_first_enabled = bool(payload.get("atomic_roi_first", True))
    roi_padding_ratio = float(payload.get("atomic_roi_padding_ratio", 1.2))
    fallback_to_full = bool(payload.get("atomic_roi_fallback_to_full", True))
    fallback_min_hit_ratio = float(payload.get("atomic_roi_fallback_min_hit_ratio", 1.0))
    fallback_similarity_gap = float(payload.get("atomic_roi_fallback_similarity_gap", 0.0))
    roi_meta: Dict[str, Any] = {
        "enabled": roi_first_enabled,
        "used": False,
        "bounds": None,
        "fallback_to_full": False,
        "timing_sec": 0.0,
    }

    selected = None
    full_out = None
    roi_out = None
    full_started_at = 0.0
    full_sec = 0.0

    if roi_first_enabled and coarse_bounds is not None and coarse_bounds != (0, 0, 0, 0):
        h, w = target_cv.shape[:2]
        roi_bounds = _expand_bounds(coarse_bounds, w, h, roi_padding_ratio)
        x1, y1, x2, y2 = roi_bounds
        roi = target_cv[y1:y2, x1:x2]
        if roi.size > 0 and (x2 - x1) > 0 and (y2 - y1) > 0:
            roi_meta["used"] = True
            roi_meta["bounds"] = roi_bounds
            roi_started_at = time.perf_counter()
            roi_out = _run_once(roi, target_offset_xy=(x1, y1))
            roi_meta["timing_sec"] = max(0.0, time.perf_counter() - roi_started_at)
            should_fallback = fallback_to_full and (
                float(roi_out["hit_ratio"]) < max(0.0, min(1.0, fallback_min_hit_ratio))
                or float(roi_out["final_similarity"]) + max(0.0, fallback_similarity_gap) < similarity_threshold
            )
            roi_meta["fallback_to_full"] = should_fallback
            if not should_fallback:
                selected = roi_out

    if selected is None:
        full_started_at = time.perf_counter()
        full_out = _run_once(target_cv, target_offset_xy=(0, 0))
        full_sec = max(0.0, time.perf_counter() - full_started_at)
        selected = full_out

    return {
        "enabled": True,
        "applied": True,
        "mode": auto_mode,
        "auto_decision": auto_decision,
        "parts_count": len(parts),
        "part_similarity_threshold": part_threshold,
        "weights": {"hit": w_hit, "similarity": w_sim, "geometry": w_geo},
        "part_results": selected["part_results"],
        "hit_count": selected["hit_count"],
        "hit_ratio": selected["hit_ratio"],
        "mean_similarity": selected["mean_similarity"],
        "geometry_score": selected["geometry_score"],
        "final_similarity": selected["final_similarity"],
        "final_bounds": selected["final_bounds"],
        "final_scale": selected["final_scale"],
        "atomic_roi": roi_meta,
        "atomic_full_timing_sec": full_sec,
        "atomic_roi_timing_sec": float(roi_meta["timing_sec"]),
        "atomic_roi_used": bool(roi_meta["used"]),
        "atomic_roi_bounds": roi_meta["bounds"],
        "atomic_fallback_to_full": bool(roi_meta["fallback_to_full"]),
        "atomic_result_source": "roi" if selected is roi_out else "full",
        "atomic_full_result": full_out,
    }


def _write_final_match_result_image(result: CheckResult, output: Path | None) -> None:
    """Write a unified preview image for the final returned result."""
    if output is None:
        return
    try:
        target_path = result.details.get("target_path")
        if not target_path:
            return
        target_pil = load_image(str(target_path))
        target_cv = _pil_to_cv2(target_pil)
        bounds_tuple = result.details.get("bounds")
        if not isinstance(bounds_tuple, (list, tuple)) or len(bounds_tuple) != 4:
            return

        left, top, right, bottom = [int(v) for v in bounds_tuple]
        h, w = target_cv.shape[:2]
        left = max(0, min(left, w))
        top = max(0, min(top, h))
        right = max(left, min(right, w))
        bottom = max(top, min(bottom, h))

        vis = target_cv.copy()
        color = (0, 0, 255) if result.passed else (0, 128, 255)
        cv2.rectangle(vis, (left, top), (right, bottom), color, 2)
        backend = str(result.details.get("backend_used", "unknown"))
        similarity = float(result.details.get("similarity", 0.0) or 0.0)
        base_threshold = float(result.details.get("similarity_threshold", 0.0) or 0.0)
        decision_reason = "normal"
        effective_threshold = base_threshold
        if bool(result.details.get("passed_by_relaxed_template_threshold", False)):
            decision_reason = "auto_relaxed_template"
            effective_threshold = float(
                result.details.get("auto_relaxed_template_threshold", base_threshold) or base_threshold
            )
        elif bool(result.details.get("passed_by_auto_offscale_relax", False)):
            decision_reason = "auto_offscale_relax"
            off_meta = result.details.get("auto_offscale_relax", {}) or {}
            effective_threshold = float(off_meta.get("offscale_threshold", base_threshold) or base_threshold)
        status = "PASS" if result.passed else "FAIL"
        label = (
            f"{status} | backend={backend} | sim={similarity:.3f} | "
            f"th={effective_threshold:.3f} (base={base_threshold:.3f}) | reason={decision_reason}"
        )
        cv2.putText(
            vis,
            label,
            (left, max(0, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / "final_match_result.png"), vis)
    except Exception:
        # Preview generation should not affect checker behavior.
        return


def _check_image_match_template(
    payload: Dict[str, Any],
    output: Path | None = None,
) -> CheckResult:
    """
    在目标图片中查找模板图片，支持多尺度匹配以适应不同分辨率。
    
    Args:
        payload: 包含以下字段的字典：
            - template_image: 模板图 = 要找的图案（小图）。注意：这里「要找的」是 template，不是 target。
            - target_image: 目标图 = 被搜索的那张图（大图）。注意：target 指「被搜索的对象/场所」，不是「要找的目标物」。
            语义：在 target_image 里找 template_image。若传 image_a/image_b，会按尺寸自动把小图当模板、大图当目标。
            - similarity_threshold (可选): 相似度阈值，默认 0.65
            - scale_min (可选): 最小缩放比例，默认0.5
            - scale_max (可选): 最大缩放比例，默认2.0
            - scale_step (可选): 缩放步长，默认0.1
            - match_method (可选): OpenCV匹配方法，默认cv2.TM_CCOEFF_NORMED
            - boundary_guard (可选): 是否启用边界缩放保护，默认True
            - boundary_similarity_margin (可选): 边界缩放额外相似度裕量，默认0.12
            - offscale_guard (可选): 是否启用偏离1x缩放保护，默认True
            - offscale_min_deviation (可选): 触发保护的最小缩放偏移量，默认0.15
            - offscale_similarity_margin (可选): 偏离1x时额外相似度裕量，默认0.08
            - offscale_extra_per_unit (可选): 每增加1.0缩放偏移量附加的阈值增量，默认0.25
            - offscale_max_extra (可选): 偏离1x保护的最大附加阈值，默认0.20
            - template_refine (可选): 是否启用模板二阶段精修（粗匹配后ROI复检），默认True
            - refine_padding_ratio (可选): 二阶段ROI扩边比例，默认0.2
            - refine_scale_window (可选): 二阶段围绕粗scale搜索窗口比例，默认0.25
            - refine_scale_step (可选): 二阶段搜索步长，默认max(0.01, scale_step*0.5)
            - refine_accept_delta (可选): 二阶段结果可接受的分数劣化容忍，默认0.01
        output: 输出目录（可选）
    
    Returns:
        CheckResult对象，包含：
            - passed: 是否找到匹配（相似度 >= threshold）
            - basis: 判定依据
            - control_info: 匹配到的位置信息
            - details: 详细信息（相似度、bounds、缩放比例等）
    """
    # 解析输入
    template_path = _resolve_template_image(payload)
    target_path = _resolve_target_image(payload)
    
    similarity_threshold = float(payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    scale_min = float(payload.get("scale_min", DEFAULT_SCALE_MIN))
    scale_max = float(payload.get("scale_max", DEFAULT_SCALE_MAX))
    scale_step = float(payload.get("scale_step", DEFAULT_SCALE_STEP))
    boundary_guard = bool(payload.get("boundary_guard", False))
    # 边界保护默认只对“命中最小缩放边界”更敏感（极小缩放更容易误匹配）。
    # 若你明确希望对“命中最大缩放边界”也做同样的拒绝，可设置 boundary_guard_upper=true。
    boundary_guard_upper = bool(payload.get("boundary_guard_upper", False))
    boundary_similarity_margin = float(payload.get("boundary_similarity_margin", 0.12))
    offscale_guard = bool(payload.get("offscale_guard", True))
    # 默认采用“更宽松”off-scale 保护：
    # - 约 0.7x~1.3x 的缩放通常不触发额外拒绝
    # - 即使触发，附加阈值也更小，尽量降低跨端分辨率场景漏报 
    offscale_min_deviation = float(payload.get("offscale_min_deviation", 0.5))
    offscale_similarity_margin = float(payload.get("offscale_similarity_margin", 0.01))
    offscale_extra_per_unit = float(payload.get("offscale_extra_per_unit", 0.04))
    offscale_max_extra = float(payload.get("offscale_max_extra", 0.05))
    
    # 解析匹配方法
    match_method_str = payload.get("match_method", "TM_CCOEFF_NORMED")
    match_method_map = {
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF": cv2.TM_SQDIFF,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }
    match_method = match_method_map.get(match_method_str, DEFAULT_MATCH_METHOD)
    
    # 加载图片
    template_pil = load_image(template_path)
    target_pil = load_image(target_path)
    
    # 约定：在“大图”中找“小图”。若当前 template 比 target 大，则交换（避免在小图里找大图导致相似度恒为 0）
    tw, th = template_pil.size
    gw, gh = target_pil.size
    swapped_by_size = (tw * th) > (gw * gh)
    if swapped_by_size:
        template_pil, target_pil = target_pil, template_pil
        template_path, target_path = target_path, template_path

    # 计算实际可用缩放范围（与 _multi_scale_template_match 一致）
    # 计算实际可用缩放范围（与 _multi_scale_template_match 一致）
    tw, th = template_pil.size
    gw, gh = target_pil.size
    effective_scale_min = max(scale_min, 0.1)
    effective_scale_max = min(scale_max, gw / tw, gh / th)
    
    # 转换为OpenCV格式
    template_cv = _pil_to_cv2(template_pil)
    target_cv = _pil_to_cv2(target_pil)
    
    # 执行多尺度模板匹配
    similarity, bounds_tuple, best_scale = _multi_scale_template_match(
        template_cv,
        target_cv,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_step=scale_step,
        method=match_method,
    )

    # 二阶段 refinement：先粗匹配得到 ROI，再在局部区域做细粒度复检，提升定位精度。
    template_refine_enabled = bool(payload.get("template_refine", True))
    refine_info: Dict[str, Any] = {
        "enabled": template_refine_enabled,
        "attempted": False,
        "accepted": False,
    }
    if template_refine_enabled and bounds_tuple != (0, 0, 0, 0):
        target_h, target_w = target_cv.shape[:2]
        refine_padding_ratio = float(payload.get("refine_padding_ratio", 0.2))
        refine_scale_window = float(payload.get("refine_scale_window", 0.25))
        refine_scale_step = float(payload.get("refine_scale_step", max(0.01, scale_step * 0.5)))
        refine_accept_delta = float(payload.get("refine_accept_delta", 0.01))

        coarse_bounds = bounds_tuple
        roi_bounds = _expand_bounds(coarse_bounds, target_w, target_h, refine_padding_ratio)
        x1, y1, x2, y2 = roi_bounds
        roi = target_cv[y1:y2, x1:x2]
        refine_info.update(
            {
                "attempted": True,
                "coarse_similarity": similarity,
                "coarse_scale": best_scale,
                "coarse_bounds": coarse_bounds,
                "roi_bounds": roi_bounds,
            }
        )
        if roi.size > 0:
            local_scale_min = max(0.1, best_scale * (1.0 - refine_scale_window))
            local_scale_max = max(local_scale_min + 0.01, best_scale * (1.0 + refine_scale_window))
            refine_similarity, refine_local_bounds, refine_scale = _multi_scale_template_match(
                template_cv,
                roi,
                scale_min=local_scale_min,
                scale_max=local_scale_max,
                scale_step=refine_scale_step,
                method=match_method,
            )
            rl, rt, rr, rb = refine_local_bounds
            refine_global_bounds = (x1 + rl, y1 + rt, x1 + rr, y1 + rb)
            refine_info.update(
                {
                    "refine_similarity": refine_similarity,
                    "refine_scale": refine_scale,
                    "refine_bounds": refine_global_bounds,
                    "scale_range": {
                        "min": local_scale_min,
                        "max": local_scale_max,
                        "step": refine_scale_step,
                    },
                }
            )
            if refine_similarity + refine_accept_delta >= similarity:
                similarity = refine_similarity
                bounds_tuple = refine_global_bounds
                best_scale = refine_scale
                refine_info["accepted"] = True

    atomic_info = _run_atomic_template_match(
        template_cv=template_cv,
        target_cv=target_cv,
        similarity_threshold=similarity_threshold,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_step=scale_step,
        match_method=match_method,
        payload=payload,
        coarse_bounds=bounds_tuple,
    )
    if atomic_info.get("applied"):
        atomic_similarity = float(atomic_info.get("final_similarity", 0.0) or 0.0)
        if atomic_similarity >= similarity:
            similarity = atomic_similarity
            bounds_tuple = tuple(atomic_info.get("final_bounds", bounds_tuple))
            best_scale = float(atomic_info.get("final_scale", best_scale) or best_scale)
    
    # 判断是否匹配成功：先做阈值判定，再做边界缩放保护
    passed_by_threshold = similarity >= similarity_threshold
    passed = passed_by_threshold
    rejection_reason: str | None = None
    rejection_meta: Dict[str, Any] = {}
    if passed_by_threshold and boundary_guard and effective_scale_min <= effective_scale_max:
        # 命中最小/最大缩放边界且相似度仅略高于阈值时，常是误匹配（尤其是极小缩放）
        scale_eps = max(scale_step / 2.0, 1e-6)
        at_lower = abs(best_scale - effective_scale_min) <= scale_eps
        at_upper = abs(best_scale - effective_scale_max) <= scale_eps
        # 这里的 threshold 是“额外严格”阈值，但不应超过 1.0（TM_CCOEFF_NORMED 的上界）
        boundary_threshold = min(1.0, similarity_threshold + boundary_similarity_margin)
        should_reject = (at_lower or (boundary_guard_upper and at_upper)) and similarity < boundary_threshold
        if should_reject:
            passed = False
            rejection_reason = "boundary_scale"
            rejection_meta = {
                "position": "lower" if at_lower else "upper",
                "threshold": boundary_threshold,
            }
    if (
        passed
        and passed_by_threshold
        and offscale_guard
    ):
        # 非1x缩放下，若仅略高于基础阈值，通常是“形状近似”误匹配
        scale_deviation = abs(best_scale - 1.0)
        extra_deviation = max(scale_deviation - offscale_min_deviation, 0.0)
        dynamic_extra = min(extra_deviation * offscale_extra_per_unit, offscale_max_extra)
        # off-scale 更严格阈值，同样不应超过 1.0
        offscale_threshold = min(1.0, similarity_threshold + offscale_similarity_margin + dynamic_extra)
        if scale_deviation >= offscale_min_deviation and similarity < offscale_threshold:
            passed = False
            rejection_reason = "offscale_low_confidence"
            rejection_meta = {
                "deviation": scale_deviation,
                "threshold": offscale_threshold,
                "dynamic_extra": dynamic_extra,
            }
    
    # 构建bounds
    bounds = Bounds.from_sequence(bounds_tuple)
    
    atomic_basis_suffix = ""
    atomic_mode = str(atomic_info.get("mode", "off"))
    atomic_applied = bool(atomic_info.get("applied", False))
    atomic_reason = ""
    if atomic_applied:
        atomic_decision = atomic_info.get("auto_decision", {}) or {}
        atomic_reason = str(atomic_decision.get("reason", "applied"))
        atomic_basis_suffix = (
            "; atomic="
            f"{atomic_mode}/{atomic_reason}, "
            f"parts={int(atomic_info.get('parts_count', 0))}, "
            f"hit={int(atomic_info.get('hit_count', 0))}/"
            f"{int(atomic_info.get('parts_count', 0))}, "
            f"geo={float(atomic_info.get('geometry_score', 0.0) or 0.0):.3f}"
        )
    elif atomic_mode != "off":
        atomic_reason = str(
            (atomic_info.get("auto_decision", {}) or {}).get("reason")
            or atomic_info.get("reason")
            or "not_applied"
        )
        atomic_basis_suffix = f"; atomic={atomic_mode}/{atomic_reason}"

    # 构建basis
    if passed:
        basis = (
            f"image_match: Found template in target image with similarity={similarity:.4f} "
            f"(threshold={similarity_threshold:.4f}), bounds={bounds.as_box()}, scale={best_scale:.2f}"
        )
    else:
        if rejection_reason == "boundary_scale":
            basis = (
                f"image_match: Boundary-scale match rejected. "
                f"Similarity={similarity:.4f} is below boundary threshold="
                f"{rejection_meta.get('threshold', (similarity_threshold + boundary_similarity_margin)):.4f} "
                f"at {rejection_meta.get('position')} scale "
                f"(base threshold={similarity_threshold:.4f}, scale={best_scale:.2f}), "
                f"bounds={bounds.as_box()}"
            )
        elif rejection_reason == "offscale_low_confidence":
            basis = (
                f"image_match: Off-scale low-confidence match rejected. "
                f"Similarity={similarity:.4f} is below off-scale threshold="
                f"{rejection_meta.get('threshold', (similarity_threshold + offscale_similarity_margin)):.4f} "
                f"(base threshold={similarity_threshold:.4f}, scale={best_scale:.2f}, "
                f"deviation={rejection_meta.get('deviation', abs(best_scale - 1.0)):.2f}), "
                f"bounds={bounds.as_box()}"
            )
        else:
            basis = (
                f"image_match: Template not found in target image. "
                f"Best similarity={similarity:.4f} < threshold={similarity_threshold:.4f}, "
                f"best_match_bounds={bounds.as_box()}, scale={best_scale:.2f}"
            )
    basis = f"{basis}{atomic_basis_suffix}"
    
    # 构建ControlInfo
    control_info = ControlInfo(
        bounds=bounds,
        source="image_match",
        extras={
            "similarity": similarity,
            "best_scale": best_scale,
            "template_size": template_pil.size,
            "target_size": target_pil.size,
        },
    )
    
    # 构建details
    details: Dict[str, Any] = {
        "method": "multi_scale_template_match",
        "similarity": similarity,
        "similarity_threshold": similarity_threshold,
        "bounds": bounds_tuple,
        "best_scale": best_scale,
        "template_path": template_path,
        "target_path": target_path,
        "template_size": template_pil.size,
        "target_size": target_pil.size,
        "match_method": match_method_str,
        "swapped_by_size": swapped_by_size,
        "passed_by_threshold": passed_by_threshold,
        "boundary_guard": {
            "enabled": boundary_guard,
            "rejected": rejection_reason == "boundary_scale",
            "position": rejection_meta.get("position"),
            "upper_enabled": boundary_guard_upper,
            "similarity_margin": boundary_similarity_margin,
            "effective_scale_min": effective_scale_min,
            "effective_scale_max": effective_scale_max,
        },
        "offscale_guard": {
            "enabled": offscale_guard,
            "rejected": rejection_reason == "offscale_low_confidence",
            "threshold": rejection_meta.get("threshold"),
            "min_deviation": offscale_min_deviation,
            "similarity_margin": offscale_similarity_margin,
            "extra_per_unit": offscale_extra_per_unit,
            "max_extra": offscale_max_extra,
            "dynamic_extra": rejection_meta.get("dynamic_extra", 0.0),
            "scale_deviation": abs(best_scale - 1.0),
        },
        "rejection_reason": rejection_reason,
        "scale_range": {
            "min": scale_min,
            "max": scale_max,
            "step": scale_step,
        },
        "template_refine": refine_info,
        "template_atomic": atomic_info,
    }
    
    # 调试输出：红框标出匹配区域，便于直观查看
    if output:
        output.mkdir(parents=True, exist_ok=True)
        target_vis = target_cv.copy()
        box_color = (0, 0, 255)  # BGR 红框，更醒目
        cv2.rectangle(
            target_vis,
            (bounds.left, bounds.top),
            (bounds.right, bounds.bottom),
            box_color,
            2,
        )
        cv2.putText(
            target_vis,
            f"Similarity: {similarity:.3f}, Scale: {best_scale:.2f}",
            (bounds.left, max(0, bounds.top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )
        cv2.imwrite(str(output / "match_result.png"), target_vis)
        cv2.imwrite(str(output / "template.png"), template_cv)
        cv2.imwrite(str(output / "target.png"), target_cv)
    
    return CheckResult(passed, basis, control_info, details)


def check_image_match(
    payload: Dict[str, Any],
    output: Path | None = None,
) -> CheckResult:
    """
    对外统一入口（默认优先 feature 匹配，必要时回退模板匹配）。

    Payload 可选字段：
    - match_backend: "auto" | "feature" | "template"（默认 "auto"）
    - fallback_to_template: 当后端为 auto/feature 且 feature 失败时，是否回退模板匹配（默认 True）
    - feature_similarity_threshold: feature 匹配阈值（可选）
    - template_similarity_threshold: template 匹配阈值（可选）
    - auto_bbox_fusion: feature通过后是否尝试与template做bbox融合（默认True）
    - auto_bbox_margin_ratio: bbox融合时全宽边距比例（默认0.02）
    - auto_offscale_near_threshold_relax: auto模式下是否启用offscale边缘放行（默认True）
    - auto_offscale_min_template_similarity: 边缘放行要求的最小template相似度（默认0.88）
    - auto_offscale_gap_max: 边缘放行允许的最大阈值差距(offscale_threshold-similarity)，默认0.035
    - auto_offscale_scale_min / auto_offscale_scale_max: 边缘放行允许的scale区间，默认[0.75, 1.6]
    - auto_offscale_min_feature_good_matches: 边缘放行要求的最小feature匹配点数，默认4
    - auto_offscale_min_feature_similarity: 边缘放行要求的最小feature相似度，默认0.01
    """
    started_at = time.perf_counter()
    feature_sec = 0.0
    template_sec = 0.0
    template_calls = 0
    write_result_sec = 0.0
    target_size = ()
    template_size = ()
    target_megapixels = 0.0
    target_file_size_mb = 0.0

    def _finalize(result: CheckResult) -> CheckResult:
        nonlocal write_result_sec, target_size, template_size, target_megapixels, target_file_size_mb
        write_started_at = time.perf_counter()
        _write_final_match_result_image(result, output)
        write_result_sec += max(0.0, time.perf_counter() - write_started_at)

        target_size = tuple(result.details.get("target_size") or ())
        template_size = tuple(result.details.get("template_size") or ())
        if len(target_size) == 2:
            try:
                tw = float(target_size[0])
                th = float(target_size[1])
                target_megapixels = (tw * th) / 1_000_000
            except Exception:
                target_megapixels = 0.0

        target_path = str(payload.get("target_image", payload.get("target", ""))).strip()
        if target_path:
            target_file = Path(target_path).expanduser()
            if not target_file.is_absolute():
                target_file = (Path.cwd() / target_file).resolve()
            try:
                target_file_size_mb = target_file.stat().st_size / 1024.0 / 1024.0
            except OSError:
                target_file_size_mb = 0.0

        total_sec = max(0.0, time.perf_counter() - started_at)
        timing_summary = {
            "total_sec": round(total_sec, 6),
            "feature_sec": round(feature_sec, 6),
            "template_sec": round(template_sec, 6),
            "template_calls": template_calls,
            "write_result_sec": round(write_result_sec, 6),
            "target_size": target_size,
            "template_size": template_size,
            "target_megapixels": round(target_megapixels, 6),
            "target_file_size_mb": round(target_file_size_mb, 6),
            "feature_similarity_threshold": float(feature_threshold),
            "template_similarity_threshold": float(template_threshold),
        }
        result.details["timing_breakdown"] = timing_summary
        _append_timing_csv(payload, result, timing_summary)
        _maybe_print_timing(
            payload,
            (
                "[image_match_timing] "
                f"backend={result.details.get('backend_used', 'unknown')} "
                f"total={timing_summary['total_sec']:.3f}s "
                f"feature={timing_summary['feature_sec']:.3f}s "
                f"template={timing_summary['template_sec']:.3f}s "
                f"write={timing_summary['write_result_sec']:.3f}s "
                f"template_calls={template_calls} "
                f"target_mp={timing_summary['target_megapixels']:.3f} "
                f"case={payload.get('timing_case_id', payload.get('case_id', ''))}"
            ),
        )
        return result

    backend = str(payload.get("match_backend", "auto")).strip().lower()
    fallback_to_template = bool(payload.get("fallback_to_template", True))
    auto_bbox_fusion = bool(payload.get("auto_bbox_fusion", True))
    auto_bbox_margin_ratio = float(payload.get("auto_bbox_margin_ratio", 0.02))

    # 阈值自动策略：
    # 1) template 维持历史口径（默认 0.65，或沿用 similarity_threshold）
    # 2) feature 使用单独阈值（默认 0.35）；若仅提供 similarity_threshold，则自动映射到 feature 口径
    has_legacy_threshold = "similarity_threshold" in payload
    legacy_threshold = float(payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))

    template_threshold = float(payload.get("template_similarity_threshold", legacy_threshold))
    if "feature_similarity_threshold" in payload:
        feature_threshold = float(payload["feature_similarity_threshold"])
    elif has_legacy_threshold:
        # 当用户只提供 template 口径阈值（常见为 0.8~0.9）时，自动映射到 feature 口径。
        feature_threshold = min(0.45, max(0.25, legacy_threshold * 0.4))
    else:
        # 与 image_match_checker_feature.py 默认保持一致
        feature_threshold = 0.35

    template_payload = dict(payload)
    template_payload["similarity_threshold"] = template_threshold

    feature_payload = dict(payload)
    feature_payload["similarity_threshold"] = feature_threshold

    if backend == "template":
        template_started_at = time.perf_counter()
        result = _check_image_match_template(template_payload, output=output)
        template_sec += max(0.0, time.perf_counter() - template_started_at)
        template_calls += 1
        result.details["backend_used"] = "template"
        result.details["threshold_strategy"] = {
            "mode": "template_only",
            "template_similarity_threshold": template_threshold,
        }
        return _finalize(result)

    # 延迟导入避免模块加载时循环依赖
    from .image_match_checker_feature import check_image_match_feature

    feature_started_at = time.perf_counter()
    feature_result = check_image_match_feature(feature_payload, output=output)
    feature_sec += max(0.0, time.perf_counter() - feature_started_at)
    feature_result.details["backend_used"] = "feature"
    feature_result.details["threshold_strategy"] = {
        "mode": "feature_only" if backend == "feature" else "auto",
        "feature_similarity_threshold": feature_threshold,
        "template_similarity_threshold": template_threshold,
        "legacy_similarity_threshold": legacy_threshold if has_legacy_threshold else None,
    }

    if backend == "feature":
        return _finalize(feature_result)

    # auto 模式下：若 feature 命中“强几何证据”，允许略低于阈值也通过（提高跨端/布局变化召回）。
    feature_stats = feature_result.details.get("feature_stats", {})
    feature_good = int(feature_stats.get("good_matches", 0) or 0)
    feature_inliers = int(feature_stats.get("inliers", 0) or 0)
    feature_h = bool(feature_stats.get("homography_found", False))
    feature_similarity = float(feature_result.details.get("similarity", 0.0) or 0.0)
    feature_inlier_ratio = (
        float(feature_inliers) / float(feature_good)
        if feature_good > 0
        else 0.0
    )
    strong_feature_min_inlier_ratio = float(payload.get("strong_feature_min_inlier_ratio", 0.35))
    strong_feature_gate = (
        feature_h
        and feature_good >= max(40, int(payload.get("min_good_matches", 12)) * 3)
        and feature_inliers >= max(12, int(payload.get("min_inliers", 8)) * 2)
        # Coverage-based similarity can be low for rich templates; rely more on geometry strength.
        and feature_similarity >= feature_threshold * 0.5
        and feature_inlier_ratio >= strong_feature_min_inlier_ratio
    )

    # 宽条模板（如标题栏/工具栏）在跨端缩放下常出现“少量但稳定”的特征匹配：
    # 覆盖率低会压低 similarity，但 homography + inlier ratio 仍有可信信号。
    template_size_for_gate = feature_result.details.get("template_size") or ()
    target_size_for_gate = feature_result.details.get("target_size") or ()
    template_aspect_for_gate = 0.0
    target_width_ratio_for_gate = 0.0
    if (
        isinstance(template_size_for_gate, (list, tuple))
        and len(template_size_for_gate) == 2
        and template_size_for_gate[0]
        and template_size_for_gate[1]
    ):
        tw_gate = float(template_size_for_gate[0])
        th_gate = float(template_size_for_gate[1])
        template_aspect_for_gate = max(tw_gate / max(th_gate, 1.0), th_gate / max(tw_gate, 1.0))
        if (
            isinstance(target_size_for_gate, (list, tuple))
            and len(target_size_for_gate) == 2
            and target_size_for_gate[0]
        ):
            target_width_ratio_for_gate = float(target_size_for_gate[0]) / max(tw_gate, 1.0)
    matched_w = float(feature_stats.get("matched_template_width", 0.0) or 0.0)
    matched_h = float(feature_stats.get("matched_template_height", 0.0) or 0.0)
    matched_scale_x = (
        matched_w / float(max(template_size_for_gate[0], 1))
        if isinstance(template_size_for_gate, (list, tuple)) and len(template_size_for_gate) == 2
        else 0.0
    )
    matched_scale_y = (
        matched_h / float(max(template_size_for_gate[1], 1))
        if isinstance(template_size_for_gate, (list, tuple)) and len(template_size_for_gate) == 2
        else 0.0
    )
    strip_feature_gate = (
        bool(payload.get("strip_feature_gate_enabled", True))
        and feature_h
        and feature_good >= int(payload.get("strip_feature_min_good_matches", 14))
        and feature_inliers >= int(payload.get("strip_feature_min_inliers", 8))
        and feature_inlier_ratio >= float(payload.get("strip_feature_min_inlier_ratio", 0.5))
        and feature_similarity >= feature_threshold * float(payload.get("strip_feature_similarity_ratio", 0.75))
        and template_aspect_for_gate >= float(payload.get("strip_feature_min_template_aspect", 4.0))
        and target_width_ratio_for_gate >= float(payload.get("strip_feature_min_target_width_ratio", 1.8))
        and 0.6 <= matched_scale_x <= 1.4
        and 0.5 <= matched_scale_y <= 1.8
    )
    if (strong_feature_gate or strip_feature_gate) and not feature_result.passed:
        feature_result.passed = True
        if strong_feature_gate:
            feature_result.basis = (
                "image_match_feature: Passed by strong geometry gate. "
                f"similarity={feature_similarity:.4f}, good_matches={feature_good}, "
                f"inliers={feature_inliers}, inlier_ratio={feature_inlier_ratio:.3f}"
            )
            feature_result.details["strong_feature_gate"] = True
        else:
            feature_result.basis = (
                "image_match_feature: Passed by strip geometry gate. "
                f"similarity={feature_similarity:.4f}, good_matches={feature_good}, "
                f"inliers={feature_inliers}, inlier_ratio={feature_inlier_ratio:.3f}, "
                f"template_aspect={template_aspect_for_gate:.2f}"
            )
            feature_result.details["strip_feature_gate"] = {
                "applied": True,
                "template_aspect": template_aspect_for_gate,
                "target_width_ratio": target_width_ratio_for_gate,
                "matched_scale_x": matched_scale_x,
                "matched_scale_y": matched_scale_y,
            }
        # 继续走后续 auto 流程，以便做 bbox 融合优化（不直接 return）。

    # auto: 优先 feature；失败后可回退到模板匹配
    if feature_result.passed:
        if auto_bbox_fusion:
            template_started_at = time.perf_counter()
            template_result_for_fusion = _check_image_match_template(template_payload, output=output)
            template_sec += max(0.0, time.perf_counter() - template_started_at)
            template_calls += 1
            f_bounds = feature_result.details.get("bounds")
            t_bounds = template_result_for_fusion.details.get("bounds")
            t_similarity = float(template_result_for_fusion.details.get("similarity", 0.0) or 0.0)
            t_size = feature_result.details.get("template_size") or ()
            g_size = feature_result.details.get("target_size") or ()
            f_stats = feature_result.details.get("feature_stats", {})
            f_good = int(f_stats.get("good_matches", 0) or 0)
            if (
                isinstance(f_bounds, (list, tuple))
                and len(f_bounds) == 4
                and isinstance(t_bounds, (list, tuple))
                and len(t_bounds) == 4
                and isinstance(t_size, (list, tuple))
                and len(t_size) == 2
                and isinstance(g_size, (list, tuple))
                and len(g_size) == 2
                and t_similarity >= 0.30
                and f_good >= 80
            ):
                tw, th = float(t_size[0]), float(t_size[1])
                gw, gh = float(g_size[0]), float(g_size[1])
                aspect = max(tw / max(th, 1.0), th / max(tw, 1.0))
                width_ratio = gw / max(tw, 1.0)
                if aspect >= 2.0 and width_ratio >= 1.8:
                    margin = int(round(gw * max(0.0, min(auto_bbox_margin_ratio, 0.2))))
                    fused = (
                        margin,
                        int(t_bounds[1]),
                        int(gw) - margin,
                        int(t_bounds[3]),
                    )
                    feature_result.details["bounds_before_fusion"] = tuple(feature_result.details.get("bounds", (0, 0, 0, 0)))
                    feature_result.details["bounds"] = fused
                    feature_result.control_info.bounds = Bounds.from_sequence(fused)
                    feature_result.details["auto_bbox_fusion"] = {
                        "applied": True,
                        "margin_ratio": auto_bbox_margin_ratio,
                        "template_similarity": t_similarity,
                        "template_bounds": t_bounds,
                    }
                    feature_result.basis = (
                        f"{feature_result.basis}; fused_bbox={feature_result.control_info.bounds.as_box()} "
                        f"(template_y + full-width margin)"
                    )
                else:
                    feature_result.details["auto_bbox_fusion"] = {
                        "applied": False,
                        "reason": "aspect_or_width_ratio_not_satisfied",
                        "aspect": aspect,
                        "width_ratio": width_ratio,
                    }
            else:
                feature_result.details["auto_bbox_fusion"] = {
                    "applied": False,
                    "reason": "insufficient_fusion_signal",
                }
        return _finalize(feature_result)

    if not fallback_to_template:
        return _finalize(feature_result)

    template_started_at = time.perf_counter()
    template_result = _check_image_match_template(template_payload, output=output)
    template_sec += max(0.0, time.perf_counter() - template_started_at)
    template_calls += 1
    template_result.details["backend_used"] = "template_fallback"
    template_result.details["feature_failed_basis"] = feature_result.basis
    template_result.details["feature_similarity"] = feature_result.details.get("similarity")
    template_result.details["threshold_strategy"] = feature_result.details.get("threshold_strategy")

    # auto 模式默认：若用户未显式给 template 阈值，可按场景自适应放宽。
    # 典型场景：横向长条组件（高宽比很大）在跨端布局中像素差异明显，原阈值过高会漏报。
    template_similarity = float(template_result.details.get("similarity", 0.0) or 0.0)
    if "template_similarity_threshold" not in payload:
        template_size = template_result.details.get("template_size") or ()
        feature_stats_for_relax = feature_result.details.get("feature_stats", {})
        feature_similarity_for_relax = float(feature_result.details.get("similarity", 0.0) or 0.0)
        auto_relax_min_feature_similarity = float(payload.get("auto_relax_min_feature_similarity", 0.22))
        relax_by_feature = (
            bool(feature_stats_for_relax.get("homography_found", False))
            and int(feature_stats_for_relax.get("good_matches", 0) or 0) >= 25
            and int(feature_stats_for_relax.get("inliers", 0) or 0) >= 8
            and feature_similarity_for_relax >= auto_relax_min_feature_similarity
        )
        template_best_scale = float(template_result.details.get("best_scale", 0.0) or 0.0)
        scale_step = float((template_result.details.get("scale_range", {}) or {}).get("step", 0.05) or 0.05)
        boundary_meta = template_result.details.get("boundary_guard", {}) or {}
        effective_scale_min = float(boundary_meta.get("effective_scale_min", 0.0) or 0.0)
        at_lower_scale = abs(template_best_scale - effective_scale_min) <= max(scale_step / 2.0, 1e-6)
        relax_allowed = relax_by_feature
        auto_relaxed_template_threshold_min = float(
            payload.get("auto_relaxed_template_threshold_min", 0.45)
        )
        relaxed_template_threshold = max(
            auto_relaxed_template_threshold_min,
            min(template_threshold * 0.6, 0.55),
        )
        if (
            isinstance(template_size, (list, tuple))
            and len(template_size) == 2
            and template_size[0]
            and template_size[1]
        ):
            w = float(template_size[0])
            h = float(template_size[1])
            aspect = max(w / h, h / w)
            target_size = template_result.details.get("target_size") or ()
            width_ratio = 0.0
            if isinstance(target_size, (list, tuple)) and len(target_size) == 2 and target_size[0]:
                width_ratio = float(target_size[0]) / max(w, 1.0)
            # 对超长条UI再放宽一点（如标题栏、工具栏），减少布局变形漏报
            # Long strip templates often vary across devices (fonts/spacing). Relax more aggressively.
            if aspect >= 5.0:
                relaxed_template_threshold = min(relaxed_template_threshold, 0.36)
            elif aspect >= 4.0:
                relaxed_template_threshold = min(relaxed_template_threshold, 0.40)
            template_result.details["template_aspect_ratio"] = aspect
            template_result.details["template_width_ratio"] = width_ratio
            relax_by_template = (
                template_similarity >= 0.35
                and template_best_scale >= 0.55
                and not at_lower_scale
                and aspect >= 3.0
                and width_ratio >= 1.8
            )
            relax_allowed = relax_allowed or relax_by_template
        template_result.details["auto_relaxed_template_threshold"] = relaxed_template_threshold
        template_result.details["auto_relaxed_template_context"] = {
            "relax_by_feature": relax_by_feature,
            "template_best_scale": template_best_scale,
            "effective_scale_min": effective_scale_min,
            "at_lower_scale": at_lower_scale,
        }
        template_result.details["auto_relaxed_template_allowed"] = relax_allowed
        if relax_allowed and template_similarity >= relaxed_template_threshold and not template_result.passed:
            template_result.passed = True
            template_result.basis = (
                "image_match: Passed by auto-relaxed template threshold. "
                f"similarity={template_similarity:.4f}, relaxed_threshold={relaxed_template_threshold:.4f}"
            )
            template_result.details["passed_by_relaxed_template_threshold"] = True

    # auto + offscale 边缘放行：
    # 仅在“接近 offscale 阈值、且有最低 feature 支持、且缩放不极端”时放行，尽量避免引入误报。
    auto_offscale_near_threshold_relax = bool(payload.get("auto_offscale_near_threshold_relax", True))
    if auto_offscale_near_threshold_relax and not template_result.passed:
        rejection_reason = str(template_result.details.get("rejection_reason") or "")
        off_meta = template_result.details.get("offscale_guard", {}) or {}
        boundary_meta = template_result.details.get("boundary_guard", {}) or {}
        best_scale = float(template_result.details.get("best_scale", 0.0) or 0.0)
        scale_step = float((template_result.details.get("scale_range", {}) or {}).get("step", 0.05) or 0.05)
        effective_scale_min = float(boundary_meta.get("effective_scale_min", 0.0) or 0.0)
        at_lower_scale = abs(best_scale - effective_scale_min) <= max(scale_step / 2.0, 1e-6)
        feature_good = int((feature_result.details.get("feature_stats", {}) or {}).get("good_matches", 0) or 0)
        feature_sim = float(feature_result.details.get("similarity", 0.0) or 0.0)
        min_template_similarity = float(payload.get("auto_offscale_min_template_similarity", 0.8))
        gap_max = float(payload.get("auto_offscale_gap_max", 0.05))
        scale_min = float(payload.get("auto_offscale_scale_min", 0.75))
        scale_max = float(payload.get("auto_offscale_scale_max", 1.6))
        min_feature_good = int(payload.get("auto_offscale_min_feature_good_matches", 4))
        min_feature_similarity = float(payload.get("auto_offscale_min_feature_similarity", 0.01))
        off_threshold = off_meta.get("threshold")
        if off_threshold is None:
            off_threshold = (
                template_threshold
                + float(off_meta.get("similarity_margin", 0.0) or 0.0)
                + float(off_meta.get("dynamic_extra", 0.0) or 0.0)
            )
        off_threshold = float(off_threshold)
        off_gap = off_threshold - template_similarity
        can_relax = (
            rejection_reason == "offscale_low_confidence"
            and template_similarity >= min_template_similarity
            and off_gap >= 0.0
            and off_gap <= gap_max
            and scale_min <= best_scale <= scale_max
            and not at_lower_scale
            and (feature_good >= min_feature_good or feature_sim >= min_feature_similarity)
        )
        template_result.details["auto_offscale_relax"] = {
            "enabled": True,
            "can_relax": can_relax,
            "offscale_threshold": off_threshold,
            "offscale_gap": off_gap,
            "best_scale": best_scale,
            "at_lower_scale": at_lower_scale,
            "feature_good_matches": feature_good,
            "feature_similarity": feature_sim,
            "constraints": {
                "min_template_similarity": min_template_similarity,
                "gap_max": gap_max,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "min_feature_good_matches": min_feature_good,
                "min_feature_similarity": min_feature_similarity,
            },
        }
        if can_relax:
            template_result.passed = True
            template_result.basis = (
                "image_match: Passed by auto-offscale near-threshold relax. "
                f"similarity={template_similarity:.4f}, offscale_threshold={off_threshold:.4f}, gap={off_gap:.4f}"
            )
            template_result.details["passed_by_auto_offscale_relax"] = True

    if template_result.passed:
        return _finalize(template_result)

    # 两者都失败时默认返回 feature 结果，并附上模板结果摘要，便于调参诊断。
    feature_result.details["template_fallback_attempted"] = True
    feature_result.details["template_fallback_passed"] = False
    feature_result.details["template_similarity"] = template_result.details.get("similarity")
    feature_result.details["template_basis"] = template_result.basis
    return _finalize(feature_result)
