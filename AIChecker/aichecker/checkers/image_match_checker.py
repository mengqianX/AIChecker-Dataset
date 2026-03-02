from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import load_image

# 默认相似度阈值（真实截图常因分辨率/压缩/缩放差异导致 TM_CCOEFF_NORMED 在 0.5–0.8，故默认放宽以提升召回）
DEFAULT_SIMILARITY_THRESHOLD = 0.65
# 默认缩放范围（用于多尺度匹配，放宽以覆盖不同 DPI/分辨率）
DEFAULT_SCALE_MIN = 0.3
DEFAULT_SCALE_MAX = 3.0
DEFAULT_SCALE_STEP = 0.05
# 默认匹配方法
DEFAULT_MATCH_METHOD = cv2.TM_CCOEFF_NORMED


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
    
    scales = np.arange(actual_scale_min, actual_scale_max + scale_step, scale_step)
    
    for scale in scales:
        # 缩放模板
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)
        
        if scaled_w < 1 or scaled_h < 1 or scaled_w > target_w or scaled_h > target_h:
            continue
        
        scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        
        # 执行模板匹配
        result = cv2.matchTemplate(target, scaled_template, method)
        
        # 根据匹配方法找到最佳匹配位置
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


def check_image_match(
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
        debug_dir: 调试输出目录（可选）
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
    boundary_guard = bool(payload.get("boundary_guard", True))
    boundary_similarity_margin = float(payload.get("boundary_similarity_margin", 0.12))
    offscale_guard = bool(payload.get("offscale_guard", True))
    offscale_min_deviation = float(payload.get("offscale_min_deviation", 0.15))
    offscale_similarity_margin = float(payload.get("offscale_similarity_margin", 0.08))
    offscale_extra_per_unit = float(payload.get("offscale_extra_per_unit", 0.25))
    offscale_max_extra = float(payload.get("offscale_max_extra", 0.20))
    
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
        boundary_threshold = similarity_threshold + boundary_similarity_margin
        if (at_lower or at_upper) and similarity < boundary_threshold:
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
        offscale_threshold = similarity_threshold + offscale_similarity_margin + dynamic_extra
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
            "similarity_margin": boundary_similarity_margin,
            "effective_scale_min": effective_scale_min,
            "effective_scale_max": effective_scale_max,
        },
        "offscale_guard": {
            "enabled": offscale_guard,
            "rejected": rejection_reason == "offscale_low_confidence",
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
    
    return CheckResult(
        passed=passed,
        basis=basis,
        control_info=control_info,
        details=details,
    )
