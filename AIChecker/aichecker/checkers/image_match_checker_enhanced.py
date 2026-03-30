"""
增强版图片匹配Checker - 包含学术创新点

主要改进：
1. 自适应尺度范围计算
2. 分层搜索策略
3. UI元素特征增强
4. 多模板融合
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import load_image

# 默认参数
DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_SCALE_MIN = 0.5
DEFAULT_SCALE_MAX = 2.0
DEFAULT_SCALE_STEP = 0.1
DEFAULT_MATCH_METHOD = cv2.TM_CCOEFF_NORMED

# DPI映射（常见Android设备）
DPI_MAPPING = {
    'mdpi': 160,
    'hdpi': 240,
    'xhdpi': 320,
    'xxhdpi': 480,
    'xxxhdpi': 560,
}


def _compute_adaptive_scale_range(
    template_size: Tuple[int, int],
    target_size: Tuple[int, int],
    device_dpi: Optional[int] = None,
    template_dpi: Optional[int] = None,
) -> Tuple[float, float]:
    """
    自适应尺度范围计算（创新点1）
    
    根据设备DPI和图像尺寸自适应计算尺度范围，而不是使用固定值。
    
    Args:
        template_size: 模板图片尺寸 (width, height)
        target_size: 目标图片尺寸 (width, height)
        device_dpi: 目标设备的DPI（可选）
        template_dpi: 模板图片来源设备的DPI（可选，默认240）
    
    Returns:
        (scale_min, scale_max): 自适应计算的尺度范围
    """
    template_w, template_h = template_size
    target_w, target_h = target_size
    
    # 基础尺寸比率
    base_ratio_w = target_w / template_w
    base_ratio_h = target_h / template_h
    base_ratio = min(base_ratio_w, base_ratio_h)  # 取较小值，确保不超出边界
    
    # 如果提供了DPI信息，进行DPI校正
    if device_dpi and template_dpi:
        dpi_ratio = device_dpi / template_dpi
        base_ratio *= dpi_ratio
    
    # 自适应范围：在基础比率附近搜索
    # 允许50%的缩小和200%的放大（考虑UI元素可能的尺寸变化）
    scale_min = max(0.3, base_ratio * 0.5)
    scale_max = min(5.0, base_ratio * 2.0)
    
    return scale_min, scale_max


def _hierarchical_template_match(
    template: np.ndarray,
    target: np.ndarray,
    scale_min: float,
    scale_max: float,
    method: int = DEFAULT_MATCH_METHOD,
) -> Tuple[float, Tuple[int, int, int, int], float]:
    """
    分层搜索策略（创新点2）
    
    先用粗尺度快速定位，再用细尺度精确定位，减少计算量。
    
    Args:
        template: 模板图片
        target: 目标图片
        scale_min: 最小缩放比例
        scale_max: 最大缩放比例
        method: 匹配方法
    
    Returns:
        (相似度, bounds, 最佳缩放比例)
    """
    template_h, template_w = template.shape[:2]
    target_h, target_w = target.shape[:2]
    
    # 第一阶段：粗搜索（大步长，快速定位）
    coarse_step = 0.2  # 粗搜索步长
    coarse_scales = np.arange(scale_min, scale_max + coarse_step, coarse_step)
    
    best_coarse_score = -np.inf if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else np.inf
    best_coarse_scale = 1.0
    best_coarse_loc = None
    
    for scale in coarse_scales:
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)
        
        if scaled_w < 1 or scaled_h < 1 or scaled_w > target_w or scaled_h > target_h:
            continue
        
        scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(target, scaled_template, method)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            match_val = 1.0 - min_val if method == cv2.TM_SQDIFF_NORMED else -min_val
            match_loc = min_loc
        else:
            match_val = max_val
            match_loc = max_loc
        
        if match_val > best_coarse_score:
            best_coarse_score = match_val
            best_coarse_scale = scale
            best_coarse_loc = match_loc
    
    if best_coarse_loc is None:
        return 0.0, (0, 0, 0, 0), 1.0
    
    # 第二阶段：细搜索（小步长，在最佳粗尺度附近精确定位）
    fine_step = 0.05  # 细搜索步长
    fine_range = 0.3  # 在最佳尺度附近±0.3范围内搜索
    fine_min = max(scale_min, best_coarse_scale - fine_range)
    fine_max = min(scale_max, best_coarse_scale + fine_range)
    fine_scales = np.arange(fine_min, fine_max + fine_step, fine_step)
    
    best_fine_score = best_coarse_score
    best_fine_scale = best_coarse_scale
    best_fine_loc = best_coarse_loc
    best_fine_size = (int(template_w * best_coarse_scale), int(template_h * best_coarse_scale))
    
    for scale in fine_scales:
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)
        
        if scaled_w < 1 or scaled_h < 1 or scaled_w > target_w or scaled_h > target_h:
            continue
        
        scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(target, scaled_template, method)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            match_val = 1.0 - min_val if method == cv2.TM_SQDIFF_NORMED else -min_val
            match_loc = min_loc
        else:
            match_val = max_val
            match_loc = max_loc
        
        if match_val > best_fine_score:
            best_fine_score = match_val
            best_fine_scale = scale
            best_fine_loc = match_loc
            best_fine_size = (scaled_w, scaled_h)
    
    # 计算bounds
    left = best_fine_loc[0]
    top = best_fine_loc[1]
    right = left + best_fine_size[0]
    bottom = top + best_fine_size[1]
    
    # 确保bounds在目标图片范围内
    left = max(0, min(left, target_w))
    top = max(0, min(top, target_h))
    right = max(left, min(right, target_w))
    bottom = max(top, min(bottom, target_h))
    
    return best_fine_score, (left, top, right, bottom), best_fine_scale


def _enhance_ui_template(template: np.ndarray, enhancement_type: str = "edges") -> np.ndarray:
    """
    UI元素特征增强（创新点3）
    
    针对UI元素的特殊预处理，提高匹配鲁棒性。
    
    Args:
        template: 原始模板图片
        enhancement_type: 增强类型
            - "edges": 边缘增强（适合有清晰边界的UI元素）
            - "lab": LAB颜色空间（对光照变化更鲁棒）
            - "gray": 灰度图（忽略颜色，关注形状）
            - "original": 原始图片
    
    Returns:
        增强后的模板图片
    """
    if enhancement_type == "edges":
        # 边缘增强：UI元素通常有清晰边界
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # 转换回BGR以便匹配
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    elif enhancement_type == "lab":
        # LAB颜色空间：对光照变化更鲁棒
        return cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
    
    elif enhancement_type == "gray":
        # 灰度图：忽略颜色，关注形状
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    else:
        # 原始图片
        return template


def _multi_template_match(
    templates: List[np.ndarray],
    target: np.ndarray,
    scale_min: float,
    scale_max: float,
    method: int = DEFAULT_MATCH_METHOD,
) -> Tuple[float, Tuple[int, int, int, int], float, int]:
    """
    多模板融合（创新点4）
    
    使用多个模板（不同角度、光照、版本）进行匹配，提高鲁棒性。
    
    Args:
        templates: 模板图片列表
        target: 目标图片
        scale_min: 最小缩放比例
        scale_max: 最大缩放比例
        method: 匹配方法
    
    Returns:
        (最佳相似度, bounds, 最佳缩放比例, 最佳模板索引)
    """
    best_overall_score = -np.inf if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else np.inf
    best_overall_result = None
    best_template_idx = 0
    
    for idx, template in enumerate(templates):
        score, bounds, scale = _hierarchical_template_match(
            template, target, scale_min, scale_max, method
        )
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_result = (score, bounds, scale)
            best_template_idx = idx
    
    if best_overall_result is None:
        return 0.0, (0, 0, 0, 0), 1.0, 0
    
    return (*best_overall_result, best_template_idx)


def check_image_match_enhanced(
    payload: Dict[str, Any],
    debug_dir: Path | None = None,
) -> CheckResult:
    """
    增强版图片匹配检测器
    
    包含以下创新点：
    1. 自适应尺度范围计算
    2. 分层搜索策略
    3. UI元素特征增强
    4. 多模板融合（如果提供多个模板）
    
    Args:
        payload: 包含以下字段的字典：
            - template_image: 模板图片路径（或列表，支持多模板）
            - target_image: 目标图片路径
            - similarity_threshold: 相似度阈值（默认0.8）
            - device_dpi: 目标设备DPI（可选）
            - template_dpi: 模板来源设备DPI（可选，默认240）
            - enhancement_type: 增强类型（可选，默认"original"）
            - use_hierarchical: 是否使用分层搜索（默认True）
            - scale_min, scale_max, scale_step: 手动指定缩放范围（可选）
        debug_dir: 调试输出目录
    
    Returns:
        CheckResult对象
    """
    from .image_match_checker import _resolve_template_image, _resolve_target_image, _pil_to_cv2
    
    # 解析输入
    template_paths = payload.get("template_image")
    if isinstance(template_paths, str):
        template_paths = [template_paths]
    elif not isinstance(template_paths, list):
        template_paths = [_resolve_template_image(payload)]
    
    target_path = _resolve_target_image(payload)
    
    similarity_threshold = float(payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    device_dpi = payload.get("device_dpi")
    template_dpi = payload.get("template_dpi", 240)  # 默认hdpi
    enhancement_type = payload.get("enhancement_type", "original")
    use_hierarchical = payload.get("use_hierarchical", True)
    
    # 加载图片
    templates_pil = [load_image(path) for path in template_paths]
    target_pil = load_image(target_path)
    
    # 转换为OpenCV格式
    templates_cv = [_pil_to_cv2(img) for img in templates_pil]
    target_cv = _pil_to_cv2(target_pil)
    
    # UI元素特征增强
    if enhancement_type != "original":
        templates_cv = [_enhance_ui_template(t, enhancement_type) for t in templates_cv]
        target_cv = _enhance_ui_template(target_cv, enhancement_type)
    
    # 自适应尺度范围计算
    if payload.get("scale_min") is not None:
        scale_min = float(payload["scale_min"])
        scale_max = float(payload["scale_max"])
    else:
        template_size = templates_pil[0].size
        target_size = target_pil.size
        scale_min, scale_max = _compute_adaptive_scale_range(
            template_size, target_size, device_dpi, template_dpi
        )
    
    scale_step = float(payload.get("scale_step", DEFAULT_SCALE_STEP))
    
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
    
    # 执行匹配
    if len(templates_cv) > 1:
        # 多模板融合
        similarity, bounds_tuple, best_scale, best_template_idx = _multi_template_match(
            templates_cv, target_cv, scale_min, scale_max, match_method
        )
    elif use_hierarchical:
        # 分层搜索
        similarity, bounds_tuple, best_scale = _hierarchical_template_match(
            templates_cv[0], target_cv, scale_min, scale_max, match_method
        )
        best_template_idx = 0
    else:
        # 标准多尺度匹配（使用原始实现）
        from .image_match_checker import _multi_scale_template_match
        similarity, bounds_tuple, best_scale = _multi_scale_template_match(
            templates_cv[0], target_cv, scale_min, scale_max, scale_step, match_method
        )
        best_template_idx = 0
    
    # 判断是否匹配成功
    passed = similarity >= similarity_threshold
    
    # 构建bounds
    bounds = Bounds.from_sequence(bounds_tuple)
    
    # 构建basis
    enhancement_info = f", enhancement={enhancement_type}" if enhancement_type != "original" else ""
    hierarchical_info = ", hierarchical_search" if use_hierarchical else ""
    multi_template_info = f", best_template_idx={best_template_idx}" if len(templates_cv) > 1 else ""
    
    if passed:
        basis = (
            f"image_match_enhanced: Found template in target image with similarity={similarity:.4f} "
            f"(threshold={similarity_threshold:.4f}), bounds={bounds.as_box()}, "
            f"scale={best_scale:.2f}{enhancement_info}{hierarchical_info}{multi_template_info}"
        )
    else:
        basis = (
            f"image_match_enhanced: Template not found. "
            f"Best similarity={similarity:.4f} < threshold={similarity_threshold:.4f}, "
            f"best_match_bounds={bounds.as_box()}, scale={best_scale:.2f}"
        )
    
    # 构建ControlInfo
    control_info = ControlInfo(
        bounds=bounds,
        source="image_match_enhanced",
        extras={
            "similarity": similarity,
            "best_scale": best_scale,
            "template_size": templates_pil[0].size,
            "target_size": target_pil.size,
            "enhancement_type": enhancement_type,
            "use_hierarchical": use_hierarchical,
            "best_template_idx": best_template_idx if len(templates_cv) > 1 else None,
        },
    )
    
    # 构建details
    details: Dict[str, Any] = {
        "method": "enhanced_multi_scale_template_match",
        "similarity": similarity,
        "similarity_threshold": similarity_threshold,
        "bounds": bounds_tuple,
        "best_scale": best_scale,
        "template_paths": template_paths,
        "target_path": target_path,
        "template_size": templates_pil[0].size,
        "target_size": target_pil.size,
        "match_method": match_method_str,
        "scale_range": {
            "min": scale_min,
            "max": scale_max,
            "adaptive": payload.get("scale_min") is None,
        },
        "enhancement_type": enhancement_type,
        "use_hierarchical": use_hierarchical,
        "device_dpi": device_dpi,
        "template_dpi": template_dpi,
    }
    
    if len(templates_cv) > 1:
        details["num_templates"] = len(templates_cv)
        details["best_template_idx"] = best_template_idx
    
    # 调试输出
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        target_vis = target_cv.copy()
        cv2.rectangle(
            target_vis,
            (bounds.left, bounds.top),
            (bounds.right, bounds.bottom),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            target_vis,
            f"Similarity: {similarity:.3f}, Scale: {best_scale:.2f}",
            (bounds.left, bounds.top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(debug_dir / "match_result.png"), target_vis)
    
    return CheckResult(
        passed=passed,
        basis=basis,
        control_info=control_info,
        details=details,
    )
