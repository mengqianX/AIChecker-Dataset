from __future__ import annotations

import colorsys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageStat, ImageOps
import numpy as np

from ..models import Bounds, ControlInfo, CheckResult
from ..utils import dominant_color

SAT_DELTA_THRESHOLD = 0.05
SAT_OFF = 0.02
KNOB_DELTA_THRESHOLD = 0.08

# def _knob_position_ratio(crop: Image.Image) -> float:
#     """Estimate knob x-position ratio within crop by thresholding brighter pixels."""
#     gray = ImageOps.autocontrast(crop.convert("L"))
#     stat = ImageStat.Stat(gray)
#     threshold = min(255, stat.mean[0] + 15)  # slight bias toward brighter knob
#     binary = gray.point(lambda p: 255 if p > threshold else 0)
#     w, h = binary.size
#     px = binary.load()
#     xs = []
#     for y in range(h):
#         for x in range(w):
#             if px[x, y] == 255:
#                 xs.append(x)
#     if not xs:
#         return 0.5
#     return sum(xs) / len(xs) / float(w)

def _knob_position_ratio_peak(crop: Image.Image) -> float:
    """
    更稳的滑块位置估计：用亮像素列投影找峰值（结构特征）。
    若亮像素太少，则尝试找“暗滑块”。
    """
    gray = ImageOps.autocontrast(crop.convert("L"))
    arr = np.array(gray, dtype=np.uint8)

    thr = min(255, float(arr.mean()) + 15.0)
    mask = arr > thr

    # 若亮像素太少，尝试检测暗滑块
    if mask.sum() < 10:
        thr_dark = max(0, float(arr.mean()) - 15.0)
        mask = arr < thr_dark

    if mask.sum() == 0:
        return 0.5

    col_sum = mask.sum(axis=0)  # 每列亮/暗像素数量
    top_k = max(1, int(len(col_sum) * 0.15))  # 取前 15% 列
    idx = np.argpartition(col_sum, -top_k)[-top_k:]
    weights = col_sum[idx].astype(float)
    if weights.sum() == 0:
        return 0.5

    center = (idx * weights).sum() / weights.sum()
    return float(center) / float(mask.shape[1])



def _half_brightness(gray: Image.Image) -> Tuple[float, float]:
    """Brightness mean for left/right halves of grayscale crop."""
    w, _ = gray.size
    mid = w // 2
    left = gray.crop((0, 0, mid, gray.height))
    right = gray.crop((mid, 0, w, gray.height))
    return ImageStat.Stat(left).mean[0], ImageStat.Stat(right).mean[0]

def _mean_saturation(img: Image.Image) -> float:
    """Mean saturation in HSV (0-1)."""
    small = img.convert("RGB").resize((24, 24))
    px = small.getdata()
    s_sum = 0.0
    for r, g, b in px:
        # colorsys.rgb_to_hsv returns (h, s, v)
        s_sum += colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
    return s_sum / (24 * 24)


def check_toggle_uitree(control: ControlInfo, expected_on: bool) -> CheckResult:
    return CheckResult(
        passed=control.checked is expected_on,
        basis=f"uitree checked={control.checked}, expected={expected_on}",
        control_info=control,
        details={"method": "uitree"},
    )


def check_toggle_cv(
    img_before: Image.Image,
    img_after: Image.Image,
    bounds: Bounds,
    expected: bool,
    meta: ControlInfo | None = None,
    debug_dir: Optional[Path] = None,
) -> CheckResult:
    """CV-based toggle state estimation."""
    control = meta or ControlInfo(bounds=bounds, source="cv")
    l, t, r, b = bounds.as_box()
    w, h = img_after.size
    if l < 0 or t < 0 or r > w or b > h:
        raise ValueError(
            f"Bounds out of image: bounds={bounds.as_box()} image_size={(w, h)}"
        )
    crop_after = img_after.crop(bounds.as_box())
    crop_before = img_before.crop(bounds.as_box())

    sat_after = _mean_saturation(crop_after)
    sat_before = _mean_saturation(crop_before)
    delta_sat = sat_after - sat_before

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        crop_before.save(debug_dir / "crop_before.png")
        crop_after.save(debug_dir / "crop_after.png")

    dom_after = dominant_color(crop_after)
    knob_ratio_after = _knob_position_ratio_peak(crop_after)
    knob_ratio_before = _knob_position_ratio_peak(crop_before)
    delta_knob = knob_ratio_after - knob_ratio_before


    gray_after = ImageOps.autocontrast(crop_after.convert("L"))
    left_b, right_b = _half_brightness(gray_after)

    # 饱和度变化
    if abs(delta_sat) >= SAT_DELTA_THRESHOLD:
        inferred_on = delta_sat > 0
    elif sat_after <= SAT_OFF:
        inferred_on = False
    # 滑块位移
    elif abs(delta_knob) >= KNOB_DELTA_THRESHOLD:
        inferred_on = delta_knob > 0
    else:
        if knob_ratio_after > 0.55:
            inferred_on = True
        elif knob_ratio_after < 0.45:
            inferred_on = False
        else:
            inferred_on = (right_b - left_b) > 6
    
    inferred_on = bool(inferred_on)

    control = replace(control, main_color=dom_after, checked=inferred_on, source="cv")
    passed = inferred_on == expected
    basis = (
        f"cv knob_ratio_after={knob_ratio_after:.2f} "
        f"left_b={left_b:.1f} right_b={right_b:.1f} expected={expected}"
    )
    details = {
        "method": "cv",
        "knob_ratio_after": knob_ratio_after,
        "knob_ratio_before": knob_ratio_before,
        "left_brightness": left_b,
        "right_brightness": right_b,
        "dominant_color_after": dom_after,
    }

    return CheckResult(passed=passed, basis=basis, control_info=control, details=details)
