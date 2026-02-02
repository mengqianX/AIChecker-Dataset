from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageStat, ImageOps

from ..models import Bounds, ControlInfo, CheckResult
from ..utils import dominant_color


def _knob_position_ratio(crop: Image.Image) -> float:
    """Estimate knob x-position ratio within crop by thresholding brighter pixels."""
    gray = ImageOps.autocontrast(crop.convert("L"))
    stat = ImageStat.Stat(gray)
    threshold = min(255, stat.mean[0] + 15)  # slight bias toward brighter knob
    binary = gray.point(lambda p: 255 if p > threshold else 0)
    w, h = binary.size
    px = binary.load()
    xs = []
    for y in range(h):
        for x in range(w):
            if px[x, y] == 255:
                xs.append(x)
    if not xs:
        return 0.5
    return sum(xs) / len(xs) / float(w)


def _half_brightness(gray: Image.Image) -> Tuple[float, float]:
    """Brightness mean for left/right halves of grayscale crop."""
    w, _ = gray.size
    mid = w // 2
    left = gray.crop((0, 0, mid, gray.height))
    right = gray.crop((mid, 0, w, gray.height))
    return ImageStat.Stat(left).mean[0], ImageStat.Stat(right).mean[0]


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

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        crop_before.save(debug_dir / "crop_before.png")
        crop_after.save(debug_dir / "crop_after.png")

    dom_after = dominant_color(crop_after)
    knob_ratio_after = _knob_position_ratio(crop_after)
    knob_ratio_before = _knob_position_ratio(crop_before)

    gray_after = ImageOps.autocontrast(crop_after.convert("L"))
    left_b, right_b = _half_brightness(gray_after)

    inferred_on = knob_ratio_after > 0.55 or right_b - left_b > 8

    control = replace(control, main_color=dom_after, checked=inferred_on, source="cv")
    passed = inferred_on is expected
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
