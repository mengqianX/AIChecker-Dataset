from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple
import numpy as np
# from sklearn.cluster import KMeans # type: ignore
from PIL import Image, ImageStat

from aichecker.models import CheckResult, ControlInfo


def load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Screenshot not found: {path}")
    return Image.open(p)


def clamp_channel(v: int) -> int:
    return max(0, min(255, v))


def parse_color(value: Any) -> Tuple[int, int, int]:
    """Parse color from hex string, comma string, or RGB list/tuple."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(clamp_channel(int(v)) for v in value)  # type: ignore[arg-type]
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw.startswith("#"):
            raw = raw[1:]
        if raw.startswith("0x"):
            raw = raw[2:]
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) == 3:
                return tuple(clamp_channel(int(p)) for p in parts)  # type: ignore[arg-type]
        if len(raw) in (6, 8):
            hex_part = raw[:6]
            r, g, b = int(hex_part[0:2], 16), int(hex_part[2:4], 16), int(hex_part[4:6], 16)
            return (r, g, b)
    raise ValueError(f"Unsupported color value: {value!r}")


def dominant_color(img: Image.Image) -> Tuple[int, int, int]:
    """Return the most common RGB color in a downsampled image."""
    small = img.convert("RGB").resize((24, 24))
    colors = small.getcolors(24 * 24)
    if not colors:
        return (0, 0, 0)
    colors.sort(key=lambda c: c[0], reverse=True)
    return colors[0][1]

def dominant_button_color(
    img: Image.Image,
    bright_threshold: int = 240,  # 亮度阈值，越大过滤越“白”的像素
) -> Tuple[int, int, int]:
    """
    返回更适合“按钮区域”的主色：
    - 先过滤掉接近白色的高亮前景（文字 / 图标）
    - 再从剩余像素中取出现次数最多的颜色
    """
    small = img.convert("RGB").resize((24, 24))
    colors = small.getcolors(24 * 24)
    if not colors:
        return (0, 0, 0)

    # 过滤掉“太亮”的颜色（比如白色图标、文字）
    def is_not_too_bright(rgb: Tuple[int, int, int]) -> bool:
        r, g, b = rgb
        return (r + g + b) / 3 < bright_threshold

    filtered = [(count, rgb) for count, rgb in colors if is_not_too_bright(rgb)]
    if not filtered:
        # 全是白色之类的，那就退回原来的逻辑
        filtered = colors

    filtered.sort(key=lambda c: c[0], reverse=True)
    return filtered[0][1]


def button_base_color(
    img: Image.Image,
    *,
    border_ratio: float = 0.12,
    sample_size: int = 32,
    bright_threshold: int = 240,
    quantize_step: int = 16,
) -> Tuple[int, int, int]:
    """
    更鲁棒的“按钮底色”估计：
    - 先裁掉边缘（减少阴影/描边/背景干扰）
    - 再过滤掉过亮像素（减少白色文字/图标干扰）
    - 对剩余像素做颜色量化聚类（按桶计数），取最大簇作为底色

    说明：
    - quantize_step 越大，颜色桶越粗，越抗噪；默认 16（每通道 0..255 映射为 16 个桶）
    """
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    if w <= 0 or h <= 0:
        return (0, 0, 0)

    # 裁掉边缘，保留中心区域
    pad_x = int(round(w * border_ratio))
    pad_y = int(round(h * border_ratio))
    l, t, r, b = pad_x, pad_y, max(pad_x + 1, w - pad_x), max(pad_y + 1, h - pad_y)
    center = img_rgb.crop((l, t, r, b))

    # 降采样以加速，并减少细碎噪声影响
    center = center.resize((sample_size, sample_size))
    # Pillow 14 deprecates getdata; prefer get_flattened_data when available.
    if hasattr(center, "get_flattened_data"):
        pixels: Iterable[Tuple[int, int, int]] = list(center.get_flattened_data())
    else:
        pixels = list(center.getdata())

    def is_not_too_bright(rgb: Tuple[int, int, int]) -> bool:
        rr, gg, bb = rgb
        return (rr + gg + bb) / 3 < bright_threshold

    filtered = [p for p in pixels if is_not_too_bright(p)]
    if not filtered:
        filtered = list(pixels)

    # 量化到颜色桶（近似聚类）
    def q(v: int) -> int:
        step = max(1, int(quantize_step))
        return int(v // step) * step

    buckets: dict[Tuple[int, int, int], list[Tuple[int, int, int]]] = {}
    for rr, gg, bb in filtered:
        key = (q(rr), q(gg), q(bb))
        buckets.setdefault(key, []).append((rr, gg, bb))

    # 取最大簇，并返回其均值作为底色
    best = max(buckets.values(), key=len)
    rs = sum(p[0] for p in best)
    gs = sum(p[1] for p in best)
    bs = sum(p[2] for p in best)
    n = max(1, len(best))
    return (int(round(rs / n)), int(round(gs / n)), int(round(bs / n)))



def mean_color(img: Image.Image) -> Tuple[int, int, int]:
    stat = ImageStat.Stat(img.convert("RGB"))
    return tuple(int(round(c)) for c in stat.mean[:3])  # type: ignore[return-value]

def _encode(obj: Any):
    if isinstance(obj, CheckResult):
        return {
            "passed": obj.passed,
            "status": "pass" if obj.passed else "fail",
            "basis": obj.basis,
            "control_info": _encode(obj.control_info),
            "details": obj.details,
        }
    if isinstance(obj, ControlInfo):
        return {
            "bounds": [obj.bounds.left, obj.bounds.top, obj.bounds.right, obj.bounds.bottom],
            "text": obj.text,
            "semantics": obj.semantics,
            "main_color": obj.main_color,
            "checked": obj.checked,
            "source": obj.source,
            "extras": obj.extras,
        }
    return str(obj)


def get_image_color(path: str, bounds: Tuple[int, int, int, int] | None = None) -> dict:
    """
    Compute the mean color of an image (or a cropped region) and return both RGB and hex.

    - path: image file path.
    - bounds: optional crop box (left, top, right, bottom). If omitted, use full image.
    """
    img = load_image(path)
    if bounds:
        img = img.crop(bounds)
    rgb = mean_color(img)
    hex_color = "#%02x%02x%02x" % rgb
    return {"rgb": rgb, "hex": hex_color}

# def get_dominant_color(img_path, k=3):
#     img = Image.open(img_path)
#     img = img.resize((50, 50))  # 降采样加速
#     data = np.array(img).reshape(-1, 3)
    
#     kmeans = KMeans(n_clusters=k, n_init="auto").fit(data)
#     dominant = kmeans.cluster_centers_[0]
#     return tuple(map(int, dominant))
