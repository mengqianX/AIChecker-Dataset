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


def _binary_dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1)x(2r+1) square structuring element.

    Implemented as two separable passes (horizontal then vertical) of slice
    OR-shifts, so the cost is O(r) instead of O(r^2) per pass.

    The purpose in our pipeline is to bridge nearby disconnected components
    of the diff mask (e.g. individual characters of a text button "Qwen3 - Max"
    sitting a few pixels apart) so that the largest-connected-component
    statistic recognises the text *line* as one structure rather than dozens
    of unrelated specks.
    """
    if radius <= 0:
        return mask
    h, w = mask.shape
    horizontal = mask.copy()
    for r in range(1, radius + 1):
        horizontal[:, r:] |= mask[:, : w - r]
        horizontal[:, : w - r] |= mask[:, r:]
    out = horizontal.copy()
    for r in range(1, radius + 1):
        out[r:, :] |= horizontal[: h - r, :]
        out[: h - r, :] |= horizontal[r:, :]
    return out


def _largest_connected_component_area(mask: np.ndarray) -> int:
    """4-connectivity flood-fill, returns size of largest True component."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = 0
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            stack: list[tuple[int, int]] = [(i, j)]
            size = 0
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if visited[y, x] or not mask[y, x]:
                    continue
                visited[y, x] = True
                size += 1
                stack.append((y + 1, x))
                stack.append((y - 1, x))
                stack.append((y, x + 1))
                stack.append((y, x - 1))
            if size > best:
                best = size
    return best


def _rgb_to_value_saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return HSV V and S channels as float arrays in [0, 1]."""
    rgb_f = rgb.astype(np.float32) / 255.0
    cmax = rgb_f.max(axis=-1)
    cmin = rgb_f.min(axis=-1)
    delta = cmax - cmin
    s = np.where(cmax == 0, 0, delta / np.where(cmax == 0, 1, cmax))
    return cmax, s


def _best_iou_over_shifts(
    ma: np.ndarray, mb: np.ndarray, max_shift: int
) -> tuple[float, int, int]:
    """Brute-force the (dx, dy) translation in [-R, R]^2 that maximises
    IoU(ma, shift(mb, dx, dy)).

    Cost: O((2R+1)^2 * pixels).  For R=4 on a 100x100 mask that's ~80 numpy
    bitops, well under a millisecond.

    Why this matters: the outline IoU veto fails as soon as the two ink
    masks are off by even 1-2 pixels (typical for screenshots taken across
    a layout reflow, scrolling video overlays, or device-level animation
    in flight).  Searching a small translation window restores the veto's
    core invariant ("did the icon's geometry actually change?") without
    being fooled into matching truly-different shapes — a real activation
    that reshapes the ink (outline -> filled, etc.) will still produce
    low IoU at every offset.
    """
    h, w = ma.shape
    a_sum = int(ma.sum())
    best_iou = -1.0
    best_dx = best_dy = 0
    for dy in range(-max_shift, max_shift + 1):
        y0_src = max(0, -dy); y1_src = min(h, h - dy)
        y0_dst = max(0, dy);  y1_dst = min(h, h + dy)
        for dx in range(-max_shift, max_shift + 1):
            x0_src = max(0, -dx); x1_src = min(w, w - dx)
            x0_dst = max(0, dx);  x1_dst = min(w, w + dx)
            shifted = np.zeros_like(mb)
            shifted[y0_dst:y1_dst, x0_dst:x1_dst] = mb[y0_src:y1_src, x0_src:x1_src]
            inter = int((ma & shifted).sum())
            b_sum = int(shifted.sum())
            union = a_sum + b_sum - inter
            iou = inter / max(1, union)
            if iou > best_iou:
                best_iou = iou
                best_dx, best_dy = dx, dy
    return best_iou, best_dx, best_dy


def outline_iou_and_coverage(
    crop_a: Image.Image,
    crop_b: Image.Image,
    *,
    white_v_min: float = 0.90,
    white_s_max: float = 0.18,
    black_v_max: float = 0.10,
    max_shift: int = 4,
    min_align_count: int = 8,
) -> dict:
    """
    Detect "outline ink" pixels (deliberately-rendered near-white or near-black
    pixels — e.g. the white outline of an overlay like-button) in both frames
    and compare their geometry.

    Pure-white and pure-black pixels almost never occur in natural photo /
    video content (which always picks up a colour cast and never reaches
    the channel extremes due to dynamic-range compression).  Buttons rendered
    for high-contrast overlay use, on the other hand, routinely use these
    extremes.

    The function picks whichever of {near-white, near-black} is most stably
    present *in both frames* — i.e. the ink colour with the largest
    ``min(count_a, count_b)`` — then computes:

    - ``coverage``  : max(count_a, count_b) / pixel_count
                      How much of the bbox is covered by ink in either frame.
    - ``iou``       : intersection-over-union of the two ink masks.
                      Tells us whether the outline shape stayed in the same
                      pixel positions.  ``iou ≈ 1`` means the icon is
                      geometrically identical between the two frames.

    A real button activation that turns an outline icon into a filled icon
    drastically reshapes the ink mask (low IoU); a static outline icon over
    a drifting video background leaves the mask nearly identical (IoU near
    1) regardless of how chaotic the surrounding pixels are.

    Why ``min(count_a, count_b)`` and not ``count_a + count_b``:  video
    backgrounds frequently contribute large amounts of *one-sided* ink
    (e.g. one frame is a near-black night shot, the next frame is bright
    daylight) — these pixels are not real UI ink, just background bleed,
    and they would dominate any sum-based selector.  Taking the per-frame
    minimum filters them out: only ink that survives in *both* frames is
    considered, which is exactly the geometric signature of a static UI
    overlay.  Real button outlines are rendered identically on each frame
    so their per-frame counts are nearly equal; transient video-content
    pixels collapse to ``min ≈ 0``.
    """
    a = np.array(crop_a.convert("RGB"))
    b_img = crop_b if crop_b.size == crop_a.size else crop_b.resize(crop_a.size)
    b = np.array(b_img.convert("RGB"))

    va, sa = _rgb_to_value_saturation(a)
    vb, sb = _rgb_to_value_saturation(b)
    near_white_a = (va >= white_v_min) & (sa <= white_s_max)
    near_white_b = (vb >= white_v_min) & (sb <= white_s_max)
    near_black_a = va <= black_v_max
    near_black_b = vb <= black_v_max

    white_stable = min(int(near_white_a.sum()), int(near_white_b.sum()))
    black_stable = min(int(near_black_a.sum()), int(near_black_b.sum()))
    if white_stable >= black_stable:
        ma, mb = near_white_a, near_white_b
        ink = "white"
    else:
        ma, mb = near_black_a, near_black_b
        ink = "black"

    count_a = int(ma.sum())
    count_b = int(mb.sum())
    total = max(1, ma.size)

    # Search a small translation window for the alignment that maximises IoU.
    # When both masks are too sparse to align reliably, fall back to a direct
    # IoU at zero shift — searching there only adds noise.
    if max_shift > 0 and count_a >= min_align_count and count_b >= min_align_count:
        iou, dx, dy = _best_iou_over_shifts(ma, mb, max_shift)
    else:
        inter = int((ma & mb).sum())
        union = int((ma | mb).sum())
        iou = inter / max(1, union)
        dx = dy = 0

    return {
        "ink_color": ink,
        "count_a": count_a,
        "count_b": count_b,
        "coverage": max(count_a, count_b) / total,
        "iou": iou,
        "shift": (dx, dy),
    }


def diff_structure_score(
    crop_a: Image.Image,
    crop_b: Image.Image,
    *,
    pixel_diff_threshold: int = 20,
    min_change_pixels: int = 8,
    mask_dilate_radius: int = 2,
) -> dict:
    """
    Decide whether the change between (crop_a, crop_b) looks like a real
    button state transition, as opposed to background drift on a transparent
    overlay (e.g. like-button on top of playing video).

    Returns a dict with three normalised metrics (each in [0, 1]) and a
    composite ``score``:

    - ``concentration`` = largest_connected_component / n_dilated_pixels
      Real activations form a single compact blob; background noise scatters.

      The mask is morphologically dilated by ``mask_dilate_radius`` pixels
      *before* connected-component counting so that text-style activations
      ("Qwen3 - Max" turning blue→black) — whose changed pixels are spread
      across many separated character glyphs — are recognised as a single
      coherent text-line structure.  Without this, concentration measures
      "biggest single character / total characters" and saturates near 1/N.
      A radius of 2 (i.e. a 5x5 SE) bridges typical inter-character gaps
      without merging unrelated regions of a noisy background.

    - ``coherence`` = |mean(d_i)| / mean(|d_i|), where d_i is the signed RGB
      delta of the i-th changed pixel. Real activations push pixels in one
      consistent direction; background drift cancels out.  Computed on the
      *original* (un-dilated) mask so dilated-in pixels don't dilute the
      direction estimate with their unrelated colour values.

    - ``centrality`` = inner_density / (inner_density + outer_density), where
      inner is the central 50%x50% region. Real activations concentrate at
      the centre; background frames around an unchanged button form a "donut"
      pattern with low centrality.  Also computed on the original mask.

    The composite score scales the product so that a balanced case (where all
    three signals are roughly equal) lands near the geometric mean. Threshold
    around 0.20-0.30 in practice.
    """
    a = np.array(crop_a.convert("RGB"), dtype=np.int16)
    b_img = crop_b
    if crop_b.size != crop_a.size:
        b_img = crop_b.resize(crop_a.size)
    b = np.array(b_img.convert("RGB"), dtype=np.int16)

    d = b - a
    abs_max = np.abs(d).max(axis=2)
    mask = abs_max > pixel_diff_threshold
    n_changed = int(mask.sum())
    coverage = n_changed / max(1, mask.size)

    if n_changed < min_change_pixels:
        return {
            "coverage": coverage,
            "concentration": 0.0,
            "coherence": 0.0,
            "centrality": 0.0,
            "score": 0.0,
            "n_changed": n_changed,
        }

    mask_for_cc = _binary_dilate_square(mask, mask_dilate_radius) if mask_dilate_radius > 0 else mask
    n_for_cc = int(mask_for_cc.sum()) if mask_dilate_radius > 0 else n_changed
    cmax = _largest_connected_component_area(mask_for_cc)
    concentration = cmax / max(1, n_for_cc)

    sel = d[mask]
    mean_dir = sel.mean(axis=0)
    mean_dir_norm = float(np.linalg.norm(mean_dir))
    per_pixel_norm = float(np.linalg.norm(sel, axis=1).mean())
    coherence = mean_dir_norm / max(1e-6, per_pixel_norm)

    h, w = mask.shape
    cy0, cy1 = h // 4, h - h // 4
    cx0, cx1 = w // 4, w - w // 4
    inner = mask[cy0:cy1, cx0:cx1]
    inner_area = max(1, inner.size)
    outer_area = max(1, mask.size - inner_area)
    inner_density = float(inner.sum()) / inner_area
    outer_density = float(int(mask.sum()) - int(inner.sum())) / outer_area
    centrality = inner_density / max(1e-6, inner_density + outer_density)

    score = float(concentration * coherence * centrality * 2.0)
    return {
        "coverage": float(coverage),
        "concentration": float(concentration),
        "coherence": float(coherence),
        "centrality": float(centrality),
        "score": score,
        "n_changed": n_changed,
    }

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
