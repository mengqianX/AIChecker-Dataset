"""GUI 前处理模块：为 VLM 提供结构化视觉证据。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


@dataclass
class PreprocessEvidence:
    """一次前处理输出。"""

    summary: str
    structured: dict[str, Any]
    extra_image_paths: list[Path]


class GuiPreprocessor:
    """轻量 GUI 前处理器（ROI + 差分 + 候选区域）。"""

    def __init__(
        self,
        artifact_dir: Path | None = None,
        logger: logging.Logger | None = None,
        max_extra_images: int = 2,
        toast_min_area_ratio: float = 0.002,
        toast_max_area_ratio: float = 0.15,
        toast_max_height_ratio: float = 0.22,
        toast_min_height_ratio: float = 0.02,
        toast_min_width_ratio: float = 0.18,
        toast_min_aspect_ratio: float = 1.5,
        toast_min_fill_ratio: float = 0.30,
        toast_max_surround_change: float = 0.12,
        toast_hot_min_hits: int = 3,
        toast_hot_min_episodes: int = 2,
        toast_hot_pair_fraction: float = 0.15,
        toast_hot_coverage_abort: float = 0.40,
        toast_hot_skip_global_change: float = 0.25,
        toast_diff_threshold: int = 16,
        toast_transience_horizon: int = 8,
        toast_persist_min_frames: int = 4,
        toast_transience_persist: float = 0.75,
        toast_transience_unknown: float = 0.70,
        toast_cv_max_width: int = 480,
        vlm_preview_max_long_edge: int = 512,
        vlm_preview_jpeg_quality: int = 80,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.max_extra_images = max(0, int(max_extra_images))
        self.toast_min_area_ratio = max(0.0001, float(toast_min_area_ratio))
        self.toast_max_area_ratio = min(1.0, max(self.toast_min_area_ratio, float(toast_max_area_ratio)))
        self.toast_max_height_ratio = min(1.0, max(0.05, float(toast_max_height_ratio)))
        self.toast_min_height_ratio = min(self.toast_max_height_ratio, max(0.005, float(toast_min_height_ratio)))
        self.toast_min_width_ratio = min(1.0, max(0.05, float(toast_min_width_ratio)))
        self.toast_min_aspect_ratio = min(20.0, max(1.0, float(toast_min_aspect_ratio)))
        self.toast_min_fill_ratio = min(1.0, max(0.05, float(toast_min_fill_ratio)))
        self.toast_max_surround_change = min(1.0, max(0.01, float(toast_max_surround_change)))
        self.toast_hot_min_hits = max(2, int(toast_hot_min_hits))
        self.toast_hot_min_episodes = max(1, int(toast_hot_min_episodes))
        self.toast_hot_pair_fraction = min(1.0, max(0.05, float(toast_hot_pair_fraction)))
        self.toast_hot_coverage_abort = min(1.0, max(0.1, float(toast_hot_coverage_abort)))
        self.toast_hot_skip_global_change = min(1.0, max(0.05, float(toast_hot_skip_global_change)))
        self.toast_diff_threshold = max(1, int(toast_diff_threshold))
        self.toast_transience_horizon = max(2, int(toast_transience_horizon))
        self.toast_persist_min_frames = max(2, int(toast_persist_min_frames))
        self.toast_transience_persist = min(1.0, max(0.05, float(toast_transience_persist)))
        self.toast_transience_unknown = min(1.0, max(0.05, float(toast_transience_unknown)))
        self.toast_cv_max_width = max(160, int(toast_cv_max_width))
        self.vlm_preview_max_long_edge = max(256, int(vlm_preview_max_long_edge))
        self.vlm_preview_jpeg_quality = min(95, max(40, int(vlm_preview_jpeg_quality)))

    @staticmethod
    def _read_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return image

    @staticmethod
    def _clamp_bounds(width: int, height: int, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(int(x), width - 1))
        y1 = max(0, min(int(y), height - 1))
        x2 = max(x1 + 1, min(int(x) + int(w), width))
        y2 = max(y1 + 1, min(int(y) + int(h), height))
        return x1, y1, x2, y2

    def _save_artifact(self, image: Any, file_name: str) -> Path | None:
        if self.artifact_dir is None:
            return None
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / file_name
        ok = cv2.imwrite(str(path), image)
        if not ok:
            self.logger.warning("前处理图保存失败: %s", path)
            return None
        return path

    def write_vlm_preview(
        self,
        source: Path,
        dest: Path,
        *,
        max_long_edge: int | None = None,
        jpeg_quality: int | None = None,
    ) -> Path:
        """把完整帧压成较小 JPEG，仅降低 VLM 传输/推理成本，不改原图。"""
        image = self._read_image(source)
        height, width = image.shape[:2]
        long_edge_limit = max(256, int(max_long_edge or self.vlm_preview_max_long_edge))
        long_edge = max(height, width)
        if long_edge > long_edge_limit:
            scale = long_edge_limit / float(long_edge)
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        quality = min(95, max(40, int(jpeg_quality or self.vlm_preview_jpeg_quality)))
        ok = cv2.imwrite(str(dest), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError(f"无法写入 VLM 预览图: {dest}")
        return dest

    def _resize_for_toast_cv(self, image: Any) -> tuple[Any, float]:
        height, width = image.shape[:2]
        if width <= self.toast_cv_max_width:
            return image, 1.0
        scale = self.toast_cv_max_width / float(width)
        small = cv2.resize(
            image,
            (self.toast_cv_max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return small, scale

    @staticmethod
    def _scale_box(box: dict[str, int] | None, inv_scale: float) -> dict[str, int] | None:
        if box is None or abs(inv_scale - 1.0) < 1e-6:
            return box
        return {
            "x1": int(round(int(box["x1"]) * inv_scale)),
            "y1": int(round(int(box["y1"]) * inv_scale)),
            "x2": int(round(int(box["x2"]) * inv_scale)),
            "y2": int(round(int(box["y2"]) * inv_scale)),
        }

    @staticmethod
    def _calc_diff_metrics(before_roi: Any, after_roi: Any) -> tuple[float, float, Any]:
        diff = cv2.absdiff(before_roi, after_roi)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(binary.shape[0] * binary.shape[1]))
        change_ratio = changed_pixels / total_pixels
        mean_abs_diff = float(diff_gray.mean())
        return mean_abs_diff, change_ratio, diff_gray

    @staticmethod
    def _calc_frame_change_ratio(image_a: Any, image_b: Any, threshold: int = 22) -> float:
        diff = cv2.absdiff(image_a, image_b)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(binary.shape[0] * binary.shape[1]))
        return changed_pixels / total_pixels

    def prepare_count_change(
        self,
        before_image: Path,
        after_image: Path,
        bounds: dict[str, int],
        task_id: str,
    ) -> PreprocessEvidence:
        """构造 count_change 场景证据。"""
        t_start = time.perf_counter()
        before = self._read_image(before_image)
        after = self._read_image(after_image)
        h, w = before.shape[:2]
        x1, y1, x2, y2 = self._clamp_bounds(
            width=w,
            height=h,
            x=int(bounds["x"]),
            y=int(bounds["y"]),
            w=int(bounds["width"]),
            h=int(bounds["height"]),
        )

        before_roi = before[y1:y2, x1:x2]
        after_roi = after[y1:y2, x1:x2]
        mean_abs_diff, change_ratio, diff_gray = self._calc_diff_metrics(before_roi, after_roi)

        out_paths: list[Path] = []
        before_path = self._save_artifact(before_roi, f"{task_id}_pp_count_before_roi.png")
        after_path = self._save_artifact(after_roi, f"{task_id}_pp_count_after_roi.png")
        diff_path = self._save_artifact(diff_gray, f"{task_id}_pp_count_diff_gray.png")
        for path in [before_path, after_path, diff_path]:
            if path is not None:
                out_paths.append(path)

        structured = {
            "evidence_type": "count_change_roi_diff",
            "roi_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "roi_mean_abs_diff": round(mean_abs_diff, 3),
            "roi_changed_pixel_ratio": round(change_ratio, 4),
            "notes": "该指标描述控件区域前后变化强度，仅用于辅助 VLM 注意力聚焦。",
        }
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        structured["preprocess_elapsed_ms"] = round(elapsed_ms, 2)
        summary = (
            "前处理证据(count_change): "
            f"控件ROI=({x1},{y1})-({x2},{y2}), "
            f"mean_abs_diff={mean_abs_diff:.3f}, changed_ratio={change_ratio:.4f}, elapsed={elapsed_ms:.2f}ms"
        )
        return PreprocessEvidence(
            summary=summary,
            structured=structured,
            extra_image_paths=out_paths[: self.max_extra_images],
        )

    def _pair_diff_gray(self, image_a: Any, image_b: Any) -> Any:
        diff = cv2.absdiff(image_a, image_b)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _pair_diff_binary(self, image_a: Any, image_b: Any, *, close: bool = True) -> Any:
        blurred = self._pair_diff_gray(image_a, image_b)
        _, binary = cv2.threshold(blurred, self.toast_diff_threshold, 255, cv2.THRESH_BINARY)
        if not close:
            return binary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _build_chronic_hot_mask(
        self,
        binaries: list[Any],
        frame_shape: tuple[int, int],
    ) -> tuple[Any | None, dict[str, Any]]:
        """标出整段视频里反复变化的粗网格，供选帧时挖掉广告/轮播。"""
        n_pairs = len(binaries)
        if n_pairs < 3:
            return None, {"enabled": False, "reason": "too_few_frames", "hot_cell_count": 0, "coverage": 0.0}

        height, width = frame_shape
        rows, cols = 12, 8
        cell_h = max(1, height // rows)
        cell_w = max(1, width // cols)
        grid_h = rows * cell_h
        grid_w = cols * cell_w
        hits = np.zeros((rows, cols), dtype=np.int32)
        lit_series: list[Any] = []
        for binary in binaries:
            total = int(binary.size)
            if total > 0 and (cv2.countNonZero(binary) / float(total)) >= self.toast_hot_skip_global_change:
                # 整页跳转不计入慢性热区，避免把一次转场叠加 toast 出现/消失标成广告槽。
                lit_series.append(np.zeros((rows, cols), dtype=np.uint8))
                continue
            cropped = binary[:grid_h, :grid_w]
            cells = cropped.reshape(rows, cell_h, cols, cell_w)
            frac = (cells > 0).mean(axis=(1, 3))
            lit = (frac >= 0.12).astype(np.uint8)
            hits += lit.astype(np.int32)
            lit_series.append(lit)

        min_hits = max(self.toast_hot_min_hits, int(n_pairs * self.toast_hot_pair_fraction))
        # toast 自己出现/消失会连着点亮 2～4 对，不能当成广告槽挖掉。
        # 只把「多段发作」或跨很多段的格子标成热区。
        stacked = np.stack(lit_series, axis=0) if lit_series else np.zeros((0, rows, cols), dtype=np.uint8)
        episodes = np.zeros((rows, cols), dtype=np.int32)
        if stacked.shape[0] > 0:
            prev = np.zeros((rows, cols), dtype=np.uint8)
            for frame_lit in stacked:
                episodes += ((frame_lit > 0) & (prev == 0)).astype(np.int32)
                prev = frame_lit
        hot_cells = (hits >= min_hits) & (
            (episodes >= self.toast_hot_min_episodes)
            | (hits >= max(min_hits + 2, int(0.40 * n_pairs)))
        )
        coverage = float(hot_cells.mean())
        if coverage >= self.toast_hot_coverage_abort:
            return None, {
                "enabled": False,
                "reason": "global_motion",
                "hot_cell_count": int(hot_cells.sum()),
                "coverage": round(coverage, 4),
                "min_hits": min_hits,
            }

        mask = np.zeros((height, width), dtype=np.uint8)
        for row in range(rows):
            y1 = row * cell_h
            y2 = height if row == rows - 1 else (row + 1) * cell_h
            for col in range(cols):
                if not hot_cells[row, col]:
                    continue
                x1 = col * cell_w
                x2 = width if col == cols - 1 else (col + 1) * cell_w
                mask[y1:y2, x1:x2] = 255
        return mask, {
            "enabled": True,
            "reason": "chronic_slots",
            "hot_cell_count": int(hot_cells.sum()),
            "coverage": round(coverage, 4),
            "min_hits": min_hits,
        }

    @staticmethod
    def _empty_overlay_score() -> dict[str, Any]:
        return {
            "score": 0.0,
            "box": None,
            "fill_ratio": 0.0,
            "surround_change": 1.0,
            "area_ratio": 0.0,
            "height_ratio": 0.0,
            "width_ratio": 0.0,
            "aspect_ratio": 0.0,
            "score_kind": "none",
        }

    @staticmethod
    def _toast_shape_score(width_ratio: float, height_ratio: float, aspect_ratio: float) -> float:
        """宽短条 / 居中胶囊偏高，细线、方块、半屏面板偏低。不限制出现在顶/底。"""
        width_term = min(1.0, max(0.0, width_ratio / 0.55))
        height_term = max(0.0, 1.0 - abs(height_ratio - 0.07) / 0.14)
        aspect_term = min(1.0, max(0.0, (aspect_ratio - 1.2) / 4.8))
        return 0.35 * width_term + 0.35 * height_term + 0.30 * aspect_term

    def _score_rectangular_overlay(
        self,
        binary: Any,
        hot_mask: Any | None,
        *,
        score_kind: str = "overlay",
    ) -> dict[str, Any]:
        """在热区外找一块框外稳定的矩形覆盖。"""
        if hot_mask is not None:
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(hot_mask))
        height, width = binary.shape[:2]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(max(1, height * width))
        ring = max(8, int(round(0.025 * width)))
        best: dict[str, Any] = self._empty_overlay_score()
        best["score_kind"] = score_kind
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            area = float(box_w * box_h)
            area_ratio = area / frame_area
            height_ratio = float(box_h) / float(max(1, height))
            width_ratio = float(box_w) / float(max(1, width))
            aspect_ratio = float(box_w) / float(max(1, box_h))
            if area_ratio < self.toast_min_area_ratio or area_ratio > self.toast_max_area_ratio:
                continue
            if height_ratio > self.toast_max_height_ratio or height_ratio < self.toast_min_height_ratio:
                continue
            if width_ratio < self.toast_min_width_ratio or aspect_ratio < self.toast_min_aspect_ratio:
                continue
            roi = binary[y : y + box_h, x : x + box_w]
            fill_ratio = float(cv2.countNonZero(roi)) / max(1.0, area)
            if fill_ratio < self.toast_min_fill_ratio:
                continue

            x0 = max(0, x - ring)
            y0 = max(0, y - ring)
            x1 = min(width, x + box_w + ring)
            y1 = min(height, y + box_h + ring)
            ring_region = binary[y0:y1, x0:x1].copy()
            inner_y = y - y0
            inner_x = x - x0
            ring_region[inner_y : inner_y + box_h, inner_x : inner_x + box_w] = 0
            ring_pixels = int(ring_region.size) - int(area)
            surround_change = float(cv2.countNonZero(ring_region)) / float(max(1, ring_pixels))
            if surround_change > self.toast_max_surround_change:
                continue

            locality = max(0.0, 1.0 - surround_change / self.toast_max_surround_change)
            shape = self._toast_shape_score(width_ratio, height_ratio, aspect_ratio)
            score = 0.45 * fill_ratio + 0.30 * locality + 0.25 * shape
            if score <= float(best["score"]):
                continue
            best = {
                "score": float(score),
                "box": {"x1": int(x), "y1": int(y), "x2": int(x + box_w), "y2": int(y + box_h)},
                "fill_ratio": float(fill_ratio),
                "surround_change": float(surround_change),
                "area_ratio": float(area_ratio),
                "height_ratio": float(height_ratio),
                "width_ratio": float(width_ratio),
                "aspect_ratio": float(aspect_ratio),
                "score_kind": score_kind,
            }
        return best

    def _score_toast_appearance(self, gray_a: Any, gray_b: Any) -> dict[str, Any]:
        """
        在目标帧里找一块新出现的宽短条/胶囊。
        整页跳转时 pair-diff 轮廓会连成一整块，居中半透明 toast 也靠这个补。
        """
        height, width = gray_b.shape[:2]
        best = self._empty_overlay_score()
        best["score_kind"] = "appearance"
        absdiff = np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32))
        global_mae = float(absdiff.mean())
        row_mae = absdiff.mean(axis=1)
        peak_threshold = max(8.0, 1.5 * global_mae)
        peak_rows: list[int] = []
        for y in range(int(0.04 * height), int(0.96 * height)):
            window = row_mae[max(0, y - 4) : y + 5]
            if float(row_mae[y]) >= peak_threshold and float(row_mae[y]) >= float(window.max()):
                peak_rows.append(y)
        centers = peak_rows[:16]
        step = max(8, int(0.07 * height))
        centers.extend(list(range(int(0.07 * height), int(0.93 * height), step)))

        for center_y in centers:
            for height_frac in (0.045, 0.07, 0.11):
                box_h = max(10, int(round(height * height_frac)))
                y = max(0, min(height - box_h, int(center_y - box_h // 2)))
                for width_frac in (0.36, 0.55, 0.78, 0.90):
                    box_w = max(24, int(round(width * width_frac)))
                    x_options = [(width - box_w) // 2, int(0.04 * width)]
                    for x in x_options:
                        if x < 0 or x + box_w > width:
                            continue
                        win_b = gray_b[y : y + box_h, x : x + box_w]
                        win_a = gray_a[y : y + box_h, x : x + box_w]
                        mae = float(np.mean(np.abs(win_b.astype(np.float32) - win_a.astype(np.float32))))
                        if mae < 10.0:
                            continue
                        relative = mae / (global_mae + 6.0)
                        std_b = float(win_b.std())
                        if std_b > 72.0:
                            continue
                        mean_b = float(win_b.mean())
                        y0 = max(0, y - 8)
                        x0 = max(0, x - 8)
                        y1 = min(height, y + box_h + 8)
                        x1 = min(width, x + box_w + 8)
                        contrast = abs(mean_b - float(gray_b[y0:y1, x0:x1].mean()))
                        if contrast < 6.0:
                            continue
                        width_ratio = float(box_w) / float(max(1, width))
                        height_ratio = float(box_h) / float(max(1, height))
                        aspect_ratio = float(box_w) / float(max(1, box_h))
                        shape = self._toast_shape_score(width_ratio, height_ratio, aspect_ratio)
                        score = (
                            0.35 * min(1.0, mae / 45.0)
                            + 0.25 * min(1.0, relative / 1.6)
                            + 0.25 * min(1.0, contrast / 24.0)
                            + 0.15 * shape
                        )
                        if global_mae > 25.0:
                            score *= min(1.0, contrast / 12.0)
                        if score <= float(best["score"]):
                            continue
                        best = {
                            "score": float(score),
                            "box": {"x1": int(x), "y1": int(y), "x2": int(x + box_w), "y2": int(y + box_h)},
                            "fill_ratio": min(1.0, mae / 40.0),
                            "surround_change": max(0.0, 1.0 - contrast / 40.0),
                            "area_ratio": float(box_w * box_h) / float(max(1, height * width)),
                            "height_ratio": height_ratio,
                            "width_ratio": width_ratio,
                            "aspect_ratio": aspect_ratio,
                            "score_kind": "appearance",
                        }
        return best

    def _uniform_panel_area_ratio(self, gray: Any, box: dict[str, int] | None) -> float:
        """box 若落在大块同色面板（对话框/底栏菜单/键盘）里，面积占比会明显大于 toast。"""
        if box is None or gray is None:
            return 0.0
        height, width = gray.shape[:2]
        x1 = max(0, min(int(box["x1"]), width - 1))
        x2 = max(x1 + 1, min(int(box["x2"]), width))
        y1 = max(0, min(int(box["y1"]), height - 1))
        y2 = max(y1 + 1, min(int(box["y2"]), height))
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        mu = float(patch.mean())
        similar = (np.abs(gray.astype(np.float32) - mu) <= 22.0).astype(np.uint8)
        _num_labels, labels = cv2.connectedComponents(similar, connectivity=8)
        cx = min(width - 1, max(0, (x1 + x2) // 2))
        cy = min(height - 1, max(0, (y1 + y2) // 2))
        label = int(labels[cy, cx])
        if label <= 0:
            return 0.0
        return float(np.count_nonzero(labels == label)) / float(max(1, height * width))

    def _penalize_large_panel(self, item: dict[str, Any], gray: Any) -> None:
        ratio = self._uniform_panel_area_ratio(gray, item.get("box"))
        item["panel_area_ratio"] = round(ratio, 4)
        if ratio >= 0.10:
            item["score"] = float(item.get("score") or 0.0) * 0.42

    def _box_mask_fraction(self, box: dict[str, int] | None, mask: Any | None) -> float:
        if box is None or mask is None:
            return 0.0
        height, width = mask.shape[:2]
        x1 = max(0, min(int(box["x1"]), width - 1))
        x2 = max(x1 + 1, min(int(box["x2"]), width))
        y1 = max(0, min(int(box["y1"]), height - 1))
        y2 = max(y1 + 1, min(int(box["y2"]), height))
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(cv2.countNonZero(roi)) / float(roi.size)

    @staticmethod
    def _better_overlay(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        if float(right.get("score") or 0.0) > float(left.get("score") or 0.0):
            return right
        return left

    @staticmethod
    def _choose_stabler_pair(
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str]:
        if left is None and right is None:
            return None, "none"
        if left is None:
            return right, "pair_center_after"
        if right is None:
            return left, "pair_before_center"
        if float(left.get("score", 0.0)) >= float(right.get("score", 0.0)):
            return left, "pair_before_center"
        return right, "pair_center_after"

    @staticmethod
    def _box_patch(gray: Any, box: dict[str, int]) -> Any:
        height, width = gray.shape[:2]
        x1 = max(0, min(int(box["x1"]), width - 1))
        x2 = max(x1 + 1, min(int(box["x2"]), width))
        y1 = max(0, min(int(box["y1"]), height - 1))
        y2 = max(y1 + 1, min(int(box["y2"]), height))
        return gray[y1:y2, x1:x2]

    @staticmethod
    def _patch_mae(patch_a: Any, patch_b: Any) -> float:
        if patch_a.size == 0 or patch_b.size == 0:
            return 255.0
        if patch_a.shape != patch_b.shape:
            patch_b = cv2.resize(
                patch_b,
                (int(patch_a.shape[1]), int(patch_a.shape[0])),
                interpolation=cv2.INTER_AREA,
            )
        return float(np.mean(np.abs(patch_a.astype(np.float32) - patch_b.astype(np.float32))))

    def _match_box_side(self, patch: Any, patch_a: Any, patch_b: Any) -> str:
        dist_a = self._patch_mae(patch, patch_a)
        dist_b = self._patch_mae(patch, patch_b)
        if min(dist_a, dist_b) > 48.0:
            return "other"
        if dist_a + 8.0 < dist_b:
            return "a"
        if dist_b + 8.0 < dist_a:
            return "b"
        return "tie"

    def _overlay_transience(
        self,
        grays: list[Any],
        box: dict[str, int],
        idx_a: int,
        idx_b: int,
    ) -> tuple[float, str]:
        """
        后续帧里这块覆盖是消失了，还是一直留着。
        1.0=确认消失（toast），persist=一直在（引导气泡），unknown=未来帧不够。
        """
        patch_a = self._box_patch(grays[idx_a], box)
        patch_b = self._box_patch(grays[idx_b], box)
        if patch_a.size == 0 or patch_b.size == 0:
            return self.toast_transience_unknown, "empty_box"

        bg_side, fg_side = "a", "b"
        if idx_a > 0:
            before_side = self._match_box_side(self._box_patch(grays[idx_a - 1], box), patch_a, patch_b)
            if before_side == "b":
                bg_side, fg_side = "b", "a"

        start = max(idx_a, idx_b) + 1
        if start >= len(grays):
            return self.toast_transience_unknown, "no_future"

        horizon = min(len(grays), start + self.toast_transience_horizon)
        after_sides = [
            self._match_box_side(self._box_patch(grays[idx], box), patch_a, patch_b)
            for idx in range(start, horizon)
        ]
        fg_count = sum(side == fg_side for side in after_sides)
        bg_count = sum(side == bg_side for side in after_sides)
        other_count = sum(side == "other" for side in after_sides)
        switched_to_bg = False
        seen_fg = False
        for side in after_sides:
            if side == fg_side:
                seen_fg = True
            elif side == bg_side and seen_fg:
                switched_to_bg = True
                break

        if switched_to_bg or (bg_count >= 1 and fg_count == 0):
            return 1.0, "disappeared"
        if other_count >= 1 and fg_count == 0:
            return 1.0, "left_scene"
        if fg_count >= self.toast_persist_min_frames:
            return self.toast_transience_persist, "still_present"
        if after_sides and fg_count == len(after_sides):
            return self.toast_transience_unknown, "still_present_short"
        return self.toast_transience_unknown, "unclear"

    def _apply_overlay_transience(self, pair_scores: list[dict[str, Any]], grays: list[Any]) -> None:
        for pair_idx, item in enumerate(pair_scores):
            overlay_score = float(item.get("score") or 0.0)
            box = item.get("box")
            item["overlay_score"] = overlay_score
            if box is None or overlay_score <= 0.0:
                item["transience"] = None
                item["transience_weight"] = None
                continue
            weight, tag = self._overlay_transience(grays, box, pair_idx, pair_idx + 1)
            change_ratio = float(item.get("pair_change_ratio") or 0.0)
            if tag in {"disappeared", "left_scene"} and change_ratio >= 0.25:
                weight = min(weight, 0.55)
            elif tag == "still_present":
                aspect = float(item.get("aspect_ratio") or 0.0)
                height_ratio = float(item.get("height_ratio") or 0.0)
                if aspect >= 2.4 and height_ratio <= 0.16:
                    weight = max(weight, 0.92)
            item["transience"] = tag
            item["transience_weight"] = round(weight, 4)
            item["score"] = overlay_score * weight

    def score_toast_sequence(self, image_paths: list[Path]) -> list[dict[str, Any]]:
        """
        全帧扫描：慢性热区（多段发作才挖广告）、矩形覆盖、以及新出现的宽短条/胶囊。
        大块同色面板（对话框/键盘/底栏菜单）降权；snackbar 短时停留不再按常驻气泡打压。
        """
        t_start = time.perf_counter()
        if len(image_paths) < 2:
            raise ValueError("toast 打分至少需要 2 帧")

        images = [self._read_image(path) for path in image_paths]
        ref_h, ref_w = images[0].shape[:2]
        aligned: list[Any] = []
        for image in images:
            if image.shape[0] == ref_h and image.shape[1] == ref_w:
                aligned.append(image)
            else:
                aligned.append(cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_AREA))

        scaled: list[Any] = []
        scale = 1.0
        for image in aligned:
            small, scale = self._resize_for_toast_cv(image)
            scaled.append(small)
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        pair_binaries = [
            self._pair_diff_binary(scaled[idx], scaled[idx + 1])
            for idx in range(len(scaled) - 1)
        ]
        raw_binaries = [
            self._pair_diff_binary(scaled[idx], scaled[idx + 1], close=False)
            for idx in range(len(scaled) - 1)
        ]
        hot_mask, hot_meta = self._build_chronic_hot_mask(pair_binaries, scaled[0].shape[:2])
        grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in scaled]
        pair_scores: list[dict[str, Any]] = []
        for idx, closed in enumerate(pair_binaries):
            overlay = self._better_overlay(
                self._score_rectangular_overlay(closed, hot_mask, score_kind="overlay"),
                self._score_rectangular_overlay(raw_binaries[idx], hot_mask, score_kind="overlay_raw"),
            )
            appearance = self._score_toast_appearance(grays[idx], grays[idx + 1])
            if hot_mask is not None and self._box_mask_fraction(appearance.get("box"), hot_mask) >= 0.5:
                appearance["score"] = float(appearance.get("score") or 0.0) * 0.4
            chosen = self._better_overlay(overlay, appearance)
            self._penalize_large_panel(chosen, grays[idx + 1])
            change = 0.0
            if closed.size:
                change = float(cv2.countNonZero(closed)) / float(closed.size)
            chosen["pair_change_ratio"] = round(change, 4)
            pair_scores.append(chosen)
        self._apply_overlay_transience(pair_scores, grays)

        results: list[dict[str, Any]] = []
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        for idx in range(len(scaled)):
            left = pair_scores[idx - 1] if idx > 0 else None
            right = pair_scores[idx] if idx < len(scaled) - 1 else None
            chosen, source = self._choose_stabler_pair(left, right)
            score = float(chosen["score"]) if chosen else 0.0
            box = None if chosen is None else self._scale_box(chosen.get("box"), inv_scale)
            results.append(
                {
                    "index": idx,
                    "score": round(score, 4),
                    "source": source,
                    "candidate_roi_box": box,
                    "fill_ratio": None if chosen is None else round(float(chosen.get("fill_ratio", 0.0)), 4),
                    "surround_change": None if chosen is None else round(float(chosen.get("surround_change", 1.0)), 4),
                    "area_ratio": None if chosen is None else round(float(chosen.get("area_ratio", 0.0)), 4),
                    "height_ratio": None if chosen is None else round(float(chosen.get("height_ratio", 0.0)), 4),
                    "aspect_ratio": None if chosen is None else round(float(chosen.get("aspect_ratio", 0.0)), 4),
                    "width_ratio": None if chosen is None else round(float(chosen.get("width_ratio", 0.0)), 4),
                    "score_kind": None if chosen is None else chosen.get("score_kind"),
                    "overlay_score": None if chosen is None else round(float(chosen.get("overlay_score", chosen.get("score", 0.0))), 4),
                    "transience": None if chosen is None else chosen.get("transience"),
                    "transience_weight": None if chosen is None else chosen.get("transience_weight"),
                    "hot_mask": hot_meta,
                    "score_elapsed_ms": round(elapsed_ms, 2),
                    "reason": "矩形覆盖或新出现的宽短条，并按后续是否消失加权",
                }
            )
        return results
