"""Motion-noise analysis for videos with dynamic ads, carousels, or live content."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionNoiseRegion:
    """One persistent local-motion region."""

    bbox: tuple[int, int, int, int]
    area_ratio: float
    persistence: float
    mean_heat: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "area_ratio": self.area_ratio,
            "persistence": self.persistence,
            "mean_heat": self.mean_heat,
            "confidence": self.confidence,
        }


@dataclass
class MotionNoiseResult:
    """Persistent-motion mask and metrics."""

    dynamic_noise_mask: Any
    stable_region_mask: Any
    motion_heatmap: Any
    regions: list[MotionNoiseRegion]
    raw_change_ratios: list[float]
    masked_change_ratios: list[float]
    metrics: dict[str, Any]

    @property
    def has_dynamic_noise(self) -> bool:
        return bool(self.metrics.get("has_dynamic_noise", False))

    def metrics_dict(self) -> dict[str, Any]:
        out = dict(self.metrics)
        out["regions"] = [region.as_dict() for region in self.regions]
        return out


class MotionNoiseAnalyzer:
    """Detect persistent local motion that should be ignored by page-response checks."""

    def __init__(
        self,
        *,
        diff_threshold: int = 22,
        persistence_threshold: float = 0.45,
        min_region_area_ratio: float = 0.02,
        max_region_area_ratio: float = 0.45,
        max_total_mask_area_ratio: float = 0.60,
        min_stable_area_ratio: float = 0.30,
        morph_kernel_size: int = 5,
        dilation_kernel_size: int = 9,
    ) -> None:
        self.diff_threshold = max(0, min(255, int(diff_threshold)))
        self.persistence_threshold = max(0.0, min(1.0, float(persistence_threshold)))
        self.min_region_area_ratio = max(0.0, min(1.0, float(min_region_area_ratio)))
        self.max_region_area_ratio = max(0.0, min(1.0, float(max_region_area_ratio)))
        self.max_total_mask_area_ratio = max(0.0, min(1.0, float(max_total_mask_area_ratio)))
        self.min_stable_area_ratio = max(0.0, min(1.0, float(min_stable_area_ratio)))
        self.morph_kernel_size = max(1, int(morph_kernel_size))
        self.dilation_kernel_size = max(1, int(dilation_kernel_size))

    @staticmethod
    def _read_gray_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _diff_mask(self, gray_a: Any, gray_b: Any) -> Any:
        diff = cv2.absdiff(gray_a, gray_b)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask > 0

    @staticmethod
    def _change_ratio(gray_a: Any, gray_b: Any, threshold: int, ignore_mask: Any | None = None) -> float:
        diff = cv2.absdiff(gray_a, gray_b)
        _, changed = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        changed_bool = changed > 0
        if ignore_mask is None:
            valid = np.ones(changed_bool.shape, dtype=bool)
        else:
            valid = ~ignore_mask.astype(bool)
        valid_pixels = int(valid.sum())
        if valid_pixels <= 0:
            return float(changed_bool.mean())
        return int((changed_bool & valid).sum()) / valid_pixels

    def analyze(self, sampled_frames: list[Any]) -> MotionNoiseResult:
        if len(sampled_frames) < 2:
            empty = np.zeros((1, 1), dtype=bool)
            return MotionNoiseResult(
                dynamic_noise_mask=empty,
                stable_region_mask=~empty,
                motion_heatmap=np.zeros((1, 1), dtype=np.float32),
                regions=[],
                raw_change_ratios=[],
                masked_change_ratios=[],
                metrics={
                    "enabled": True,
                    "has_dynamic_noise": False,
                    "reason": "insufficient_frames",
                    "transition_count": 0,
                },
            )

        grays = [self._read_gray_image(Path(frame.image_path)) for frame in sampled_frames]
        base_shape = grays[0].shape
        if any(gray.shape != base_shape for gray in grays):
            raise ValueError("motion noise analysis requires frames with identical dimensions")

        raw_change_ratios: list[float] = []
        transition_masks: list[Any] = []
        for idx in range(len(grays) - 1):
            raw_change_ratios.append(self._change_ratio(grays[idx], grays[idx + 1], self.diff_threshold))
            transition_masks.append(self._diff_mask(grays[idx], grays[idx + 1]))

        transition_count = len(transition_masks)
        heatmap = np.mean(np.stack(transition_masks, axis=0).astype(np.float32), axis=0)
        persistent = heatmap >= self.persistence_threshold
        h, w = persistent.shape
        total_pixels = max(1, h * w)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            persistent.astype(np.uint8),
            connectivity=8,
        )
        regions: list[MotionNoiseRegion] = []
        mask = np.zeros((h, w), dtype=bool)

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            area_ratio = area / total_pixels
            if area_ratio < self.min_region_area_ratio or area_ratio > self.max_region_area_ratio:
                continue

            region_mask = labels == label
            mean_heat = float(heatmap[region_mask].mean())
            persistence = float((heatmap[region_mask] >= self.persistence_threshold).mean())
            confidence = min(1.0, mean_heat * persistence)
            regions.append(
                MotionNoiseRegion(
                    bbox=(x, y, x + width, y + height),
                    area_ratio=area_ratio,
                    persistence=persistence,
                    mean_heat=mean_heat,
                    confidence=confidence,
                )
            )
            mask |= region_mask

        if bool(mask.any()):
            kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0

        total_mask_area_ratio = float(mask.mean())
        stable_area_ratio = 1.0 - total_mask_area_ratio
        reject_reason = ""
        if total_mask_area_ratio > self.max_total_mask_area_ratio:
            reject_reason = "mask_area_too_large"
            mask = np.zeros((h, w), dtype=bool)
            total_mask_area_ratio = 0.0
            stable_area_ratio = 1.0
        elif stable_area_ratio < self.min_stable_area_ratio:
            reject_reason = "stable_area_too_small"
            mask = np.zeros((h, w), dtype=bool)
            total_mask_area_ratio = 0.0
            stable_area_ratio = 1.0

        has_dynamic_noise = bool(mask.any())
        stable_region_mask = ~mask
        masked_change_ratios = [
            self._change_ratio(grays[idx], grays[idx + 1], self.diff_threshold, ignore_mask=mask)
            for idx in range(len(grays) - 1)
        ]

        metrics = {
            "enabled": True,
            "has_dynamic_noise": has_dynamic_noise,
            "reason": "detected" if has_dynamic_noise else (reject_reason or "not_detected"),
            "transition_count": transition_count,
            "diff_threshold": self.diff_threshold,
            "persistence_threshold": self.persistence_threshold,
            "min_region_area_ratio": self.min_region_area_ratio,
            "max_region_area_ratio": self.max_region_area_ratio,
            "max_total_mask_area_ratio": self.max_total_mask_area_ratio,
            "dynamic_noise_area_ratio": total_mask_area_ratio,
            "stable_area_ratio": stable_area_ratio,
            "region_count": len(regions),
            "raw_change_mean": float(np.mean(raw_change_ratios)) if raw_change_ratios else 0.0,
            "masked_change_mean": float(np.mean(masked_change_ratios)) if masked_change_ratios else 0.0,
        }
        return MotionNoiseResult(
            dynamic_noise_mask=mask,
            stable_region_mask=stable_region_mask,
            motion_heatmap=heatmap,
            regions=regions,
            raw_change_ratios=raw_change_ratios,
            masked_change_ratios=masked_change_ratios,
            metrics=metrics,
        )
