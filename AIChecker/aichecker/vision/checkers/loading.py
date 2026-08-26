"""加载异常检测模块：CV-first + VLM-fallback。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from aichecker.vision.evaluator import VisionEvaluator
from aichecker.vision.motion_noise import MotionNoiseAnalyzer
from aichecker.vision.prompt_builders import build_prompt_for_type


@dataclass
class LoadingDetectionResult:
    """loading 检测结果。"""

    bug_detected: bool
    task_intent: str
    reason: str
    decision_basis: str
    anomaly_type: str
    anomaly_types: list[str]
    raw_response: str
    decision_source: str
    cv_metrics: dict[str, Any]
    timing: dict[str, float] | None = None


class LoadingDetector:
    """优先使用 CV 信号判定加载异常，不确定时回退到 VLM。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        cv_static_mean_threshold: float = 0.006,
        cv_static_max_threshold: float = 0.015,
        cv_spinner_mean_upper: float = 0.03,
        cv_spinner_std_upper: float = 0.008,
        cv_clear_progress_max_threshold: float = 0.08,
        cv_black_white_tail_window: int = 5,
        cv_black_white_min_consecutive: int = 3,
        cv_black_white_min_window_ratio: float = 0.6,
        cv_garbled_min_consecutive: int = 2,
        cv_garbled_band_top_ratio: float = 0.35,
        cv_garbled_band_bottom_ratio: float = 0.55,
        cv_noise_garble_block_std: float = 6.0,
        cv_noise_garble_residual: float = 3.5,
        cv_noise_garble_block_std_high: float = 8.0,
        cv_localized_garble_block: int = 16,
        cv_localized_garble_resid_threshold: float = 12.0,
        cv_localized_garble_rnd_cluster: int = 150,
        cv_localized_garble_pure_cluster: int = 180,
        cv_localized_garble_min_frames: int = 1,
        cv_min_long_loading_duration_sec: float = 5.0,
        cv_transient_spike_max_transitions: int = 2,
        enable_motion_noise_mask: bool = True,
        motion_noise_analyzer: MotionNoiseAnalyzer | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.cv_static_mean_threshold = cv_static_mean_threshold
        self.cv_static_max_threshold = cv_static_max_threshold
        self.cv_spinner_mean_upper = cv_spinner_mean_upper
        self.cv_spinner_std_upper = cv_spinner_std_upper
        self.cv_clear_progress_max_threshold = cv_clear_progress_max_threshold
        self.cv_black_white_tail_window = max(1, int(cv_black_white_tail_window))
        self.cv_black_white_min_consecutive = max(1, int(cv_black_white_min_consecutive))
        self.cv_black_white_min_window_ratio = max(0.0, min(1.0, float(cv_black_white_min_window_ratio)))
        self.cv_garbled_min_consecutive = max(1, int(cv_garbled_min_consecutive))
        self.cv_garbled_band_top_ratio = max(0.0, min(1.0, float(cv_garbled_band_top_ratio)))
        self.cv_garbled_band_bottom_ratio = max(0.0, min(1.0, float(cv_garbled_band_bottom_ratio)))
        if self.cv_garbled_band_bottom_ratio <= self.cv_garbled_band_top_ratio:
            self.cv_garbled_band_bottom_ratio = min(1.0, self.cv_garbled_band_top_ratio + 0.2)
        self.cv_noise_garble_block_std = max(0.0, float(cv_noise_garble_block_std))
        self.cv_noise_garble_residual = max(0.0, float(cv_noise_garble_residual))
        self.cv_noise_garble_block_std_high = max(
            self.cv_noise_garble_block_std, float(cv_noise_garble_block_std_high)
        )
        self.cv_localized_garble_block = max(8, int(cv_localized_garble_block))
        self.cv_localized_garble_resid_threshold = max(0.0, float(cv_localized_garble_resid_threshold))
        self.cv_localized_garble_rnd_cluster = max(1, int(cv_localized_garble_rnd_cluster))
        self.cv_localized_garble_pure_cluster = max(1, int(cv_localized_garble_pure_cluster))
        self.cv_localized_garble_min_frames = max(1, int(cv_localized_garble_min_frames))
        self.cv_min_long_loading_duration_sec = max(0.0, float(cv_min_long_loading_duration_sec))
        self.cv_transient_spike_max_transitions = max(1, int(cv_transient_spike_max_transitions))
        self.enable_motion_noise_mask = bool(enable_motion_noise_mask)
        self.motion_noise_analyzer = motion_noise_analyzer or MotionNoiseAnalyzer()

    @staticmethod
    def _read_gray_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _read_color_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return image

    @staticmethod
    def _calc_change_ratio(gray_a: Any, gray_b: Any, threshold: int = 22) -> float:
        diff = cv2.absdiff(gray_a, gray_b)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(binary.shape[0] * binary.shape[1]))
        return changed_pixels / total_pixels

    @staticmethod
    def _build_stats(ratios: list[float]) -> dict[str, float]:
        if not ratios:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        mean = sum(ratios) / len(ratios)
        variance = sum((x - mean) ** 2 for x in ratios) / len(ratios)
        std = variance**0.5
        return {"mean": mean, "max": max(ratios), "min": min(ratios), "std": std}

    @staticmethod
    def _longest_run_at_or_below(ratios: list[float], threshold: float) -> tuple[int, int, int]:
        """Return longest consecutive run with ratio <= threshold as (length, start, end)."""
        best_len = 0
        best_start = -1
        best_end = -1
        current_len = 0
        current_start = 0

        for idx, ratio in enumerate(ratios):
            if ratio <= threshold:
                if current_len == 0:
                    current_start = idx
                current_len += 1
                if current_len > best_len:
                    best_len = current_len
                    best_start = current_start
                    best_end = idx
            else:
                current_len = 0

        return best_len, best_start, best_end

    def _calc_baseline_change_ratios(self, sampled_frames: list[Any]) -> list[float]:
        """每帧相对起始帧（点击前）的变化比例。"""
        if len(sampled_frames) < 2:
            return []
        gray0 = self._read_gray_image(sampled_frames[0].image_path)
        ratios: list[float] = []
        for frame in sampled_frames[1:]:
            gray = self._read_gray_image(frame.image_path)
            ratios.append(self._calc_change_ratio(gray0, gray))
        return ratios

    def _has_stable_baseline_progress(
        self,
        baseline_diffs: list[float],
        *,
        tail_frame_indices: list[int] | None = None,
    ) -> bool:
        """
        是否相对起始状态形成了稳定新状态。

        仅尾段持续达到“明显变化”档，才视为真正响应；微弱漂移或单帧闪一下不算。
        """
        if not baseline_diffs:
            return False
        if tail_frame_indices:
            vals = [
                float(baseline_diffs[idx - 1])
                for idx in tail_frame_indices
                if idx >= 1 and (idx - 1) < len(baseline_diffs)
            ]
        else:
            window = max(1, int(round(len(baseline_diffs) * 0.2)))
            vals = [float(x) for x in baseline_diffs[-window:]]
        if not vals:
            return False
        stats = self._build_stats(vals)
        # 尾段整体达到 clear-progress 强度，且每帧都明显不同于起始态。
        return (
            float(stats["mean"]) >= self.cv_clear_progress_max_threshold
            and float(stats["min"]) >= self.cv_static_max_threshold
        )

    def _suppress_transient_high_spikes(self, ratios: list[float]) -> tuple[list[float], list[list[int]]]:
        """
        去掉短脉冲大跳变：大跳变次数很少，且去掉后序列回到静止档，则视为噪音。

        典型：按钮按下/抬起各造成一次相邻帧跳变，中间不一定连续高值。
        """
        th = self.cv_clear_progress_max_threshold
        high_idxs = [idx for idx, ratio in enumerate(ratios) if float(ratio) >= th]
        if not high_idxs or len(high_idxs) > self.cv_transient_spike_max_transitions:
            return list(ratios), []
        trial = list(ratios)
        for idx in high_idxs:
            trial[idx] = 0.0
        trial_stats = self._build_stats(trial)
        if (
            float(trial_stats["mean"]) <= self.cv_static_mean_threshold
            and float(trial_stats["max"]) <= self.cv_static_max_threshold
        ):
            spans = [[idx, idx + 1] for idx in high_idxs]
            return trial, spans
        return list(ratios), []

    def _build_stage_metrics(self, sampled_frames: list[Any], ratios: list[float]) -> dict[str, Any]:
        total_frames = len(sampled_frames)
        total_transitions = len(ratios)
        first_ts = float(getattr(sampled_frames[0], "timestamp_sec", 0.0)) if sampled_frames else 0.0
        last_ts = float(getattr(sampled_frames[-1], "timestamp_sec", first_ts)) if sampled_frames else first_ts
        duration_sec = max(0.0, last_ts - first_ts)

        frame_window = max(1, int(round(total_frames * 0.2))) if total_frames else 0
        early_frame_indices = list(range(0, min(total_frames, frame_window)))
        tail_frame_indices = list(range(max(0, total_frames - frame_window), total_frames))
        middle_start = len(early_frame_indices)
        middle_end = tail_frame_indices[0] if tail_frame_indices else total_frames
        middle_frame_indices = list(range(middle_start, max(middle_start, middle_end)))

        transition_window = max(1, int(round(total_transitions * 0.2))) if total_transitions else 0
        early_ratios = ratios[:transition_window] if transition_window else []
        tail_ratios = ratios[-transition_window:] if transition_window else []
        middle_ratios = ratios[transition_window : total_transitions - transition_window] if transition_window else []

        return {
            "frame_count": total_frames,
            "transition_count": total_transitions,
            "first_timestamp_sec": first_ts,
            "last_timestamp_sec": last_ts,
            "duration_sec": duration_sec,
            "early_frame_indices": early_frame_indices,
            "middle_frame_indices": middle_frame_indices,
            "tail_frame_indices": tail_frame_indices,
            "early_change_stats": self._build_stats(early_ratios),
            "middle_change_stats": self._build_stats(middle_ratios),
            "tail_change_stats": self._build_stats(tail_ratios),
        }

    @staticmethod
    def _calc_screen_stats(gray: Any) -> dict[str, float]:
        total = max(1, int(gray.shape[0] * gray.shape[1]))
        black_pixels = int((gray <= 16).sum())
        white_pixels = int((gray >= 239).sum())
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        return {
            "mean_luma": mean_val,
            "std_luma": std_val,
            "black_ratio": black_pixels / total,
            "white_ratio": white_pixels / total,
        }

    def _calc_garbled_screen_stats(self, gray: Any) -> dict[str, float | bool]:
        """
        识别局部花屏/乱码：文本带内多行同时高方差，且水平边缘密度异常偏低。
        """
        h, w = gray.shape[:2]
        y0 = int(h * self.cv_garbled_band_top_ratio)
        y1 = int(h * self.cv_garbled_band_bottom_ratio)
        if y1 <= y0 + 8 or w < 32:
            return {
                "garbled_match": False,
                "noisy_rows": 0.0,
                "row_osc": 0.0,
                "hdiff_mean": 0.0,
                "text_band_p50": 0.0,
                "text_band_p90": 0.0,
                "text_band_max": 0.0,
            }

        band = gray[y0:y1, :]
        row_stds = band.std(axis=1)
        noisy_rows = float((row_stds > 35.0).mean())
        row_osc = float(np.abs(np.diff(band.mean(axis=1))).mean())
        hdiff_mean = float(np.abs(band[:, 1:].astype(np.int16) - band[:, :-1].astype(np.int16)).mean())

        block_size = 16
        block_stds: list[float] = []
        for y in range(y0, y1 - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_stds.append(float(gray[y : y + block_size, x : x + block_size].std()))

        if block_stds:
            bs_arr = np.array(block_stds, dtype=np.float64)
            p50 = float(np.percentile(bs_arr, 50))
            p90 = float(np.percentile(bs_arr, 90))
            mx = float(bs_arr.max())
        else:
            p50 = p90 = mx = 0.0

        garbled_match = (
            noisy_rows >= 0.85
            and mx >= 90.0
            and p90 >= 35.0
            and hdiff_mean <= 3.0
            and row_osc <= 1.5
        )
        return {
            "garbled_match": garbled_match,
            "noisy_rows": noisy_rows,
            "row_osc": row_osc,
            "hdiff_mean": hdiff_mean,
            "text_band_p50": p50,
            "text_band_p90": p90,
            "text_band_max": mx,
        }

    def _calc_noise_garble_stats(self, gray: Any) -> dict[str, float | bool]:
        """
        识别全屏花屏：包含颗粒噪声/雪花点（含 RGB 通道错位）以及撕裂/重影条纹。
        两类异常都会破坏本应平坦的区域，使分块标准差中位数显著升高；
        噪点类还伴随较高的中值滤波残差。
        """
        h, w = gray.shape[:2]
        if h < 16 or w < 16:
            return {
                "noise_garble_match": False,
                "noise_residual": 0.0,
                "median_block_std": 0.0,
            }

        gray_u8 = gray.astype(np.uint8)
        median = cv2.medianBlur(gray_u8, 3).astype(np.float32)
        noise_residual = float(np.abs(gray.astype(np.float32) - median).mean())

        block_size = 8
        hh = (h // block_size) * block_size
        ww = (w // block_size) * block_size
        blocks = (
            gray[:hh, :ww]
            .astype(np.float32)
            .reshape(hh // block_size, block_size, ww // block_size, block_size)
            .transpose(0, 2, 1, 3)
            .reshape(-1, block_size * block_size)
        )
        median_block_std = float(np.median(blocks.std(axis=1)))

        # 强撕裂/重影或强噪点：平坦区中位方差本身就异常高。
        # 弱噪点：中位方差偏高，且中值残差（高频雪花）明显。
        noise_garble_match = median_block_std >= self.cv_noise_garble_block_std_high or (
            median_block_std >= self.cv_noise_garble_block_std
            and noise_residual >= self.cv_noise_garble_residual
        )
        return {
            "noise_garble_match": noise_garble_match,
            "noise_residual": noise_residual,
            "median_block_std": median_block_std,
        }

    @staticmethod
    def _largest_block_cluster(block_mask: Any) -> int:
        """在块级布尔掩码上求最大 4-连通连通域的块数。"""
        mask = block_mask.astype(np.uint8)
        if int(mask.sum()) == 0:
            return 0
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        if num <= 1:
            return 0
        return int(stats[1:, cv2.CC_STAT_AREA].max())

    def _calc_localized_garble_stats(self, bgr: Any) -> dict[str, float | bool]:
        """
        识别局部块状花屏：画面整体正常，但某一矩形区域坏成随机雪花或纯原色彩条。
        - 随机雪花：中值滤波残差在局部聚成大团（结构化照片会被中值滤波保留，残差小）。
        - 纯原色彩条：完全饱和的原色/间色（纯红/绿/蓝/青/黄/品红）块聚成大团，
          正常 UI/照片几乎不会出现如此大片的纯色簇。
        """
        h, w = bgr.shape[:2]
        bs = self.cv_localized_garble_block
        if h < bs * 2 or w < bs * 2:
            return {
                "localized_garble_match": False,
                "rnd_noise_cluster": 0.0,
                "pure_color_cluster": 0.0,
            }

        channels = cv2.split(bgr.astype(np.int16))
        b_ch, g_ch, r_ch = channels[0], channels[1], channels[2]
        max_ch = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        min_ch = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        pure = ((max_ch >= 225) & (min_ch <= 45)).astype(np.float32)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3)).astype(np.float32)

        hh = (h // bs) * bs
        ww = (w // bs) * bs
        rows = hh // bs
        cols = ww // bs

        def _block_mean(arr: Any) -> Any:
            return arr[:hh, :ww].reshape(rows, bs, cols, bs).mean(axis=(1, 3))

        pure_block_mask = _block_mean(pure) > 0.5
        rnd_block_mask = _block_mean(residual) > self.cv_localized_garble_resid_threshold

        rnd_cluster = self._largest_block_cluster(rnd_block_mask)
        pure_cluster = self._largest_block_cluster(pure_block_mask)

        localized_garble_match = (
            rnd_cluster >= self.cv_localized_garble_rnd_cluster
            or pure_cluster >= self.cv_localized_garble_pure_cluster
        )
        return {
            "localized_garble_match": localized_garble_match,
            "rnd_noise_cluster": float(rnd_cluster),
            "pure_color_cluster": float(pure_cluster),
        }

    @staticmethod
    def _max_consecutive_true(flags: list[bool]) -> int:
        best = 0
        current = 0
        for flag in flags:
            if flag:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _detect_black_white_screen(self, sampled_frames: list[Any]) -> tuple[str | None, str, dict[str, Any]]:
        """
        识别黑/白屏与花屏异常。黑/白屏只在末尾窗口持续命中时判定；花屏扫描全段帧并要求连续命中。
        """
        tail_window = min(len(sampled_frames), self.cv_black_white_tail_window)
        required_consecutive = min(tail_window, self.cv_black_white_min_consecutive)
        tail_frames = sampled_frames[-tail_window:]
        frame_stats: list[dict[str, Any]] = []

        for frame_idx, frame in enumerate(tail_frames, start=len(sampled_frames) - tail_window):
            gray = self._read_gray_image(frame.image_path)
            stats = self._calc_screen_stats(gray)
            black_match = float(stats["black_ratio"]) >= 0.92 and float(stats["std_luma"]) <= 12.0
            white_match = float(stats["white_ratio"]) >= 0.92 and float(stats["std_luma"]) <= 12.0
            frame_stats.append(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": float(getattr(frame, "timestamp_sec", 0.0)),
                    "black_match": black_match,
                    "white_match": white_match,
                    **stats,
                }
            )

        def _terminal_run_count(key: str) -> int:
            count = 0
            for item in reversed(frame_stats):
                if bool(item[key]):
                    count += 1
                else:
                    break
            return count

        black_count = sum(1 for item in frame_stats if bool(item["black_match"]))
        white_count = sum(1 for item in frame_stats if bool(item["white_match"]))
        black_terminal_run = _terminal_run_count("black_match")
        white_terminal_run = _terminal_run_count("white_match")
        black_window_ratio = black_count / max(1, tail_window)
        white_window_ratio = white_count / max(1, tail_window)
        metrics: dict[str, Any] = {
            "tail_window": tail_window,
            "required_consecutive": required_consecutive,
            "required_window_ratio": self.cv_black_white_min_window_ratio,
            "black_match_count": black_count,
            "white_match_count": white_count,
            "black_terminal_run": black_terminal_run,
            "white_terminal_run": white_terminal_run,
            "black_window_ratio": black_window_ratio,
            "white_window_ratio": white_window_ratio,
            "last_frame": frame_stats[-1] if frame_stats else {},
            "tail_frames": frame_stats,
        }

        if (
            black_terminal_run >= required_consecutive
            and black_window_ratio >= self.cv_black_white_min_window_ratio
        ):
            return "black_screen", "CV判定末尾窗口持续为近纯黑画面，疑似黑屏异常。", metrics
        if (
            white_terminal_run >= required_consecutive
            and white_window_ratio >= self.cv_black_white_min_window_ratio
        ):
            return "white_screen", "CV判定末尾窗口持续为近纯白画面，疑似白屏异常。", metrics

        garbled_frame_stats: list[dict[str, Any]] = []
        garbled_flags: list[bool] = []
        localized_flags: list[bool] = []
        for frame_idx, frame in enumerate(sampled_frames):
            color = self._read_color_image(frame.image_path)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            band_stats = self._calc_garbled_screen_stats(gray)
            noise_stats = self._calc_noise_garble_stats(gray)
            localized_stats = self._calc_localized_garble_stats(color)
            band_match = bool(band_stats["garbled_match"])
            noise_match = bool(noise_stats["noise_garble_match"])
            localized_match = bool(localized_stats["localized_garble_match"])
            garbled_match = band_match or noise_match
            garbled_flags.append(garbled_match)
            localized_flags.append(localized_match)
            garbled_frame_stats.append(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": float(getattr(frame, "timestamp_sec", 0.0)),
                    "garbled_match": garbled_match or localized_match,
                    "band_match": band_match,
                    "noise_match": noise_match,
                    "localized_match": localized_match,
                    **{k: v for k, v in band_stats.items() if k != "garbled_match"},
                    **{k: v for k, v in noise_stats.items() if k != "noise_garble_match"},
                    **{k: v for k, v in localized_stats.items() if k != "localized_garble_match"},
                }
            )

        garbled_max_consecutive = self._max_consecutive_true(garbled_flags)
        garbled_match_count = sum(1 for flag in garbled_flags if flag)
        band_match_count = sum(1 for item in garbled_frame_stats if item.get("band_match"))
        noise_match_count = sum(1 for item in garbled_frame_stats if item.get("noise_match"))
        localized_match_count = sum(1 for flag in localized_flags if flag)
        garbled_metrics: dict[str, Any] = {
            "required_consecutive": self.cv_garbled_min_consecutive,
            "garbled_match_count": garbled_match_count,
            "garbled_max_consecutive": garbled_max_consecutive,
            "band_match_count": band_match_count,
            "noise_match_count": noise_match_count,
            "localized_match_count": localized_match_count,
            "localized_min_frames": self.cv_localized_garble_min_frames,
            "garbled_band_top_ratio": self.cv_garbled_band_top_ratio,
            "garbled_band_bottom_ratio": self.cv_garbled_band_bottom_ratio,
            "noise_garble_block_std_threshold": self.cv_noise_garble_block_std,
            "noise_garble_residual_threshold": self.cv_noise_garble_residual,
            "noise_garble_block_std_high_threshold": self.cv_noise_garble_block_std_high,
            "localized_rnd_cluster_threshold": self.cv_localized_garble_rnd_cluster,
            "localized_pure_cluster_threshold": self.cv_localized_garble_pure_cluster,
            "frames": garbled_frame_stats,
        }
        metrics["garbled"] = garbled_metrics

        # 局部块状花屏：区域坏得足够大且足够罕见，允许单帧命中（往往是一闪而过）。
        if localized_match_count >= self.cv_localized_garble_min_frames:
            return (
                "garbled_screen",
                "CV判定画面局部出现大块随机雪花或纯色彩条，疑似局部渲染/解码花屏。",
                metrics,
            )
        if garbled_max_consecutive >= self.cv_garbled_min_consecutive:
            if noise_match_count >= band_match_count:
                reason = "CV判定画面出现持续全屏噪点/雪花花屏（含通道错位），疑似渲染或解码异常。"
            else:
                reason = "CV判定画面中部文本区域出现持续花屏/乱码条纹，疑似渲染异常。"
            return "garbled_screen", reason, metrics
        return None, "", metrics

    def _cv_decide(
        self,
        ratios: list[float],
        duration_sec: float | None = None,
        *,
        stable_progress: bool | None = None,
        _spike_filter_applied: bool = False,
    ) -> tuple[bool | None, str, str, dict[str, Any]]:
        stats = self._build_stats(ratios)
        metrics: dict[str, Any] = {
            "frame_change_ratios": ratios,
            **stats,
            "duration_sec": duration_sec,
            "min_long_loading_duration_sec": self.cv_min_long_loading_duration_sec,
            "stable_baseline_progress": stable_progress,
        }
        if not ratios:
            return None, "CV证据不足：缺少相邻帧变化数据。", "unknown", metrics

        mean_ratio = float(stats["mean"])
        max_ratio = float(stats["max"])
        std_ratio = float(stats["std"])

        if mean_ratio <= self.cv_static_mean_threshold and max_ratio <= self.cv_static_max_threshold:
            return (
                True,
                "CV判定页面长时间近乎静止，疑似加载无反馈/卡住。",
                "no_response",
                metrics,
            )

        # 低变化长跑优先于“相对起始态已切换”：点击后进入 loading 并卡住时，
        # 尾段相对起始帧也会稳定变样，不能据此直接判有效响应。
        low_run_len, low_run_start, low_run_end = self._longest_run_at_or_below(
            ratios,
            self.cv_spinner_mean_upper,
        )
        if duration_sec is not None and ratios:
            seconds_per_transition = duration_sec / max(1, len(ratios))
            low_run_duration_sec = low_run_len * seconds_per_transition
        else:
            seconds_per_transition = None
            low_run_duration_sec = None
        metrics["longest_low_change_run"] = {
            "threshold": self.cv_spinner_mean_upper,
            "transition_count": low_run_len,
            "start_transition_index": low_run_start,
            "end_transition_index": low_run_end,
            "estimated_duration_sec": low_run_duration_sec,
            "seconds_per_transition": seconds_per_transition,
        }
        if low_run_duration_sec is not None and low_run_duration_sec > self.cv_min_long_loading_duration_sec:
            return (
                True,
                "CV判定存在持续小幅规律变化的低变化区间，疑似加载态长时间停留。",
                "long_loading",
                metrics,
            )

        # 无长时间低变化区间时，相对起始状态已形成稳定新外观 → 视为真正响应。
        if stable_progress is True:
            return (
                False,
                "CV判定相对起始状态出现稳定变化，视为有效响应。",
                "none",
                metrics,
            )

        if max_ratio >= self.cv_clear_progress_max_threshold:
            # 未形成稳定新状态：短脉冲大跳变视为噪音（如按钮按下闪一下）。
            if stable_progress is False and not _spike_filter_applied:
                cleaned, spans = self._suppress_transient_high_spikes(ratios)
                metrics["transient_spike_filter"] = {
                    "applied": bool(spans),
                    "suppressed_spans": spans,
                    "max_pulse_transitions": self.cv_transient_spike_max_transitions,
                }
                if spans:
                    nested_result, nested_reason, nested_type, nested_metrics = self._cv_decide(
                        cleaned,
                        duration_sec=duration_sec,
                        stable_progress=False,
                        _spike_filter_applied=True,
                    )
                    nested_metrics["transient_spike_filter"] = metrics["transient_spike_filter"]
                    nested_metrics["frame_change_ratios_before_spike_filter"] = list(ratios)
                    nested_metrics["stable_baseline_progress"] = False
                    return nested_result, nested_reason, nested_type, nested_metrics
                # 无法解释为短脉冲：不要仅凭 max 判“有响应”，继续后续分支。
            elif stable_progress is None:
                # 未提供基线信息时保持旧行为（兼容直接调用 _cv_decide 的单测）。
                return (
                    False,
                    "CV判定存在明显页面变化，未见持续加载卡死特征。",
                    "none",
                    metrics,
                )

        if mean_ratio <= self.cv_spinner_mean_upper and std_ratio <= self.cv_spinner_std_upper:
            if duration_sec is not None and duration_sec < self.cv_min_long_loading_duration_sec:
                metrics["long_loading_candidate"] = True
                return (
                    None,
                    (
                        "CV观察到持续小幅变化，但视频跨度不足"
                        f"{self.cv_min_long_loading_duration_sec:.1f}s，暂不直接判定长时间加载。"
                    ),
                    "unknown",
                    metrics,
                )
            return (
                True,
                "CV判定持续小幅规律变化，疑似加载指示器长期存在。",
                "long_loading",
                metrics,
            )

        return (
            None,
            "CV判定不确定：变化特征介于静止与明显进展之间，转交VLM语义判定。",
            "unknown",
            metrics,
        )

    @staticmethod
    def _unique_keep_order(items: list[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            if item and item not in out:
                out.append(item)
        return out

    @staticmethod
    def _pick_primary_anomaly(anomaly_types: list[str]) -> str:
        if not anomaly_types:
            return "none"
        priority = [
            "black_screen",
            "white_screen",
            "garbled_screen",
            "no_response",
            "long_loading",
            "unknown",
            "none",
        ]
        for p in priority:
            if p in anomaly_types:
                return p
        return anomaly_types[0]

    def _run_loading_fallback(
        self,
        sampled_frames: list[Any],
        task_id: str,
        *,
        prompt_type: str = "loading",
    ) -> dict[str, Any]:
        """
        VLM 兜底语义判断（首尾帧）。

        prompt_type:
        - loading: 长时间加载检测
        - no_response: 无响应检测（与 long_loading 分离）
        """
        before = sampled_frames[0]
        after = sampled_frames[-1]
        normalized_prompt = (prompt_type or "loading").strip().lower() or "loading"
        prompt_pack = build_prompt_for_type(
            task_type=normalized_prompt,
            context={
                "task_type": normalized_prompt,
                "before_timestamp_sec": before.timestamp_sec,
                "after_timestamp_sec": after.timestamp_sec,
            },
        )
        required_fields: dict[str, type] = {
            "bug_detected": bool,
            "reason": str,
            "decision_basis": str,
            "anomaly_type": str,
        }
        if normalized_prompt == "no_response":
            required_fields["has_effective_response"] = bool

        t_vlm_start = time.perf_counter()
        eval_result = self.evaluator.evaluate_json(
            before_image=before.image_path,
            after_image=after.image_path,
            task_id=f"{task_id}_{normalized_prompt}_fallback",
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            required_fields=required_fields,
        )
        elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        parsed = eval_result.parsed_json
        anomaly_type = str(parsed.get("anomaly_type", "unknown")).strip().lower() or "unknown"
        bug_detected = bool(parsed["bug_detected"])
        has_effective_response = parsed.get("has_effective_response", None)

        # no_response 专用：以 has_effective_response 为准，避免和 long_loading 混报。
        if normalized_prompt == "no_response":
            if has_effective_response is False:
                bug_detected = True
                anomaly_type = "no_response"
            elif has_effective_response is True:
                bug_detected = False
                anomaly_type = "none"
            elif bug_detected and anomaly_type in {"long_loading", "unknown", ""}:
                anomaly_type = "no_response"
            elif (not bug_detected) and anomaly_type == "no_response":
                anomaly_type = "none"

        return {
            "bug_detected": bug_detected,
            "reason": str(parsed["reason"]),
            "decision_basis": str(parsed.get("decision_basis", parsed["reason"])),
            "anomaly_type": anomaly_type,
            "has_effective_response": has_effective_response,
            "prompt_type": normalized_prompt,
            "raw_response": eval_result.raw_response,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def detect(
        self,
        sampled_frames: list[Any],
        task_id: str,
        *,
        enable_vlm_fallback: bool = True,
        vlm_prompt_type: str = "loading",
    ) -> LoadingDetectionResult:
        """对整段 sampled_frames 做 loading / no_response 检测。

        enable_vlm_fallback=False 时只跑 CV，仍会标记 needs_vlm，供上层先汇总结论再决定是否问 VLM。
        vlm_prompt_type:
        - loading: 长时间加载 VLM
        - no_response: 无响应 VLM（推荐无响应回归使用）
        """
        if len(sampled_frames) < 2:
            raise ValueError("loading 检测至少需要 2 帧")
        t_start = time.perf_counter()

        raw_ratios: list[float] = []
        for idx in range(len(sampled_frames) - 1):
            gray_a = self._read_gray_image(sampled_frames[idx].image_path)
            gray_b = self._read_gray_image(sampled_frames[idx + 1].image_path)
            raw_ratios.append(self._calc_change_ratio(gray_a, gray_b))

        ratios = raw_ratios
        motion_noise_metrics: dict[str, Any] = {
            "enabled": self.enable_motion_noise_mask,
            "has_dynamic_noise": False,
            "reason": "disabled" if not self.enable_motion_noise_mask else "not_run",
        }
        raw_stage_metrics = self._build_stage_metrics(sampled_frames, raw_ratios)
        if self.enable_motion_noise_mask:
            try:
                motion_result = self.motion_noise_analyzer.analyze(sampled_frames)
                motion_noise_metrics = motion_result.metrics_dict()
                motion_noise_metrics["raw_change_ratios"] = motion_result.raw_change_ratios
                motion_noise_metrics["masked_change_ratios"] = motion_result.masked_change_ratios
                if motion_result.has_dynamic_noise:
                    ratios = motion_result.masked_change_ratios
            except Exception as exc:  # pragma: no cover - defensive fallback for corrupt frames
                motion_noise_metrics = {
                    "enabled": True,
                    "has_dynamic_noise": False,
                    "reason": "analysis_error",
                    "error": str(exc),
                }

        stage_metrics = self._build_stage_metrics(sampled_frames, ratios)
        baseline_diffs = self._calc_baseline_change_ratios(sampled_frames)
        stable_progress = self._has_stable_baseline_progress(
            baseline_diffs,
            tail_frame_indices=list(stage_metrics.get("tail_frame_indices") or []),
        )
        screen_anomaly_type, screen_reason, screen_metrics = self._detect_black_white_screen(sampled_frames)
        cv_result, cv_reason, cv_anomaly_type, cv_metrics = self._cv_decide(
            ratios,
            duration_sec=float(stage_metrics["duration_sec"]),
            stable_progress=stable_progress,
        )
        cv_metrics["frame_change_ratios_raw"] = raw_ratios
        cv_metrics["raw_change_stats"] = self._build_stats(raw_ratios)
        cv_metrics["baseline_change_ratios"] = baseline_diffs
        cv_metrics["baseline_change_stats"] = self._build_stats(baseline_diffs)
        cv_metrics["stable_baseline_progress"] = stable_progress
        cv_metrics["motion_noise"] = motion_noise_metrics
        cv_metrics["screen_stats"] = screen_metrics
        cv_metrics["stages"] = stage_metrics
        cv_metrics["raw_stages"] = raw_stage_metrics

        no_response_hit = cv_result is True and cv_anomaly_type == "no_response"
        long_loading_hit = cv_result is True and cv_anomaly_type == "long_loading"
        # no_response 只认 CV 明确的无响应，不把 long_loading 强行提升（易误杀正常点击后静止）。
        # long_loading 与 no_response 边界模糊时保持 long_loading / uncertain，交给 VLM 语义判定。
        no_response_detected = no_response_hit
        # 无响应是更严格的无进展；长时间加载检测项可继承无响应命中。
        long_loading_detected = long_loading_hit or no_response_hit

        cv_metrics["signals"] = {
            "no_response": {
                "detected": no_response_detected,
                "status": "detected" if no_response_detected else "not_detected",
                "source": "cv_frame_change",
                "reason": cv_reason if no_response_hit else "",
                "evidence": {
                    "mean_change_ratio": float(cv_metrics["mean"]),
                    "max_change_ratio": float(cv_metrics["max"]),
                    "static_mean_threshold": self.cv_static_mean_threshold,
                    "static_max_threshold": self.cv_static_max_threshold,
                    "duration_sec": float(stage_metrics["duration_sec"]),
                    "cv_anomaly_type": cv_anomaly_type,
                    "stage_change_stats": {
                        "early": stage_metrics["early_change_stats"],
                        "middle": stage_metrics["middle_change_stats"],
                        "tail": stage_metrics["tail_change_stats"],
                    },
                },
            },
            "black_white_screen": {
                "detected": screen_anomaly_type in {"black_screen", "white_screen", "garbled_screen"},
                "status": (
                    "detected"
                    if screen_anomaly_type in {"black_screen", "white_screen", "garbled_screen"}
                    else "not_detected"
                ),
                "source": "cv_tail_window" if screen_anomaly_type in {"black_screen", "white_screen"} else (
                    "cv_garbled_scan" if screen_anomaly_type == "garbled_screen" else "cv_tail_window"
                ),
                "anomaly_type": screen_anomaly_type or "none",
                "reason": screen_reason,
                "evidence": screen_metrics,
            },
            "load_failed": {
                "detected": False,
                "status": "delegated_to_load_failure_prompt_detector",
                "source": "load_failure_prompt_detector",
                "reason": "页面加载失败提示由独立检测器判断。",
                "evidence": {},
            },
            "long_loading": {
                "detected": long_loading_detected,
                "status": (
                    "detected"
                    if long_loading_detected
                    else ("candidate_duration_too_short" if bool(cv_metrics.get("long_loading_candidate")) else "not_detected")
                ),
                "source": (
                    "inherited_from_no_response"
                    if (no_response_hit and not long_loading_hit)
                    else "cv_frame_change"
                ),
                "reason": (
                    cv_reason
                    if long_loading_hit or bool(cv_metrics.get("long_loading_candidate"))
                    else ("无响应命中，长时间加载信号继承为检测到。" if no_response_hit else "")
                ),
                "evidence": {
                    "mean_change_ratio": float(cv_metrics["mean"]),
                    "std_change_ratio": float(cv_metrics["std"]),
                    "spinner_mean_upper": self.cv_spinner_mean_upper,
                    "spinner_std_upper": self.cv_spinner_std_upper,
                    "duration_sec": float(stage_metrics["duration_sec"]),
                    "min_duration_sec": self.cv_min_long_loading_duration_sec,
                    "inherited_from": "no_response" if (no_response_hit and not long_loading_hit) else None,
                    "stage_change_stats": {
                        "early": stage_metrics["early_change_stats"],
                        "middle": stage_metrics["middle_change_stats"],
                        "tail": stage_metrics["tail_change_stats"],
                    },
                },
            },
        }
        cv_state = "uncertain" if cv_result is None else ("positive" if cv_result else "negative")
        # CV 明确：纯静止→no_response；稳定新状态→通过。
        # long_loading / uncertain 对 no_response 语义不清 → 交 VLM，不由 CV 强行升降。
        needs_vlm = cv_state == "uncertain" or (
            long_loading_hit and not bool(stable_progress) and not no_response_hit
        )
        if cv_state == "uncertain":
            vlm_trigger = "cv_uncertain"
        elif needs_vlm:
            vlm_trigger = "long_loading_ambiguous_for_no_response"
        else:
            vlm_trigger = "not_needed"
        cv_metrics["vlm_fallback"] = {
            "needed": needs_vlm,
            "trigger": vlm_trigger,
            "enabled": bool(enable_vlm_fallback),
            "skipped": bool(needs_vlm and not enable_vlm_fallback),
        }

        detected_anomaly_types: list[str] = []
        reason_parts: list[str] = []
        source_flags: list[str] = []
        raw_response_parts: list[str] = []

        if screen_anomaly_type is not None:
            detected_anomaly_types.append(screen_anomaly_type)
            reason_parts.append(screen_reason)
            source_flags.append("cv_black_white")

        fallback_elapsed_ms = 0.0
        delegated_load_failed = {
            "detected": False,
            "status": "delegated_to_load_failure_prompt_detector",
            "source": "load_failure_prompt_detector",
            "reason": "页面加载失败提示由独立检测器判断。",
            "evidence": {"scan_count": 0, "hit_count": 0},
        }
        delegated_failure_probe = {
            "enabled": False,
            "reason": "页面加载失败提示已拆分到 load_failure_prompt_detector 独立检测。",
            "scan_selected_indices": [],
            "scan_probe_indices": [],
            "scan_count": 0,
            "scan_prefilter_scores": [],
            "hit_count": 0,
            "load_failed": False,
            "failure_type": "none",
            "evidence_text": "",
            "confidence": None,
            "reason_detail": "",
            "elapsed_ms": 0.0,
        }

        if cv_state == "positive" and not needs_vlm:
            if cv_anomaly_type not in {"none", "unknown"}:
                detected_anomaly_types.append(cv_anomaly_type)
            reason_parts.append(cv_reason)
            source_flags.append("cv_loading_signal")
            cv_metrics["failure_text_probe"] = dict(delegated_failure_probe)
            cv_metrics["signals"]["load_failed"] = dict(delegated_load_failed)
            reason_parts.append("页面加载失败提示由独立检测器判断。")
        elif cv_state == "negative" and not needs_vlm:
            reason_parts.append(cv_reason)
            source_flags.append("cv_clear_progress")
            cv_metrics["failure_text_probe"] = dict(delegated_failure_probe)
            cv_metrics["signals"]["load_failed"] = dict(delegated_load_failed)
            reason_parts.append("页面加载失败提示由独立检测器判断。")
        else:
            if cv_state == "positive" and cv_anomaly_type not in {"none", "unknown"}:
                detected_anomaly_types.append(cv_anomaly_type)
                reason_parts.append(cv_reason)
                source_flags.append("cv_loading_signal")
            else:
                reason_parts.append(cv_reason)
                source_flags.append("cv_uncertain")

            cv_metrics["failure_text_probe"] = dict(delegated_failure_probe)
            cv_metrics["signals"]["load_failed"] = dict(delegated_load_failed)
            reason_parts.append("页面加载失败提示由独立检测器判断。")

            if enable_vlm_fallback:
                fallback_result = self._run_loading_fallback(
                    sampled_frames=sampled_frames,
                    task_id=task_id,
                    prompt_type=vlm_prompt_type,
                )
                fallback_elapsed_ms = float(fallback_result["elapsed_ms"])
                reason_parts.append(f"VLM兜底：{fallback_result['reason']}")
                reason_parts.append(f"VLM依据：{fallback_result['decision_basis']}")
                source_flags.append("vlm_fallback")
                raw_response_parts.append(f"[loading_fallback]\n{fallback_result['raw_response']}")
                fallback_anomaly_type = str(fallback_result["anomaly_type"]).strip().lower() or "unknown"
                cv_metrics["vlm_fallback"].update(
                    {
                        "bug_detected": bool(fallback_result["bug_detected"]),
                        "anomaly_type": fallback_anomaly_type,
                        "reason": str(fallback_result.get("reason", "")),
                        "prompt_type": str(fallback_result.get("prompt_type") or vlm_prompt_type),
                        "has_effective_response": fallback_result.get("has_effective_response"),
                    }
                )

                prompt_is_no_response = str(fallback_result.get("prompt_type") or vlm_prompt_type).lower() == "no_response"
                if prompt_is_no_response:
                    # 无响应专用路径：去掉 CV 带来的 long_loading 主类型干扰，只回写 no_response。
                    detected_anomaly_types[:] = [x for x in detected_anomaly_types if x != "long_loading"]
                    if bool(fallback_result["bug_detected"]) or fallback_anomaly_type == "no_response":
                        detected_anomaly_types.append("no_response")
                        cv_metrics["signals"]["no_response"] = {
                            **cv_metrics["signals"]["no_response"],
                            "detected": True,
                            "status": "detected",
                            "source": "vlm_fallback",
                            "reason": str(fallback_result.get("reason", "")),
                        }
                    else:
                        cv_metrics["signals"]["no_response"] = {
                            **cv_metrics["signals"]["no_response"],
                            "detected": False,
                            "status": "not_detected",
                            "source": "vlm_fallback",
                            "reason": str(fallback_result.get("reason", "")),
                        }
                else:
                    if bool(fallback_result["bug_detected"]) and fallback_anomaly_type in {
                        "black_screen",
                        "white_screen",
                        "garbled_screen",
                        "no_response",
                        "long_loading",
                        "unknown",
                    }:
                        detected_anomaly_types.append(fallback_anomaly_type)

                    # 用 VLM 结果回写结构化信号，供按检测项读取。
                    if bool(fallback_result["bug_detected"]) and fallback_anomaly_type == "no_response":
                        cv_metrics["signals"]["no_response"] = {
                            **cv_metrics["signals"]["no_response"],
                            "detected": True,
                            "status": "detected",
                            "source": "vlm_fallback",
                            "reason": str(fallback_result.get("reason", "")),
                        }
                        cv_metrics["signals"]["long_loading"] = {
                            **cv_metrics["signals"]["long_loading"],
                            "detected": True,
                            "status": "detected",
                            "source": "inherited_from_no_response",
                            "reason": "VLM判定无响应，长时间加载信号继承为检测到。",
                            "evidence": {
                                **cv_metrics["signals"]["long_loading"].get("evidence", {}),
                                "inherited_from": "no_response",
                            },
                        }
                    elif bool(fallback_result["bug_detected"]) and fallback_anomaly_type == "long_loading":
                        cv_metrics["signals"]["long_loading"] = {
                            **cv_metrics["signals"]["long_loading"],
                            "detected": True,
                            "status": "detected",
                            "source": "vlm_fallback",
                            "reason": str(fallback_result.get("reason", "")),
                        }
                    elif not bool(fallback_result["bug_detected"]):
                        cv_metrics["signals"]["no_response"] = {
                            **cv_metrics["signals"]["no_response"],
                            "detected": False,
                            "status": "not_detected",
                            "source": "vlm_fallback",
                            "reason": str(fallback_result.get("reason", "")),
                        }
            else:
                reason_parts.append("CV结果待VLM确认（本次未启用VLM兜底）。")
                source_flags.append("cv_pending_vlm")

        normalized_anomaly_types = self._unique_keep_order(detected_anomaly_types)
        bug_detected = len([x for x in normalized_anomaly_types if x not in {"none"}]) > 0
        if not bug_detected:
            normalized_anomaly_types = ["none"]

        primary_anomaly = self._pick_primary_anomaly(normalized_anomaly_types)
        final_reason = " | ".join(reason_parts)
        decision_source = "multi_source" if len(set(source_flags)) > 1 else (source_flags[0] if source_flags else "unknown")
        task_intent = "检测页面是否处于异常长时间加载状态（疑似卡死或无反馈）。"

        return LoadingDetectionResult(
            bug_detected=bug_detected,
            task_intent=task_intent,
            reason=final_reason,
            decision_basis=final_reason,
            anomaly_type=primary_anomaly,
            anomaly_types=normalized_anomaly_types,
            raw_response="\n\n".join(raw_response_parts),
            decision_source=decision_source,
            cv_metrics=cv_metrics,
            timing={
                "detect_elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                "failure_probe_elapsed_ms": 0.0,
                "vlm_elapsed_ms": round(fallback_elapsed_ms, 2),
            },
        )
