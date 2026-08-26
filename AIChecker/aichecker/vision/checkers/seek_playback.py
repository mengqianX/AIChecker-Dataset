"""视频 seek 后播放状态检测。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import time
from typing import Any

import cv2


@dataclass
class SeekPlaybackDetectionResult:
    """seek 后播放检测结果。"""

    result: bool
    bug_detected: bool
    task_intent: str
    reason: str
    decision_basis: str
    anomaly_type: str
    raw_response: str
    decision_source: str
    seek_detected: bool
    seek_timestamp_sec: float | None
    seek_source: str
    cv_metrics: dict[str, Any]
    timing: dict[str, float] | None = None


class SeekPlaybackDetector:
    """自动定位 seek 候选点，并判断 seek 后视频是否恢复播放。"""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        post_seek_window_sec: float = 10.0,
        post_seek_settle_sec: float = 0.5,
        long_loading_threshold_sec: float = 5.0,
        seek_roi_change_threshold: float = 0.025,
        seek_full_change_threshold: float = 0.06,
        playback_mean_change_threshold: float = 0.006,
        playback_max_change_threshold: float = 0.02,
        static_mean_change_threshold: float = 0.0025,
        static_max_change_threshold: float = 0.01,
    ) -> None:
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.post_seek_window_sec = max(1.0, float(post_seek_window_sec))
        self.post_seek_settle_sec = max(0.0, float(post_seek_settle_sec))
        self.long_loading_threshold_sec = max(1.0, float(long_loading_threshold_sec))
        self.seek_roi_change_threshold = max(0.0, float(seek_roi_change_threshold))
        self.seek_full_change_threshold = max(0.0, float(seek_full_change_threshold))
        self.playback_mean_change_threshold = max(0.0, float(playback_mean_change_threshold))
        self.playback_max_change_threshold = max(0.0, float(playback_max_change_threshold))
        self.static_mean_change_threshold = max(0.0, float(static_mean_change_threshold))
        self.static_max_change_threshold = max(0.0, float(static_max_change_threshold))

    @staticmethod
    def _read_gray(path: Any) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _change_ratio(gray_a: Any, gray_b: Any, threshold: int = 22) -> float:
        if gray_a.shape != gray_b.shape:
            gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
        diff = cv2.absdiff(gray_a, gray_b)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        total = max(1, int(binary.shape[0] * binary.shape[1]))
        return int(cv2.countNonZero(binary)) / total

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return {"mean": mean, "max": max(values), "min": min(values), "std": variance**0.5}

    @staticmethod
    def _bottom_roi(gray: Any) -> Any:
        height = int(gray.shape[0])
        y1 = int(height * 0.68)
        return gray[y1:height, :]

    @staticmethod
    def _content_roi(gray: Any) -> Any:
        height = int(gray.shape[0])
        y2 = max(1, int(height * 0.85))
        return gray[:y2, :]

    @staticmethod
    def _screen_stats(gray: Any) -> dict[str, float]:
        total = max(1, int(gray.shape[0] * gray.shape[1]))
        return {
            "mean_luma": float(gray.mean()),
            "std_luma": float(gray.std()),
            "black_ratio": int((gray <= 16).sum()) / total,
            "white_ratio": int((gray >= 239).sum()) / total,
        }

    def _transition_metrics(self, frames: list[Any]) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        for idx in range(len(frames) - 1):
            gray_a = self._read_gray(frames[idx].image_path)
            gray_b = self._read_gray(frames[idx + 1].image_path)
            transitions.append(
                {
                    "from_index": idx,
                    "to_index": idx + 1,
                    "from_timestamp_sec": float(getattr(frames[idx], "timestamp_sec", 0.0)),
                    "to_timestamp_sec": float(getattr(frames[idx + 1], "timestamp_sec", 0.0)),
                    "full_change_ratio": self._change_ratio(gray_a, gray_b),
                    "bottom_roi_change_ratio": self._change_ratio(
                        self._bottom_roi(gray_a),
                        self._bottom_roi(gray_b),
                    ),
                    "content_change_ratio": self._change_ratio(
                        self._content_roi(gray_a),
                        self._content_roi(gray_b),
                    ),
                }
            )
        return transitions

    def _frames_in_window(self, frames: list[Any], start_sec: float, end_sec: float) -> list[Any]:
        return [
            frame
            for frame in frames
            if start_sec <= float(getattr(frame, "timestamp_sec", 0.0)) <= end_sec
        ]

    def _post_seek_primary_frames(self, frames: list[Any], seek_ts: float) -> list[Any]:
        window_start = seek_ts + self.post_seek_settle_sec
        window_end = seek_ts + self.post_seek_window_sec
        return self._frames_in_window(frames, window_start, window_end)

    def _post_seek_judge_frames(
        self, frames: list[Any], seek_ts: float
    ) -> tuple[list[Any], str, float, float]:
        """选取用于判定的 post-seek 帧，必要时回退以保证能给出结论。"""
        window_start = seek_ts + self.post_seek_settle_sec
        window_end = seek_ts + self.post_seek_window_sec
        primary = self._frames_in_window(frames, window_start, window_end)
        if len(primary) >= 2:
            return primary, "primary", window_start, window_end

        # 片尾/短尾：放宽 settle，使用 seek 后全部剩余帧。
        after_seek = [
            frame
            for frame in frames
            if float(getattr(frame, "timestamp_sec", 0.0)) >= seek_ts
        ]
        if len(after_seek) >= 2:
            start = float(getattr(after_seek[0], "timestamp_sec", seek_ts))
            end = float(getattr(after_seek[-1], "timestamp_sec", seek_ts))
            return after_seek, "after_seek_fallback", start, end

        # 仅 1 帧时，补上 seek 前一帧作为对比锚点，仍给出播放结论。
        before = [
            frame
            for frame in frames
            if float(getattr(frame, "timestamp_sec", 0.0)) < seek_ts
        ]
        if before and after_seek:
            anchored = [before[-1], *after_seek]
            start = float(getattr(anchored[0], "timestamp_sec", seek_ts))
            end = float(getattr(anchored[-1], "timestamp_sec", seek_ts))
            return anchored, "pre_seek_anchor_fallback", start, end

        sparse = after_seek or frames[-1:]
        start = float(getattr(sparse[0], "timestamp_sec", seek_ts)) if sparse else seek_ts
        end = float(getattr(sparse[-1], "timestamp_sec", seek_ts)) if sparse else seek_ts
        return sparse, "sparse_fallback", start, end

    @staticmethod
    def _seek_score(item: dict[str, Any]) -> float:
        return float(item["bottom_roi_change_ratio"]) * 1.5 + float(item["full_change_ratio"])

    @staticmethod
    def _frozen_tail_count(ratios: list[float], threshold: float) -> int:
        """从窗口末尾起连续近静止转场数。"""
        frozen = 0
        for ratio in reversed(ratios):
            if float(ratio) <= threshold:
                frozen += 1
            else:
                break
        return frozen

    def _locate_seek(
        self,
        frames: list[Any],
        seek_timestamp_sec: float | None = None,
    ) -> tuple[bool, float | None, str, dict[str, Any]]:
        transitions = self._transition_metrics(frames)
        metrics: dict[str, Any] = {
            "provided_seek_timestamp_sec": seek_timestamp_sec,
            "transitions": transitions,
            "seek_roi_change_threshold": self.seek_roi_change_threshold,
            "seek_full_change_threshold": self.seek_full_change_threshold,
            "min_post_seek_frames": 2,
        }
        if seek_timestamp_sec is not None:
            metrics["seek_candidate"] = {
                "timestamp_sec": seek_timestamp_sec,
                "source": "input",
                "post_seek_frame_count": len(self._post_seek_primary_frames(frames, seek_timestamp_sec)),
            }
            return True, seek_timestamp_sec, "input", metrics

        if not transitions:
            return False, None, "none", metrics

        # 避免把开头播放器初始化误当成 seek；若帧数很少则保留全部转场。
        ignore_first = 1 if len(transitions) >= 3 else 0
        candidates = transitions[ignore_first:]
        if not candidates:
            candidates = transitions

        enriched: list[dict[str, Any]] = []
        for item in candidates:
            seek_ts = float(item["to_timestamp_sec"])
            enriched.append(
                {
                    **item,
                    "score": self._seek_score(item),
                    "post_seek_frame_count": len(self._post_seek_primary_frames(frames, seek_ts)),
                }
            )
        metrics["ranked_candidates"] = sorted(
            enriched, key=lambda item: float(item["score"]), reverse=True
        )

        # 方案 A：只在“seek 后主窗口仍可判定”的候选里选最强转场，避免片尾假峰值。
        judgable = [
            item
            for item in enriched
            if int(item["post_seek_frame_count"]) >= 2
            and (
                float(item["bottom_roi_change_ratio"]) >= self.seek_roi_change_threshold
                or float(item["full_change_ratio"]) >= self.seek_full_change_threshold
            )
        ]
        metrics["judgable_candidate_count"] = len(judgable)
        if not judgable:
            metrics["seek_candidate"] = None
            return False, None, "none", metrics

        best = max(judgable, key=lambda item: float(item["score"]))
        metrics["seek_candidate"] = best
        return True, float(best["to_timestamp_sec"]), "auto_cv", metrics

    def _detect_black_white_tail(self, frames: list[Any]) -> tuple[str | None, dict[str, Any]]:
        tail = frames[-min(len(frames), 3) :]
        stats: list[dict[str, Any]] = []
        for frame in tail:
            gray = self._read_gray(frame.image_path)
            item = {
                "timestamp_sec": float(getattr(frame, "timestamp_sec", 0.0)),
                **self._screen_stats(gray),
            }
            item["black_match"] = item["black_ratio"] >= 0.92 and item["std_luma"] <= 12.0
            item["white_match"] = item["white_ratio"] >= 0.92 and item["std_luma"] <= 12.0
            stats.append(item)

        black_count = sum(1 for item in stats if bool(item["black_match"]))
        white_count = sum(1 for item in stats if bool(item["white_match"]))
        metrics = {
            "tail_frame_count": len(tail),
            "black_match_count": black_count,
            "white_match_count": white_count,
            "tail_frames": stats,
        }
        if tail and black_count == len(tail):
            return "black_screen", metrics
        if tail and white_count == len(tail):
            return "white_screen", metrics
        return None, metrics

    def _judge_post_seek(self, frames: list[Any], seek_ts: float) -> tuple[bool, str, str, dict[str, Any]]:
        window_frames, window_mode, window_start, window_end = self._post_seek_judge_frames(
            frames, seek_ts
        )
        metrics: dict[str, Any] = {
            "window_mode": window_mode,
            "window_start_sec": window_start,
            "window_end_sec": window_end,
            "window_frame_count": len(window_frames),
        }
        if not window_frames:
            return False, "unknown", "seek 后无可用帧，判定为播放未恢复。", metrics

        screen_anomaly, screen_metrics = self._detect_black_white_tail(window_frames)
        metrics["screen_stats"] = screen_metrics
        metrics["window_first_timestamp_sec"] = float(
            getattr(window_frames[0], "timestamp_sec", 0.0)
        )
        metrics["window_last_timestamp_sec"] = float(
            getattr(window_frames[-1], "timestamp_sec", 0.0)
        )

        # 单帧也给结论：优先黑白屏，否则视为未恢复播放。
        if len(window_frames) < 2:
            if screen_anomaly in {"black_screen", "white_screen"}:
                text = "黑屏" if screen_anomaly == "black_screen" else "白屏"
                return False, screen_anomaly, f"seek 后仅有单帧且为持续{text}，播放未正常恢复。", metrics
            return False, "unknown", "seek 后仅有单帧且未见播放恢复证据，判定为播放异常。", metrics

        transitions = self._transition_metrics(window_frames)
        content_ratios = [float(item["content_change_ratio"]) for item in transitions]
        full_ratios = [float(item["full_change_ratio"]) for item in transitions]
        content_stats = self._stats(content_ratios)
        full_stats = self._stats(full_ratios)
        duration_sec = max(
            0.0,
            float(getattr(window_frames[-1], "timestamp_sec", 0.0))
            - float(getattr(window_frames[0], "timestamp_sec", 0.0)),
        )
        frozen_tail = self._frozen_tail_count(
            content_ratios, self.static_max_change_threshold
        )
        metrics.update(
            {
                "window_duration_sec": duration_sec,
                "content_change_ratios": content_ratios,
                "full_change_ratios": full_ratios,
                "content_change_stats": content_stats,
                "full_change_stats": full_stats,
                "frozen_tail_count": frozen_tail,
                "post_seek_transitions": transitions,
            }
        )

        if screen_anomaly in {"black_screen", "white_screen"}:
            text = "黑屏" if screen_anomaly == "black_screen" else "白屏"
            return False, screen_anomaly, f"seek 后检测窗口末尾持续{text}，播放未正常恢复。", metrics

        # seek 后先有变化、随后连续卡住：不能仅凭前半段 mean/max 判 pass。
        if frozen_tail >= 2:
            return (
                False,
                "no_response",
                "seek 后画面先变化随后持续静止，播放未持续恢复。",
                metrics,
            )

        if (
            float(content_stats["mean"]) >= self.playback_mean_change_threshold
            or float(content_stats["max"]) >= self.playback_max_change_threshold
        ):
            return (
                True,
                "none",
                "seek 后画面内容恢复变化，未见长时间 loading、黑白屏或卡死。",
                metrics,
            )

        if (
            duration_sec >= self.long_loading_threshold_sec
            and float(full_stats["mean"]) > self.static_mean_change_threshold
            and float(full_stats["mean"]) <= 0.03
            and float(full_stats["std"]) <= 0.008
        ):
            return False, "long_loading", "seek 后长时间只有小幅规律变化，疑似持续 loading/buffering。", metrics

        if (
            duration_sec >= self.long_loading_threshold_sec
            and float(content_stats["mean"]) <= self.static_mean_change_threshold
            and float(content_stats["max"]) <= self.static_max_change_threshold
        ):
            return False, "no_response", "seek 后画面长期近乎静止，播放进度未体现恢复。", metrics

        return False, "unknown", "seek 后未观察到明确的播放恢复证据，判定为播放异常。", metrics

    def detect(
        self,
        sampled_frames: list[Any],
        task_id: str,
        seek_timestamp_sec: float | None = None,
    ) -> SeekPlaybackDetectionResult:
        """检测视频 seek 后是否正常播放。"""
        if len(sampled_frames) < 2:
            raise ValueError("seek_playback 检测至少需要 2 帧")
        t_start = time.perf_counter()

        seek_detected, seek_ts, seek_source, seek_metrics = self._locate_seek(
            sampled_frames,
            seek_timestamp_sec=seek_timestamp_sec,
        )
        task_intent = "检测视频进度条 seek 后是否能恢复正常播放。"
        if not seek_detected or seek_ts is None:
            timing = {"elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2)}
            return SeekPlaybackDetectionResult(
                result=False,
                bug_detected=True,
                task_intent=task_intent,
                reason="未发现明确 seek 行为，无法确认 seek 后播放是否正常。",
                decision_basis="未检测到进度条拖拽/点击后的候选时间点；该检测项需要围绕 seek 后窗口判断。",
                anomaly_type="seek_not_detected",
                raw_response="",
                decision_source="cv_seek_locator",
                seek_detected=False,
                seek_timestamp_sec=None,
                seek_source=seek_source,
                cv_metrics={"seek": seek_metrics},
                timing=timing,
            )

        playback_ok, anomaly_type, basis, post_metrics = self._judge_post_seek(sampled_frames, seek_ts)
        timing = {"elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2)}
        return SeekPlaybackDetectionResult(
            result=playback_ok,
            bug_detected=not playback_ok,
            task_intent=task_intent,
            reason=basis,
            decision_basis=(
                f"seek 时间点={seek_ts:.2f}s（来源={seek_source}）。{basis}"
            ),
            anomaly_type=anomaly_type,
            raw_response="",
            decision_source="cv_post_seek_window",
            seek_detected=True,
            seek_timestamp_sec=seek_ts,
            seek_source=seek_source,
            cv_metrics={
                "seek": seek_metrics,
                "post_seek": post_metrics,
                "thresholds": {
                    "post_seek_window_sec": self.post_seek_window_sec,
                    "post_seek_settle_sec": self.post_seek_settle_sec,
                    "long_loading_threshold_sec": self.long_loading_threshold_sec,
                    "playback_mean_change_threshold": self.playback_mean_change_threshold,
                    "playback_max_change_threshold": self.playback_max_change_threshold,
                },
            },
            timing=timing,
        )
