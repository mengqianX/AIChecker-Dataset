"""视频播放是否正常检测（点击播放/暂停后内容是否变化）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from aichecker.vision.checkers.seek_playback import SeekPlaybackDetector


@dataclass
class VideoPlayDetectionResult:
    """视频播放检测结果。"""

    result: bool
    bug_detected: bool
    task_intent: str
    reason: str
    decision_basis: str
    anomaly_type: str
    raw_response: str
    decision_source: str
    play_action_detected: bool
    play_timestamp_sec: float | None
    play_source: str
    cv_metrics: dict[str, Any]
    timing: dict[str, float] | None = None


class VideoPlayDetector(SeekPlaybackDetector):
    """定位播放/暂停操作点，并判断操作后视频内容是否正常播放。"""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        post_play_window_sec: float = 8.0,
        post_play_settle_sec: float = 0.5,
        play_roi_change_threshold: float = 0.018,
        play_full_change_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(logger=logger, **kwargs)
        self.post_play_window_sec = max(1.0, float(post_play_window_sec))
        self.post_play_settle_sec = max(0.0, float(post_play_settle_sec))
        self.play_roi_change_threshold = max(0.0, float(play_roi_change_threshold))
        self.play_full_change_threshold = max(0.0, float(play_full_change_threshold))

    @staticmethod
    def _candidate_score(item: dict[str, Any]) -> float:
        return float(item["bottom_roi_change_ratio"]) * 1.5 + float(item["full_change_ratio"])

    def _locate_play_action(
        self,
        frames: list[Any],
        play_timestamp_sec: float | None = None,
    ) -> tuple[bool, float | None, str, dict[str, Any]]:
        transitions = self._transition_metrics(frames)
        metrics: dict[str, Any] = {
            "provided_play_timestamp_sec": play_timestamp_sec,
            "transitions": transitions,
            "play_roi_change_threshold": self.play_roi_change_threshold,
            "play_full_change_threshold": self.play_full_change_threshold,
        }
        if play_timestamp_sec is not None:
            metrics["play_candidate"] = {
                "timestamp_sec": play_timestamp_sec,
                "source": "input",
            }
            return True, play_timestamp_sec, "input", metrics

        if not transitions:
            return False, None, "none", metrics

        ignore_first = 1 if len(transitions) >= 3 else 0
        candidates = transitions[ignore_first:] or transitions
        hits = [
            item
            for item in candidates
            if float(item["bottom_roi_change_ratio"]) >= self.play_roi_change_threshold
            or float(item["full_change_ratio"]) >= self.play_full_change_threshold
        ]
        if not hits:
            metrics["play_candidate"] = None
            return False, None, "none", metrics

        best_score = max(self._candidate_score(item) for item in hits)
        score_floor = best_score * 0.85
        tied = [item for item in hits if self._candidate_score(item) >= score_floor]
        chosen = min(tied, key=lambda item: float(item["to_timestamp_sec"]))
        metrics["play_candidate"] = chosen
        metrics["play_candidate_alternatives"] = tied
        return True, float(chosen["to_timestamp_sec"]), "auto_cv", metrics

    def _content_motion_ok(self, frames: list[Any]) -> tuple[bool, dict[str, Any]]:
        transitions = self._transition_metrics(frames)
        content_ratios = [float(item["content_change_ratio"]) for item in transitions]
        content_stats = self._stats(content_ratios)
        metrics = {
            "content_change_ratios": content_ratios,
            "content_change_stats": content_stats,
        }
        passed = (
            float(content_stats["mean"]) >= self.playback_mean_change_threshold
            or float(content_stats["max"]) >= self.playback_max_change_threshold
        )
        return passed, metrics

    def _judge_post_play(
        self,
        frames: list[Any],
        play_ts: float,
    ) -> tuple[bool, str, str, dict[str, Any]]:
        last_ts = float(getattr(frames[-1], "timestamp_sec", 0.0))
        window_start = play_ts + self.post_play_settle_sec
        window_end = min(play_ts + self.post_play_window_sec, last_ts)
        window_frames = self._frames_in_window(frames, window_start, window_end)
        if len(window_frames) < 2:
            window_frames = [
                frame
                for frame in frames
                if float(getattr(frame, "timestamp_sec", 0.0)) >= play_ts
            ]

        if len(window_frames) < 2:
            motion_ok, motion_metrics = self._content_motion_ok(frames[1:])
            metrics = {
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "window_frame_count": len(window_frames),
                "fallback_motion": motion_metrics,
            }
            if motion_ok:
                return True, "none", "操作后可用帧不足，但整体画面存在持续内容变化，判定为播放正常。", metrics
            return False, "evidence_insufficient", "操作后可用帧不足，且未观察到足够的内容变化。", metrics

        playback_ok, anomaly_type, basis, post_metrics = self._judge_post_seek(
            frames,
            play_ts,
        )
        metrics = dict(post_metrics)
        metrics["window_end_clamped_sec"] = window_end

        if anomaly_type == "evidence_insufficient":
            tail_frames = [
                frame
                for frame in frames
                if float(getattr(frame, "timestamp_sec", 0.0)) >= play_ts
            ]
            motion_ok, motion_metrics = self._content_motion_ok(tail_frames)
            metrics["tail_motion"] = motion_metrics
            if motion_ok:
                return True, "none", "操作后检测窗口较短，但剩余片段存在明显内容变化，判定为播放正常。", metrics
            return False, "no_response", "操作后画面长期近乎静止，疑似未正常播放。", metrics

        if anomaly_type == "unknown":
            tail_frames = [
                frame
                for frame in frames
                if float(getattr(frame, "timestamp_sec", 0.0)) >= play_ts
            ]
            motion_ok, motion_metrics = self._content_motion_ok(tail_frames)
            metrics["tail_motion"] = motion_metrics
            if motion_ok:
                return True, "none", "操作后画面存在持续内容变化，判定为播放正常。", metrics

        return playback_ok, anomaly_type, basis.replace("seek", "播放操作"), metrics

    def detect(
        self,
        sampled_frames: list[Any],
        task_id: str,
        play_timestamp_sec: float | None = None,
    ) -> VideoPlayDetectionResult:
        """检测点击播放/暂停后视频内容是否正常播放。"""
        if len(sampled_frames) < 2:
            raise ValueError("video_play 检测至少需要 2 帧")
        t_start = time.perf_counter()
        task_intent = "检测点击播放/暂停按钮后视频内容是否正常播放。"

        play_detected, play_ts, play_source, play_metrics = self._locate_play_action(
            sampled_frames,
            play_timestamp_sec=play_timestamp_sec,
        )
        if not play_detected or play_ts is None:
            motion_ok, motion_metrics = self._content_motion_ok(sampled_frames[1:])
            timing = {"elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2)}
            if motion_ok:
                return VideoPlayDetectionResult(
                    result=True,
                    bug_detected=False,
                    task_intent=task_intent,
                    reason="未定位到明确的播放/暂停操作，但整体画面存在持续内容变化，判定为播放正常。",
                    decision_basis="全片内容区域存在足够帧间变化，未见卡死或黑白屏证据。",
                    anomaly_type="none",
                    raw_response="",
                    decision_source="cv_whole_video_motion",
                    play_action_detected=False,
                    play_timestamp_sec=None,
                    play_source=play_source,
                    cv_metrics={"play": play_metrics, "whole_video_motion": motion_metrics},
                    timing=timing,
                )
            return VideoPlayDetectionResult(
                result=False,
                bug_detected=True,
                task_intent=task_intent,
                reason="未发现明确播放/暂停操作，且全片内容变化不足，无法确认视频正常播放。",
                decision_basis="未检测到控件操作候选点，且内容区域帧间变化低于播放阈值。",
                anomaly_type="play_not_detected",
                raw_response="",
                decision_source="cv_play_locator",
                play_action_detected=False,
                play_timestamp_sec=None,
                play_source=play_source,
                cv_metrics={"play": play_metrics, "whole_video_motion": motion_metrics},
                timing=timing,
            )

        playback_ok, anomaly_type, basis, post_metrics = self._judge_post_play(sampled_frames, play_ts)
        timing = {"elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2)}
        return VideoPlayDetectionResult(
            result=playback_ok,
            bug_detected=not playback_ok,
            task_intent=task_intent,
            reason=basis,
            decision_basis=f"播放操作时间点={play_ts:.2f}s（来源={play_source}）。{basis}",
            anomaly_type=anomaly_type,
            raw_response="",
            decision_source="cv_post_play_window",
            play_action_detected=True,
            play_timestamp_sec=play_ts,
            play_source=play_source,
            cv_metrics={
                "play": play_metrics,
                "post_play": post_metrics,
                "thresholds": {
                    "post_play_window_sec": self.post_play_window_sec,
                    "post_play_settle_sec": self.post_play_settle_sec,
                    "play_roi_change_threshold": self.play_roi_change_threshold,
                    "play_full_change_threshold": self.play_full_change_threshold,
                    "playback_mean_change_threshold": self.playback_mean_change_threshold,
                    "playback_max_change_threshold": self.playback_max_change_threshold,
                },
            },
            timing=timing,
        )
