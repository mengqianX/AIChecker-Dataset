"""加载异常检测模块：CV-first + VLM-fallback。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from aichecker.vision.evaluator import VisionEvaluator
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
        cv_min_long_loading_duration_sec: float = 5.0,
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
        self.cv_min_long_loading_duration_sec = max(0.0, float(cv_min_long_loading_duration_sec))

    @staticmethod
    def _read_gray_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    def _detect_black_white_screen(self, sampled_frames: list[Any]) -> tuple[str | None, str, dict[str, Any]]:
        """
        识别黑/白屏异常。只在末尾窗口持续命中时判定，避免把单帧转场误报为异常。
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
        return None, "", metrics

    def _cv_decide(
        self,
        ratios: list[float],
        duration_sec: float | None = None,
    ) -> tuple[bool | None, str, str, dict[str, Any]]:
        stats = self._build_stats(ratios)
        metrics: dict[str, Any] = {
            "frame_change_ratios": ratios,
            **stats,
            "duration_sec": duration_sec,
            "min_long_loading_duration_sec": self.cv_min_long_loading_duration_sec,
        }
        if not ratios:
            return None, "CV证据不足：缺少相邻帧变化数据。", "unknown", metrics

        mean_ratio = float(stats["mean"])
        max_ratio = float(stats["max"])
        std_ratio = float(stats["std"])

        if max_ratio >= self.cv_clear_progress_max_threshold:
            return (
                False,
                "CV判定存在明显页面变化，未见持续加载卡死特征。",
                "none",
                metrics,
            )

        if mean_ratio <= self.cv_static_mean_threshold and max_ratio <= self.cv_static_max_threshold:
            return (
                True,
                "CV判定页面长时间近乎静止，疑似加载无反馈/卡住。",
                "no_response",
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
            "no_response",
            "long_loading",
            "unknown",
            "none",
        ]
        for p in priority:
            if p in anomaly_types:
                return p
        return anomaly_types[0]

    def _run_loading_fallback(self, sampled_frames: list[Any], task_id: str) -> dict[str, Any]:
        """
        loading 兜底语义判断（首尾帧）。
        """
        before = sampled_frames[0]
        after = sampled_frames[-1]
        prompt_pack = build_prompt_for_type(
            task_type="loading",
            context={
                "task_type": "loading",
                "before_timestamp_sec": before.timestamp_sec,
                "after_timestamp_sec": after.timestamp_sec,
            },
        )
        t_vlm_start = time.perf_counter()
        eval_result = self.evaluator.evaluate_json(
            before_image=before.image_path,
            after_image=after.image_path,
            task_id=f"{task_id}_loading_fallback",
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            required_fields={"bug_detected": bool, "reason": str, "decision_basis": str, "anomaly_type": str},
        )
        elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        parsed = eval_result.parsed_json
        anomaly_type = str(parsed.get("anomaly_type", "unknown")).strip().lower() or "unknown"
        return {
            "bug_detected": bool(parsed["bug_detected"]),
            "reason": str(parsed["reason"]),
            "decision_basis": str(parsed.get("decision_basis", parsed["reason"])),
            "anomaly_type": anomaly_type,
            "raw_response": eval_result.raw_response,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def detect(self, sampled_frames: list[Any], task_id: str) -> LoadingDetectionResult:
        """对整段 sampled_frames 做 loading 检测。"""
        if len(sampled_frames) < 2:
            raise ValueError("loading 检测至少需要 2 帧")
        t_start = time.perf_counter()

        ratios: list[float] = []
        for idx in range(len(sampled_frames) - 1):
            gray_a = self._read_gray_image(sampled_frames[idx].image_path)
            gray_b = self._read_gray_image(sampled_frames[idx + 1].image_path)
            ratios.append(self._calc_change_ratio(gray_a, gray_b))

        stage_metrics = self._build_stage_metrics(sampled_frames, ratios)
        screen_anomaly_type, screen_reason, screen_metrics = self._detect_black_white_screen(sampled_frames)
        cv_result, cv_reason, cv_anomaly_type, cv_metrics = self._cv_decide(
            ratios,
            duration_sec=float(stage_metrics["duration_sec"]),
        )
        cv_metrics["screen_stats"] = screen_metrics
        cv_metrics["stages"] = stage_metrics
        cv_metrics["signals"] = {
            "no_response": {
                "detected": cv_result is True and cv_anomaly_type == "no_response",
                "status": "detected" if cv_result is True and cv_anomaly_type == "no_response" else "not_detected",
                "source": "cv_frame_change",
                "reason": cv_reason if cv_anomaly_type == "no_response" else "",
                "evidence": {
                    "mean_change_ratio": float(cv_metrics["mean"]),
                    "max_change_ratio": float(cv_metrics["max"]),
                    "static_mean_threshold": self.cv_static_mean_threshold,
                    "static_max_threshold": self.cv_static_max_threshold,
                    "duration_sec": float(stage_metrics["duration_sec"]),
                    "stage_change_stats": {
                        "early": stage_metrics["early_change_stats"],
                        "middle": stage_metrics["middle_change_stats"],
                        "tail": stage_metrics["tail_change_stats"],
                    },
                },
            },
            "black_white_screen": {
                "detected": screen_anomaly_type in {"black_screen", "white_screen"},
                "status": "detected" if screen_anomaly_type in {"black_screen", "white_screen"} else "not_detected",
                "source": "cv_tail_window",
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
                "detected": cv_result is True and cv_anomaly_type == "long_loading",
                "status": (
                    "detected"
                    if cv_result is True and cv_anomaly_type == "long_loading"
                    else ("candidate_duration_too_short" if bool(cv_metrics.get("long_loading_candidate")) else "not_detected")
                ),
                "source": "cv_frame_change",
                "reason": cv_reason if cv_anomaly_type == "long_loading" or bool(cv_metrics.get("long_loading_candidate")) else "",
                "evidence": {
                    "mean_change_ratio": float(cv_metrics["mean"]),
                    "std_change_ratio": float(cv_metrics["std"]),
                    "spinner_mean_upper": self.cv_spinner_mean_upper,
                    "spinner_std_upper": self.cv_spinner_std_upper,
                    "duration_sec": float(stage_metrics["duration_sec"]),
                    "min_duration_sec": self.cv_min_long_loading_duration_sec,
                    "stage_change_stats": {
                        "early": stage_metrics["early_change_stats"],
                        "middle": stage_metrics["middle_change_stats"],
                        "tail": stage_metrics["tail_change_stats"],
                    },
                },
            },
        }
        cv_state = "uncertain" if cv_result is None else ("positive" if cv_result else "negative")

        detected_anomaly_types: list[str] = []
        reason_parts: list[str] = []
        source_flags: list[str] = []
        raw_response_parts: list[str] = []

        if screen_anomaly_type is not None:
            detected_anomaly_types.append(screen_anomaly_type)
            reason_parts.append(screen_reason)
            source_flags.append("cv_black_white")

        fallback_elapsed_ms = 0.0

        if cv_state == "positive":
            if cv_anomaly_type not in {"none", "unknown"}:
                detected_anomaly_types.append(cv_anomaly_type)
            reason_parts.append(cv_reason)
            source_flags.append("cv_loading_signal")
            cv_metrics["failure_text_probe"] = {
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
            cv_metrics["signals"]["load_failed"] = {
                "detected": False,
                "status": "delegated_to_load_failure_prompt_detector",
                "source": "load_failure_prompt_detector",
                "reason": "页面加载失败提示由独立检测器判断。",
                "evidence": {"scan_count": 0, "hit_count": 0},
            }
            reason_parts.append("页面加载失败提示由独立检测器判断。")
        elif cv_state == "negative":
            reason_parts.append(cv_reason)
            source_flags.append("cv_clear_progress")
            cv_metrics["failure_text_probe"] = {
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
            cv_metrics["signals"]["load_failed"] = {
                "detected": False,
                "status": "delegated_to_load_failure_prompt_detector",
                "source": "load_failure_prompt_detector",
                "reason": "页面加载失败提示由独立检测器判断。",
                "evidence": {"scan_count": 0, "hit_count": 0},
            }
            reason_parts.append("页面加载失败提示由独立检测器判断。")
        else:
            # cv_state == "uncertain"
            cv_metrics["failure_text_probe"] = {
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
            cv_metrics["signals"]["load_failed"] = {
                "detected": False,
                "status": "delegated_to_load_failure_prompt_detector",
                "source": "load_failure_prompt_detector",
                "reason": "页面加载失败提示由独立检测器判断。",
                "evidence": {"scan_count": 0, "hit_count": 0},
            }
            reason_parts.append(cv_reason)
            source_flags.append("cv_uncertain")
            fallback_result = self._run_loading_fallback(sampled_frames=sampled_frames, task_id=task_id)
            fallback_elapsed_ms = float(fallback_result["elapsed_ms"])
            reason_parts.append(f"VLM兜底：{fallback_result['reason']}")
            reason_parts.append(f"VLM依据：{fallback_result['decision_basis']}")
            source_flags.append("vlm_fallback")
            raw_response_parts.append(f"[loading_fallback]\n{fallback_result['raw_response']}")
            fallback_anomaly_type = str(fallback_result["anomaly_type"])
            if bool(fallback_result["bug_detected"]) and fallback_anomaly_type in {
                "black_screen",
                "white_screen",
                "no_response",
                "long_loading",
                "unknown",
            }:
                detected_anomaly_types.append(fallback_anomaly_type)

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
