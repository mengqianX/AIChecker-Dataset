"""视频基础健康检测编排：串联解耦后的单项能力，输出独立判定。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from aichecker.vision.checkers.load_failure import LoadFailurePromptDetector
from aichecker.vision.checkers.loading import LoadingDetector
from aichecker.vision.perception import ExtractedFrame


LOADING_SIGNAL_CHECKS: tuple[tuple[str, str], ...] = (
    ("no_response", "no_response"),
    ("black_white_screen", "black_white_screen"),
    ("long_loading", "long_loading"),
)

BASELINE_TASK_TYPES = frozenset(
    {
        "video_baseline",
        "page_baseline",
        "baseline_health",
        "video_health",
    }
)


@dataclass(frozen=True)
class BaselineCheckResult:
    """单项基础检测结果。"""

    check_id: str
    bug_detected: bool
    reason: str
    decision_basis: str
    anomaly_type: str
    decision_source: str
    cv_metrics: dict[str, Any]
    timing: dict[str, float] | None = None


class VideoBaselineHealthOrchestrator:
    """对同一批抽帧依次调用解耦 detector，汇总 4 项基础检测。"""

    name = "video_baseline_health_detector"

    def __init__(
        self,
        loading_detector: LoadingDetector,
        load_failure_detector: LoadFailurePromptDetector,
    ) -> None:
        self.loading_detector = loading_detector
        self.load_failure_detector = load_failure_detector

    @staticmethod
    def _segment_from_loading_signal(
        *,
        task_id: str,
        check_id: str,
        signal_key: str,
        signal: dict[str, Any],
        loading_timing: dict[str, float] | None,
    ) -> dict[str, Any]:
        detected = bool(signal.get("detected", False))
        anomaly_type = str(signal.get("anomaly_type") or ("none" if not detected else check_id))
        reason = str(signal.get("reason") or "")
        if not reason and detected:
            reason = f"{check_id} 信号命中。"
        if not reason and not detected:
            reason = f"未检测到 {check_id} 异常。"
        return {
            "segment_index": len(LOADING_SIGNAL_CHECKS),
            "segment_task_id": f"{task_id}_{check_id}",
            "check_id": check_id,
            "signal_key": signal_key,
            "result": not detected,
            "bug_detected": detected,
            "reason": reason,
            "decision_basis": reason,
            "anomaly_type": anomaly_type,
            "decision_source": str(signal.get("source") or "loading_detector"),
            "selected_prompt_type": check_id,
            "cv_metrics": {"signal": signal},
            "timing": loading_timing,
        }

    @staticmethod
    def _segment_from_load_failure(task_id: str, result: Any) -> dict[str, Any]:
        return {
            "segment_index": len(LOADING_SIGNAL_CHECKS),
            "segment_task_id": f"{task_id}_load_failure",
            "check_id": "load_failure",
            "signal_key": "load_failure",
            "result": not bool(result.bug_detected),
            "bug_detected": bool(result.bug_detected),
            "reason": str(result.reason),
            "decision_basis": str(result.decision_basis),
            "anomaly_type": str(result.anomaly_type),
            "failure_type": str(result.failure_type),
            "evidence_text": str(result.evidence_text),
            "confidence": result.confidence,
            "decision_source": str(result.decision_source),
            "selected_prompt_type": "load_failure",
            "raw_model_response": str(result.raw_response),
            "cv_metrics": dict(result.cv_metrics),
            "timing": result.timing,
        }

    def run(
        self,
        *,
        task_id: str,
        sampled_frames: list[ExtractedFrame],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], bool, str]:
        if len(sampled_frames) < 2:
            raise ValueError("video_baseline 检测至少需要 2 帧")

        t_start = time.perf_counter()
        loading_result = self.loading_detector.detect(
            sampled_frames=sampled_frames,
            task_id=task_id,
        )
        load_failure_result = self.load_failure_detector.detect(
            sampled_frames=sampled_frames,
            task_id=task_id,
        )

        signals = dict(loading_result.cv_metrics.get("signals") or {})
        segments: list[dict[str, Any]] = []
        for index, (check_id, signal_key) in enumerate(LOADING_SIGNAL_CHECKS):
            signal = dict(signals.get(signal_key) or {})
            segment = self._segment_from_loading_signal(
                task_id=task_id,
                check_id=check_id,
                signal_key=signal_key,
                signal=signal,
                loading_timing=loading_result.timing,
            )
            segment["segment_index"] = index
            segments.append(segment)

        segments.append(self._segment_from_load_failure(task_id, load_failure_result))
        segments[-1]["segment_index"] = len(segments) - 1

        checks_summary = {
            check["check_id"]: {
                "bug_detected": check["bug_detected"],
                "anomaly_type": check.get("anomaly_type"),
                "reason": check.get("reason"),
            }
            for check in segments
        }
        summary_metrics = {
            "checks": checks_summary,
            "loading_cv_metrics": loading_result.cv_metrics,
            "load_failure_cv_metrics": load_failure_result.cv_metrics,
            "timing": {
                "total_elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                "loading_elapsed_ms": (loading_result.timing or {}).get("detect_elapsed_ms"),
                "load_failure_elapsed_ms": (load_failure_result.timing or {}).get("detect_elapsed_ms"),
            },
        }

        video_bug_detected = any(bool(item["bug_detected"]) for item in segments)
        task_intent = "对输入视频同步执行无响应、黑白屏、长时间加载、页面加载失败四项基础检测。"
        return segments, summary_metrics, video_bug_detected, task_intent
