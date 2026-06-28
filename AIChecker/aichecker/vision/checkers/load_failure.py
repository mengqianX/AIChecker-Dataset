"""页面加载失败提示检测：CV candidate prefilter + VLM semantic probe。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from aichecker.vision.evaluator import VisionEvaluator
from aichecker.vision.prompt_builders import build_loading_failure_probe_prompt


@dataclass
class LoadFailureDetectionResult:
    """页面加载失败提示检测结果。"""

    bug_detected: bool
    task_intent: str
    reason: str
    decision_basis: str
    anomaly_type: str
    failure_type: str
    evidence_text: str
    confidence: float | None
    raw_response: str
    decision_source: str
    cv_metrics: dict[str, Any]
    timing: dict[str, float] | None = None


class LoadFailurePromptDetector:
    """检测控件响应后是否出现加载失败相关提示或弹窗。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        probe_top_k: int = 3,
        probe_min_cv_score: float = 0.02,
        probe_force_keep: int = 1,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.probe_top_k = max(1, int(probe_top_k))
        self.probe_min_cv_score = max(0.0, float(probe_min_cv_score))
        self.probe_force_keep = max(1, int(probe_force_keep))

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

    def _build_candidate_indices(self, sampled_frames: list[Any]) -> list[int]:
        total = len(sampled_frames)
        if total <= self.probe_top_k:
            return list(range(total))

        tail_count = min(total, max(1, self.probe_top_k))
        anchors = set(range(total - tail_count, total))
        anchors.add(0)
        anchors.add(total // 2)
        return sorted(idx for idx in anchors if 0 <= idx < total)

    def _score_candidates(self, sampled_frames: list[Any], candidate_indices: list[int]) -> list[dict[str, Any]]:
        gray_cache: dict[int, Any] = {}

        def _gray(idx: int) -> Any:
            if idx not in gray_cache:
                gray_cache[idx] = self._read_gray_image(sampled_frames[idx].image_path)
            return gray_cache[idx]

        scores: list[dict[str, Any]] = []
        for idx in candidate_indices:
            before_idx = max(0, idx - 1)
            after_idx = min(len(sampled_frames) - 1, idx + 1)
            before_center = self._calc_change_ratio(_gray(before_idx), _gray(idx))
            center_after = self._calc_change_ratio(_gray(idx), _gray(after_idx))
            initial_center = self._calc_change_ratio(_gray(0), _gray(idx)) if idx != 0 else 0.0
            cv_score = max(before_center, center_after) + 0.5 * min(before_center, center_after)
            cv_score = max(cv_score, initial_center * 0.5)
            scores.append(
                {
                    "index": idx,
                    "before_center_ratio": round(before_center, 4),
                    "center_after_ratio": round(center_after, 4),
                    "initial_center_ratio": round(initial_center, 4),
                    "cv_score": round(cv_score, 4),
                }
            )
        return sorted(scores, key=lambda item: (float(item["cv_score"]), int(item["index"])), reverse=True)

    def _select_probe_indices(self, prefilter_scores: list[dict[str, Any]]) -> list[int]:
        filtered = [item for item in prefilter_scores if float(item["cv_score"]) >= self.probe_min_cv_score]
        if len(filtered) < self.probe_force_keep:
            filtered = prefilter_scores[: self.probe_force_keep]
        return [int(item["index"]) for item in filtered[: self.probe_top_k]]

    def _run_probe(
        self,
        before_image: Path,
        center_image: Path,
        after_image: Path,
        before_ts: float,
        center_ts: float,
        after_ts: float,
        task_id: str,
        candidate_index: int,
    ) -> dict[str, Any]:
        prompt_pack = build_loading_failure_probe_prompt(
            context={
                "before_timestamp_sec": before_ts,
                "after_timestamp_sec": center_ts,
            }
        )
        probe_result = self.evaluator.evaluate_json(
            before_image=before_image,
            after_image=center_image,
            task_id=f"{task_id}_load_failure_probe_{candidate_index:04d}",
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            required_fields={
                "load_failed": bool,
                "failure_type": str,
                "evidence_text": str,
                "reason": str,
            },
            extra_image_paths=[after_image],
            image_role_labels=[
                "图1:候选前帧（响应过程前后文）",
                "图2:候选帧（重点检测失败提示/弹窗）",
                "图3:候选后帧（验证提示是否短暂出现或持续存在）",
            ],
        )
        parsed = probe_result.parsed_json
        confidence_raw = parsed.get("confidence")
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
        return {
            "candidate_index": candidate_index,
            "before_ts": before_ts,
            "center_ts": center_ts,
            "after_ts": after_ts,
            "load_failed": bool(parsed["load_failed"]),
            "failure_type": str(parsed.get("failure_type", "unknown")).strip().lower() or "unknown",
            "evidence_text": str(parsed.get("evidence_text", "")),
            "reason": str(parsed.get("reason", "")),
            "confidence": confidence,
            "raw_response": probe_result.raw_response,
        }

    def detect(self, sampled_frames: list[Any], task_id: str) -> LoadFailureDetectionResult:
        """扫描少量候选帧，判断是否出现加载失败提示或弹窗。"""
        if len(sampled_frames) < 2:
            raise ValueError("页面加载失败检测至少需要 2 帧")

        t_start = time.perf_counter()
        candidate_indices = self._build_candidate_indices(sampled_frames)
        prefilter_scores = self._score_candidates(sampled_frames, candidate_indices)
        probe_indices = self._select_probe_indices(prefilter_scores)

        hits: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []
        raw_parts: list[str] = []
        t_vlm_start = time.perf_counter()
        for idx in probe_indices:
            before_idx = max(0, idx - 1)
            after_idx = min(len(sampled_frames) - 1, idx + 1)
            before_frame = sampled_frames[before_idx]
            center_frame = sampled_frames[idx]
            after_frame = sampled_frames[after_idx]
            result = self._run_probe(
                before_image=before_frame.image_path,
                center_image=center_frame.image_path,
                after_image=after_frame.image_path,
                before_ts=before_frame.timestamp_sec,
                center_ts=center_frame.timestamp_sec,
                after_ts=after_frame.timestamp_sec,
                task_id=task_id,
                candidate_index=idx,
            )
            all_results.append(result)
            raw_parts.append(f"[load_failure_probe_idx_{idx:04d}]\n{result['raw_response']}")
            if result["load_failed"]:
                hits.append(result)

        vlm_elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        best_hit = None
        if hits:
            best_hit = sorted(
                hits,
                key=lambda item: (
                    float(item["confidence"]) if item["confidence"] is not None else 0.0,
                    float(item["center_ts"]),
                ),
                reverse=True,
            )[0]

        bug_detected = best_hit is not None
        failure_type = best_hit["failure_type"] if best_hit else "none"
        evidence_text = best_hit["evidence_text"] if best_hit else ""
        confidence = best_hit["confidence"] if best_hit else None
        reason = (
            f"页面加载失败提示命中：{best_hit['reason']}（证据文案：{evidence_text}，候选帧={best_hit['candidate_index']}）"
            if best_hit
            else "未在候选帧中发现明确的加载失败提示或失败弹窗。"
        )
        cv_metrics = {
            "candidate_indices": candidate_indices,
            "probe_indices": probe_indices,
            "prefilter_scores": prefilter_scores,
            "scan_count": len(all_results),
            "hit_count": len(hits),
            "hits": hits,
            "best_hit": best_hit,
        }

        return LoadFailureDetectionResult(
            bug_detected=bug_detected,
            task_intent="检测控件响应后是否出现页面加载失败相关提示或弹窗信息。",
            reason=reason,
            decision_basis=reason,
            anomaly_type="load_failed" if bug_detected else "none",
            failure_type=failure_type,
            evidence_text=evidence_text,
            confidence=confidence,
            raw_response="\n\n".join(raw_parts),
            decision_source="vlm_failure_prompt",
            cv_metrics=cv_metrics,
            timing={
                "detect_elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                "vlm_elapsed_ms": round(vlm_elapsed_ms, 2),
            },
        )
