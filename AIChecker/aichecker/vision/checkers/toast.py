"""Toast 消息检测模块：定位关键帧并校验文案是否符合预期。"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aichecker.vision.evaluator import VisionEvaluator
from aichecker.vision.perception import ExtractedFrame
from aichecker.vision.preprocessor import GuiPreprocessor
from aichecker.vision.prompt_builders import build_prompt_for_type, render_toast_user_prompt


@dataclass
class ToastDetectionResult:
    """Toast 检测结果。"""

    bug_detected: bool
    expectation_met: bool
    task_intent: str
    key_frame_path: Path | None
    key_frame_timestamp: float | None
    toast_text: str
    action_semantic: str
    inferred_expected_toast_text: str
    reason: str
    confidence: float | None
    raw_response: str
    scanned_candidates: int
    total_candidates: int
    evaluated_candidate_indices: list[int]
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] | None = None
    preprocess_evidence: dict[str, Any] | None = None


class ToastMessageDetector:
    """基于 VLM 的 Toast 关键帧定位与语义判定。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        preprocessor: GuiPreprocessor | None = None,
        enable_preprocess: bool = True,
        top_k_candidates: int = 2,
        early_stop_confidence: float = 0.85,
        min_peak_score: float = 0.2,
        vlm_max_long_edge: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.preprocessor = preprocessor
        self.enable_preprocess = enable_preprocess
        self.top_k_candidates = min(2, max(1, int(top_k_candidates)))
        self.early_stop_confidence = min(1.0, max(0.0, float(early_stop_confidence)))
        self.min_peak_score = max(0.0, float(min_peak_score))
        # 0 = 送抽帧原图。压缩主要伤小字 toast，对网关耗时几乎无帮助。
        edge = int(vlm_max_long_edge)
        self.vlm_max_long_edge = 0 if edge <= 0 else max(256, edge)
        self.prompt_pack = build_prompt_for_type(task_type="toast", context={})

    @staticmethod
    def _normalize_toast_text(text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"[，。！？、,:;；!?\s]+", "", normalized)
        # 常见业务同义词归一，避免“房间/直播间”类文案差异造成误判。
        normalized = normalized.replace("直播间", "房间")
        # 弱语义后缀，通常不影响主干语义。
        for noise in ["请稍后重试", "稍后重试", "请重试", "请稍候再试", "请稍后再试"]:
            normalized = normalized.replace(noise, "")
        return normalized

    @staticmethod
    def _has_failure_signal(text: str) -> bool:
        return any(token in text for token in ["失败", "错误", "异常", "超时", "无法", "不能"])

    @staticmethod
    def _has_success_signal(text: str) -> bool:
        return any(token in text for token in ["成功", "已", "完成", "恢复", "还原", "通过"])

    @staticmethod
    def _char_ngram_set(text: str, n: int) -> set[str]:
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    @classmethod
    def _semantic_equivalent_toast(cls, actual_text: str, expected_text: str) -> bool:
        actual_norm = cls._normalize_toast_text(actual_text)
        expected_norm = cls._normalize_toast_text(expected_text)
        if not actual_norm or not expected_norm:
            return False
        if actual_norm == expected_norm:
            return True
        if actual_norm in expected_norm or expected_norm in actual_norm:
            return True

        actual_fail = cls._has_failure_signal(actual_norm)
        expected_fail = cls._has_failure_signal(expected_norm)
        actual_success = cls._has_success_signal(actual_norm)
        expected_success = cls._has_success_signal(expected_norm)
        # 明显成败冲突时，不能认定为语义一致。
        if (actual_fail and expected_success and not expected_fail) or (
            expected_fail and actual_success and not actual_fail
        ):
            return False

        actual_2gram, expected_2gram = cls._char_ngram_set(actual_norm, 2), cls._char_ngram_set(expected_norm, 2)
        actual_3gram, expected_3gram = cls._char_ngram_set(actual_norm, 3), cls._char_ngram_set(expected_norm, 3)
        jaccard_2 = (
            (len(actual_2gram & expected_2gram) / len(actual_2gram | expected_2gram))
            if (actual_2gram and expected_2gram)
            else 0.0
        )
        jaccard_3 = (
            (len(actual_3gram & expected_3gram) / len(actual_3gram | expected_3gram))
            if (actual_3gram and expected_3gram)
            else 0.0
        )
        return max(jaccard_2, jaccard_3) >= 0.33

    _TAG_ACTION_HINTS = ("选择标签", "标签", "tag", "添加标签", "确认标签")
    _RECYCLE_FEEDBACK_HINTS = ("回收", "回收站", "已回收", "移入回收站", "移出回收站", "归档")

    @classmethod
    def _contains_tag_action(cls, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(hint.lower() in normalized or hint in (text or "") for hint in cls._TAG_ACTION_HINTS)

    @classmethod
    def _contains_recycle_feedback(cls, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        return any(hint in normalized for hint in cls._RECYCLE_FEEDBACK_HINTS)

    @classmethod
    def _infer_expected_for_tag_action(cls, action_semantic: str) -> str:
        if cls._contains_tag_action(action_semantic):
            return "标签已添加"
        return ""

    @classmethod
    def _apply_cross_candidate_conflicts(cls, candidates: list[dict[str, Any]]) -> None:
        tag_action_seen = any(cls._contains_tag_action(str(c.get("action_semantic", ""))) for c in candidates)
        if not tag_action_seen:
            return
        for candidate in candidates:
            feedback_text = " ".join(
                [
                    str(candidate.get("toast_text", "")),
                    str(candidate.get("action_semantic", "")),
                    str(candidate.get("reason", "")),
                ]
            )
            if not candidate.get("toast_text") or not cls._contains_recycle_feedback(feedback_text):
                continue
            candidate["toast_visible"] = True
            candidate["expectation_met"] = False
            candidate["is_uncertain_action"] = False
            if not candidate.get("inferred_expected_toast_text"):
                candidate["inferred_expected_toast_text"] = cls._infer_expected_for_tag_action("选择标签")
            candidate["reason"] = (
                f"{candidate.get('reason', '')}"
                "（后处理修正：存在标签操作上下文，但反馈为回收/归档类文案，判定为语义冲突。）"
            ).strip()

    @staticmethod
    def _build_toast_image_role_labels() -> list[str]:
        return [
            "图1:动作前完整帧（必须用于推断操作语义）",
            "图2:候选完整帧（toast出现时）",
        ]

    @staticmethod
    def _select_final_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None

        def _rank_key(candidate: dict[str, Any]) -> tuple[float, int]:
            return (float(candidate.get("confidence") or 0.0), int(candidate.get("idx") or 0))

        visible_reliable_failures = [
            c
            for c in candidates
            if bool(c.get("toast_visible"))
            and (not bool(c.get("expectation_met")))
            and (not bool(c.get("is_uncertain_action")))
        ]
        if visible_reliable_failures:
            return max(visible_reliable_failures, key=_rank_key)

        visible_expectations_met = [c for c in candidates if bool(c.get("toast_visible")) and bool(c.get("expectation_met"))]
        if visible_expectations_met:
            return max(visible_expectations_met, key=_rank_key)

        visible_uncertain_failures = [
            c
            for c in candidates
            if bool(c.get("toast_visible"))
            and (not bool(c.get("expectation_met")))
            and bool(c.get("is_uncertain_action"))
        ]
        if visible_uncertain_failures:
            return max(visible_uncertain_failures, key=_rank_key)

        invisible_candidates = [c for c in candidates if not bool(c.get("toast_visible"))]
        if invisible_candidates:
            return max(
                invisible_candidates,
                key=lambda c: (int(c.get("idx") or 0), float(c.get("confidence") or 0.0)),
            )

        return max(candidates, key=_rank_key)

    @staticmethod
    def _select_score_peaks(
        candidate_scores: list[dict[str, Any]],
        *,
        max_peaks: int,
        min_separation: int = 3,
        min_score: float = 0.2,
    ) -> list[int]:
        ranked = sorted(
            candidate_scores,
            key=lambda item: (float(item.get("score") or 0.0), int(item.get("index") or 0)),
            reverse=True,
        )
        selected: list[int] = []
        for item in ranked:
            score_value = float(item.get("score") or 0.0)
            if score_value < min_score:
                continue
            idx = int(item.get("index", -1))
            if idx < 0:
                continue
            if any(abs(idx - kept) < min_separation for kept in selected):
                continue
            selected.append(idx)
            if len(selected) >= max_peaks:
                break
        if selected:
            return selected
        if not ranked:
            return []
        fallback_score = float(ranked[0].get("score") or 0.0)
        fallback_idx = int(ranked[0].get("index", -1))
        if fallback_score <= 0.0 or fallback_idx < 0:
            return []
        return [fallback_idx]

    @staticmethod
    def _build_detect_timing(
        *,
        detect_start: float,
        scoring_elapsed_ms: float,
        preview_elapsed_ms: float,
        eval_elapsed_total_ms: float,
        vlm_calls: list[dict[str, Any]],
        frame_count: int,
        selected_indices: list[int],
        vlm_max_long_edge: int,
    ) -> dict[str, Any]:
        detect_elapsed_ms = (time.perf_counter() - detect_start) * 1000.0
        return {
            "detect_elapsed_ms": round(detect_elapsed_ms, 2),
            "scoring_elapsed_ms": round(scoring_elapsed_ms, 2),
            "preprocess_elapsed_total_ms": round(scoring_elapsed_ms, 2),
            "preview_elapsed_ms": round(preview_elapsed_ms, 2),
            "vlm_eval_elapsed_total_ms": round(eval_elapsed_total_ms, 2),
            "vlm_call_count": len(vlm_calls),
            "vlm_calls": vlm_calls,
            "vlm_max_long_edge": vlm_max_long_edge,
            "frame_count": frame_count,
            "selected_indices": selected_indices,
        }

    def _log_detect_timing(self, timing: dict[str, Any]) -> None:
        calls = timing.get("vlm_calls") or []
        call_txt = ", ".join(
            f"idx={item.get('idx')}:{item.get('elapsed_ms')}ms"
            for item in calls
        ) or "-"
        self.logger.info(
            "toast 耗时拆分: detect=%.0fms | cv_score=%.0fms | jpeg_preview=%.0fms | "
            "vlm_total=%.0fms (%s calls: %s) | frames=%s peaks=%s long_edge=%s",
            float(timing.get("detect_elapsed_ms") or 0.0),
            float(timing.get("scoring_elapsed_ms") or 0.0),
            float(timing.get("preview_elapsed_ms") or 0.0),
            float(timing.get("vlm_eval_elapsed_total_ms") or 0.0),
            int(timing.get("vlm_call_count") or 0),
            call_txt,
            timing.get("frame_count"),
            timing.get("selected_indices"),
            timing.get("vlm_max_long_edge"),
        )

    def detect(
        self,
        sampled_frames: list[ExtractedFrame],
        task_id: str,
        expected_toast_keywords: list[str] | None = None,
    ) -> ToastDetectionResult:
        """
        自动扫描帧序列中的 Toast，并判断文案是否符合预期。
        """
        if len(sampled_frames) < 2:
            raise ValueError("toast 检测至少需要 2 帧")

        task_intent = self.prompt_pack.task_intent or "检测动作触发后的 toast 文案是否与预期语义一致。"
        keywords_text = "、".join(expected_toast_keywords or []) if expected_toast_keywords else "无"

        evaluated_candidates: list[dict[str, Any]] = []
        scanned = 0
        total_candidates = len(sampled_frames)
        detect_start = time.perf_counter()
        scoring_elapsed_ms = 0.0
        preview_elapsed_ms = 0.0
        eval_elapsed_total_ms = 0.0
        vlm_calls: list[dict[str, Any]] = []
        candidate_scores: list[dict[str, Any]] = []
        max_vlm = min(max(1, self.top_k_candidates), 2)
        candidate_indices = list(range(total_candidates))
        preview_root: Path | None = None

        if self.enable_preprocess and self.preprocessor is not None:
            t_scoring_start = time.perf_counter()
            try:
                scored = self.preprocessor.score_toast_sequence(
                    [frame.image_path for frame in sampled_frames]
                )
                for item in scored:
                    idx = int(item.get("index", 0))
                    item["timestamp_sec"] = sampled_frames[idx].timestamp_sec
                candidate_scores = scored
                candidate_indices = self._select_score_peaks(
                    candidate_scores,
                    max_peaks=max_vlm,
                    min_separation=3,
                    min_score=self.min_peak_score,
                )
            except Exception as exc:  # pylint: disable=broad-except
                if self.debug:
                    self.logger.warning("toast 候选打分失败: err=%s", exc)
                candidate_scores = [
                    {
                        "index": idx,
                        "timestamp_sec": frame.timestamp_sec,
                        "score": 0.0,
                        "source": "score_error",
                        "reason": str(exc),
                    }
                    for idx, frame in enumerate(sampled_frames)
                ]
                candidate_indices = []
            scoring_elapsed_ms = (time.perf_counter() - t_scoring_start) * 1000.0
            peak_scores = [
                round(float(item.get("score") or 0.0), 3)
                for item in candidate_scores
                if int(item.get("index", -1)) in set(candidate_indices)
            ]
            self.logger.info(
                "toast CV打分: frames=%s elapsed=%.0fms max_vlm=%s peaks=%s scores=%s",
                total_candidates,
                scoring_elapsed_ms,
                max_vlm,
                candidate_indices,
                peak_scores,
            )
        elif total_candidates > max_vlm:
            candidate_indices = list(range(total_candidates - max_vlm, total_candidates))

        score_by_idx = {int(item.get("index", -1)): float(item.get("score") or 0.0) for item in candidate_scores}
        eval_order = sorted(candidate_indices, key=lambda idx: (score_by_idx.get(idx, 0.0), idx), reverse=True)
        if eval_order:
            self.logger.info(
                "toast 开始VLM: task_id=%s peaks=%s (等待网关，默认最长60s/次)",
                task_id,
                eval_order,
            )
        preview_cache: dict[Path, Path] = {}

        def _vlm_frame(source: Path) -> Path:
            nonlocal preview_root, preview_elapsed_ms
            cached = preview_cache.get(source)
            if cached is not None:
                return cached
            preprocessor = self.preprocessor
            if preprocessor is None or self.vlm_max_long_edge <= 0:
                return source
            if preview_root is None:
                preview_root = (
                    preprocessor.artifact_dir / "vlm_preview"
                    if preprocessor.artifact_dir is not None
                    else Path(tempfile.mkdtemp(prefix="toast_vlm_"))
                )
            dest = preview_root / f"{source.stem}_vlm.jpg"
            t_preview = time.perf_counter()
            preview = preprocessor.write_vlm_preview(
                source,
                dest,
                max_long_edge=self.vlm_max_long_edge,
            )
            preview_elapsed_ms += (time.perf_counter() - t_preview) * 1000.0
            preview_cache[source] = preview
            return preview

        for idx in eval_order:
            center = sampled_frames[idx]
            before = sampled_frames[idx - 1] if idx > 0 else sampled_frames[idx]
            segment_task_id = f"{task_id}_toast_scan_{idx:04d}"
            system_prompt = self.prompt_pack.system_prompt
            user_prompt = render_toast_user_prompt(
                prompt_pack=self.prompt_pack,
                context={
                    "task_intent": task_intent,
                    "candidate_timestamp_sec": center.timestamp_sec,
                    "keywords_text": keywords_text,
                },
            )

            scanned += 1
            before_preview = _vlm_frame(before.image_path)
            center_preview = _vlm_frame(center.image_path)
            t_eval_start = time.perf_counter()
            try:
                result = self.evaluator.evaluate_json(
                    before_image=before_preview,
                    after_image=center_preview,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    task_id=segment_task_id,
                    required_fields={
                        "toast_visible": bool,
                        "toast_text": str,
                        "action_semantic": str,
                        "inferred_expected_toast_text": str,
                        "expectation_met": (bool, type(None)),
                        "reverse_inference_risk": str,
                        "action_evidence_from_frame12": str,
                        "toast_evidence_from_frame2": str,
                        "reason": str,
                    },
                    image_role_labels=self._build_toast_image_role_labels(),
                )
            except Exception as exc:  # pylint: disable=broad-except
                eval_elapsed_ms = (time.perf_counter() - t_eval_start) * 1000.0
                eval_elapsed_total_ms += eval_elapsed_ms
                vlm_calls.append(
                    {
                        "idx": idx,
                        "timestamp_sec": round(center.timestamp_sec, 3),
                        "elapsed_ms": round(eval_elapsed_ms, 2),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                self.logger.warning(
                    "toast VLM失败: idx=%s ts=%.2fs elapsed=%.0fms err=%s",
                    idx,
                    center.timestamp_sec,
                    eval_elapsed_ms,
                    exc,
                )
                continue
            eval_elapsed_ms = (time.perf_counter() - t_eval_start) * 1000.0
            eval_elapsed_total_ms += eval_elapsed_ms

            parsed = result.parsed_json
            confidence = float(parsed["confidence"]) if parsed.get("confidence") is not None else 0.0
            toast_text = str(parsed.get("toast_text", ""))
            action_semantic = str(parsed.get("action_semantic", ""))
            inferred_expected_toast_text = str(parsed.get("inferred_expected_toast_text", ""))
            raw_expectation = parsed.get("expectation_met", False)
            raw_expectation_met = bool(raw_expectation) if isinstance(raw_expectation, bool) else False
            toast_visible = bool(parsed["toast_visible"])
            reverse_inference_risk = str(parsed.get("reverse_inference_risk", "low")).strip().lower()
            action_evidence_from_frame12 = str(parsed.get("action_evidence_from_frame12", "")).strip()
            toast_evidence_from_frame2 = str(
                parsed.get("toast_evidence_from_frame2") or parsed.get("toast_evidence_from_frame23", "")
            ).strip()
            action_semantic_norm = action_semantic.strip().lower()
            expectation_unknown = raw_expectation is None
            if action_semantic_norm in {"unknown", "uncertain", "不确定", "无法确定", "未知"}:
                expectation_unknown = True

            reason = str(parsed.get("reason", ""))
            if not toast_visible:
                toast_text = ""
                inferred_expected_toast_text = ""
                raw_expectation_met = False

            semantic_equivalent = self._semantic_equivalent_toast(toast_text, inferred_expected_toast_text)
            expectation_met = raw_expectation_met or semantic_equivalent
            if reverse_inference_risk not in {"low", "high"}:
                reverse_inference_risk = "high"
            is_uncertain_action = expectation_unknown or (reverse_inference_risk == "high")
            if expectation_unknown:
                expectation_met = False

            if expectation_met and (not raw_expectation_met) and semantic_equivalent:
                reason = f"{reason}（后处理修正：文案非逐字一致，但语义一致，按 expectation_met=true 处理。）".strip()
            if toast_visible and expectation_met and reverse_inference_risk == "high":
                expectation_met = False
                reason = (
                    f"{reason}（后处理修正：模型标记存在反向推断风险，按 expectation_met=false 保守处理。）"
                ).strip()
            if toast_visible and (not expectation_met) and is_uncertain_action:
                reason = f"{reason}（后处理修正：动作语义可观测性不足，按不确定处理。）".strip()

            score_item = next((item for item in candidate_scores if int(item.get("index", -1)) == idx), {})
            candidate = {
                "idx": idx,
                "frame": center,
                "toast_visible": toast_visible,
                "toast_text": toast_text,
                "action_semantic": action_semantic,
                "inferred_expected_toast_text": inferred_expected_toast_text,
                "expectation_met": expectation_met,
                "is_uncertain_action": is_uncertain_action,
                "reason": reason,
                "confidence": confidence,
                "raw_response": result.raw_response,
                "preprocess_evidence": {
                    "cv_score": score_item.get("score"),
                    "cv_source": score_item.get("source"),
                    "hot_mask": score_item.get("hot_mask"),
                    "reverse_inference_risk": reverse_inference_risk,
                    "action_evidence_from_frame12": action_evidence_from_frame12,
                    "toast_evidence_from_frame2": toast_evidence_from_frame2,
                },
            }
            evaluated_candidates.append(candidate)
            vlm_calls.append(
                {
                    "idx": idx,
                    "timestamp_sec": round(center.timestamp_sec, 3),
                    "elapsed_ms": round(eval_elapsed_ms, 2),
                    "ok": True,
                    "toast_visible": toast_visible,
                    "cv_score": score_item.get("score"),
                }
            )
            self.logger.info(
                "toast VLM完成: idx=%s ts=%.2fs cv_score=%s elapsed=%.0fms visible=%s uncertain=%s",
                idx,
                center.timestamp_sec,
                score_item.get("score"),
                eval_elapsed_ms,
                toast_visible,
                is_uncertain_action,
            )
            if toast_visible and (not is_uncertain_action):
                break

        self._apply_cross_candidate_conflicts(evaluated_candidates)
        best_candidate = self._select_final_candidate(evaluated_candidates)
        timing = self._build_detect_timing(
            detect_start=detect_start,
            scoring_elapsed_ms=scoring_elapsed_ms,
            preview_elapsed_ms=preview_elapsed_ms,
            eval_elapsed_total_ms=eval_elapsed_total_ms,
            vlm_calls=vlm_calls,
            frame_count=total_candidates,
            selected_indices=eval_order,
            vlm_max_long_edge=self.vlm_max_long_edge,
        )
        self._log_detect_timing(timing)

        if best_candidate is None:
            return ToastDetectionResult(
                bug_detected=False,
                expectation_met=False,
                task_intent=task_intent,
                key_frame_path=None,
                key_frame_timestamp=None,
                toast_text="",
                action_semantic="",
                inferred_expected_toast_text="",
                reason="未能从候选帧中识别到有效 toast 结果。",
                confidence=None,
                raw_response="",
                scanned_candidates=scanned,
                total_candidates=total_candidates,
                evaluated_candidate_indices=eval_order,
                candidate_scores=candidate_scores,
                timing=timing,
                preprocess_evidence=None,
            )

        if best_candidate["toast_visible"]:
            bug_detected = (not best_candidate["expectation_met"]) and (not bool(best_candidate.get("is_uncertain_action")))
        else:
            bug_detected = False
        return ToastDetectionResult(
            bug_detected=bug_detected,
            expectation_met=best_candidate["expectation_met"],
            task_intent=task_intent,
            key_frame_path=best_candidate["frame"].image_path,
            key_frame_timestamp=best_candidate["frame"].timestamp_sec,
            toast_text=best_candidate["toast_text"],
            action_semantic=best_candidate["action_semantic"],
            inferred_expected_toast_text=best_candidate["inferred_expected_toast_text"],
            reason=best_candidate["reason"],
            confidence=best_candidate["confidence"],
            raw_response=best_candidate["raw_response"],
            scanned_candidates=scanned,
            total_candidates=total_candidates,
            evaluated_candidate_indices=eval_order,
            candidate_scores=candidate_scores,
            timing=timing,
            preprocess_evidence=best_candidate.get("preprocess_evidence"),
        )

