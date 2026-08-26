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

    _TEXT_FAILURE_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("network_error", ("网络异常", "网络错误", "网络不给力", "网络连接", "无法连接", "连接到网络", "network")),
        ("request_failed", ("请求失败", "加载失败", "获取失败", "访问失败", "失败", "出错", "出了点问题", "error")),
        ("timeout", ("超时", "timeout", "timed out")),
        ("permission_denied", ("权限", "无权限", "permission", "denied", "unauthorized")),
    )
    # Deny-list only: block known non-failure overlays / empty-result states.
    _NON_FAILURE_EVIDENCE_MARKERS: tuple[str, ...] = (
        "累计阅读",
        "恭喜",
        "成就",
        "抽大奖",
        "礼包",
        "获得徽章",
        "没有符合条件",
        "还没有内容",
        "暂无数据",
        "暂无内容",
        "暂无记录",
        "空空如也",
    )

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
        selected = [int(item["index"]) for item in filtered[: self.probe_top_k]]
        # Always keep the earliest candidate so entry toasts are not dropped by CV ranking.
        if prefilter_scores:
            first_idx = min(int(item["index"]) for item in prefilter_scores)
            if first_idx not in selected:
                selected = sorted(set(selected) | {first_idx})
        return selected

    def _detect_persistent_visual_failure(self, sampled_frames: list[Any]) -> dict[str, Any]:
        """Detect sustained blank/black tail states that VLMs often describe inconsistently."""

        tail_count = min(3, len(sampled_frames))
        if tail_count < 2:
            return {"detected": False, "failure_type": "none", "reason": "", "tail_metrics": []}

        first_gray = self._read_gray_image(sampled_frames[0].image_path)
        tail_metrics: list[dict[str, Any]] = []
        for idx in range(len(sampled_frames) - tail_count, len(sampled_frames)):
            gray = self._read_gray_image(sampled_frames[idx].image_path)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = cv2.countNonZero(edges) / max(1, gray.shape[0] * gray.shape[1])
            change_ratio = self._calc_change_ratio(first_gray, gray)
            tail_metrics.append(
                {
                    "index": idx,
                    "timestamp_sec": sampled_frames[idx].timestamp_sec,
                    "mean": round(float(gray.mean()), 2),
                    "std": round(float(gray.std()), 2),
                    "edge_ratio": round(float(edge_ratio), 4),
                    "initial_change_ratio": round(float(change_ratio), 4),
                }
            )

        white_tail = all(
            item["mean"] >= 240.0
            and item["std"] <= 16.0
            and item["edge_ratio"] <= 0.006
            and item["initial_change_ratio"] >= 0.12
            for item in tail_metrics
        )
        if white_tail:
            return {
                "detected": True,
                "failure_type": "persistent_blank",
                "reason": "尾部连续帧为低纹理高亮白屏，且相对首帧变化明显，判定为持续空白加载失败态。",
                "tail_metrics": tail_metrics,
            }

        skeleton_tail = all(
            210.0 <= item["mean"] <= 235.0
            and item["std"] >= 45.0
            and item["edge_ratio"] <= 0.015
            and item["initial_change_ratio"] >= 0.40
            for item in tail_metrics
        )
        if skeleton_tail:
            return {
                "detected": True,
                "failure_type": "persistent_skeleton",
                "reason": "尾部连续帧为低边缘骨架占位态，且相对首帧变化明显，判定为持续骨架屏加载失败态。",
                "tail_metrics": tail_metrics,
            }

        black_tail = all(
            item["mean"] <= 35.0
            and item["edge_ratio"] <= 0.015
            and item["initial_change_ratio"] >= 0.50
            for item in tail_metrics
        )
        if black_tail:
            return {
                "detected": True,
                "failure_type": "black_screen",
                "reason": "尾部连续帧为低亮度低边缘黑屏，且相对首帧变化明显，判定为持续黑屏加载失败态。",
                "tail_metrics": tail_metrics,
            }

        return {"detected": False, "failure_type": "none", "reason": "", "tail_metrics": tail_metrics}

    @classmethod
    def _classify_failure_text(cls, text: str) -> str:
        normalized = text.strip().lower()
        if not normalized:
            return "none"
        for failure_type, keywords in cls._TEXT_FAILURE_TYPES:
            if any(keyword in normalized for keyword in keywords):
                return failure_type
        return "none"

    @staticmethod
    def _text_says_recovered_or_normal(text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        markers = (
            "未出现加载失败",
            "未发现加载失败",
            "未见加载失败",
            "无加载失败",
            "未出现明确的失败",
            "未见明确失败",
            "未发现明确失败",
            "已恢复",
            "恢复出可用业务内容",
            "页面内容已恢复",
            "显示可用业务内容",
            "内容完整",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _reason_denies_failure_text(text: str) -> bool:
        """Detect VLM reasons that explicitly deny failure-prompt evidence."""

        normalized = text.strip()
        if not normalized:
            return False
        markers = (
            "未显示任何明确的失败",
            "未出现明确的失败文案",
            "未出现加载失败相关文案",
            "无任何失败提示",
            "未见失败文案",
            "没有明确失败文案",
            "未出现明确的失败文案或错误弹窗",
            "均未显示任何明确的失败文案",
            "无任何失败提示或错误弹窗",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _resolve_failure_type_from_observation(
        *,
        text_failure_type: str,
        failure_text_visible: bool,
    ) -> str:
        """Map observation fields to a failure_type; keywords are classification-only."""

        if text_failure_type != "none":
            return text_failure_type
        if failure_text_visible:
            return "request_failed"
        return "request_failed"

    def _normalize_probe_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Convert VLM observations into detector-owned failure decisions.

        Primary gate trusts VLM failure_text_visible (+ page_recovered).
        Keywords only classify failure_type. Persistent blank/skeleton/black without
        failure copy is handled by CV visual_failure.
        """

        evidence_text = str(result.get("evidence_text", "")).strip()
        reason_text = str(result.get("reason", "")).strip()
        observation_text = "\n".join([evidence_text, reason_text])
        failure_text_visible = bool(result.get("failure_text_visible"))
        page_recovered = bool(result.get("page_recovered"))
        visual_state = str(result.get("visual_state", "unknown")).strip().lower() or "unknown"
        text_failure_type = self._classify_failure_text(evidence_text)
        if text_failure_type == "none":
            text_failure_type = self._classify_failure_text(observation_text)
        text_says_normal = self._text_says_recovered_or_normal(observation_text)

        # Only drop structurally inconsistent / clearly non-failure claims.
        if failure_text_visible and not evidence_text:
            failure_text_visible = False
        if failure_text_visible and self._reason_denies_failure_text(reason_text):
            failure_text_visible = False
        if failure_text_visible and any(
            marker in evidence_text for marker in self._NON_FAILURE_EVIDENCE_MARKERS
        ):
            failure_text_visible = False
        # Persistent empty/error-looking frames with failure copy are not "recovered"
        # just because chrome/tabs remain.
        if (
            failure_text_visible
            and page_recovered
            and visual_state in {"blank", "skeleton", "black"}
        ):
            page_recovered = False

        observation_says_failed = failure_text_visible
        normalized_load_failed = observation_says_failed and not page_recovered
        if normalized_load_failed:
            normalized_failure_type = self._resolve_failure_type_from_observation(
                text_failure_type=text_failure_type,
                failure_text_visible=failure_text_visible,
            )
            normalization_reason = f"VLM 观察到失败文案/弹窗，归一化为 {normalized_failure_type}。"
        elif page_recovered and bool(result.get("failure_text_visible")):
            normalized_failure_type = "none"
            normalization_reason = "VLM 观察到后帧已恢复业务内容，按短暂加载/提示处理，归一化为 none。"
        else:
            normalized_failure_type = "none"
            normalization_reason = "VLM 未观察到明确失败文案/弹窗，归一化为 none。"

        legacy_model_load_failed = result.get("model_load_failed")
        legacy_model_failure_type = str(result.get("model_failure_type", "unknown")).strip().lower() or "unknown"
        inconsistent_output = False
        if legacy_model_load_failed is not None:
            inconsistent_output = bool(legacy_model_load_failed) != normalized_load_failed
        if legacy_model_failure_type not in {"unknown", normalized_failure_type}:
            inconsistent_output = True
        if text_says_normal and legacy_model_load_failed is True:
            inconsistent_output = True

        normalized = dict(result)
        normalized.update(
            {
                "load_failed": normalized_load_failed,
                "failure_type": normalized_failure_type,
                "failure_text_visible": failure_text_visible,
                "page_recovered": page_recovered,
                "normalization_reason": normalization_reason,
                "inconsistent_output": inconsistent_output,
                "text_says_normal": text_says_normal,
                "text_failure_type": text_failure_type,
                "observation_says_failed": observation_says_failed,
            }
        )
        return normalized

    @staticmethod
    def _bool_from_parsed(parsed: dict[str, Any], key: str, default: bool = False) -> bool:
        value = parsed.get(key)
        return value if isinstance(value, bool) else default

    def _coerce_probe_observation(
        self,
        parsed: dict[str, Any],
        raw_response: str,
        candidate_index: int,
        before_ts: float,
        center_ts: float,
        after_ts: float,
    ) -> dict[str, Any]:
        """Coerce new observation schema, with backward compatibility for old probe JSON."""

        evidence_text = str(parsed.get("evidence_text", ""))
        reason = str(parsed.get("reason", ""))
        confidence_raw = parsed.get("confidence")
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None

        has_new_schema = "failure_text_visible" in parsed or "page_recovered" in parsed or "visual_state" in parsed
        legacy_load_failed = parsed.get("load_failed") if isinstance(parsed.get("load_failed"), bool) else None
        legacy_failure_type = str(parsed.get("failure_type", "unknown")).strip().lower() or "unknown"

        failure_text_visible = self._bool_from_parsed(parsed, "failure_text_visible", False)
        if not has_new_schema and legacy_load_failed is not None:
            # Legacy responses decided load_failed themselves; preserve that as an observation signal.
            failure_text_visible = bool(legacy_load_failed)

        page_recovered = self._bool_from_parsed(parsed, "page_recovered", False)
        visual_state = str(parsed.get("visual_state", "unknown")).strip().lower() or "unknown"

        return {
            "candidate_index": candidate_index,
            "before_ts": before_ts,
            "center_ts": center_ts,
            "after_ts": after_ts,
            "failure_text_visible": failure_text_visible,
            "evidence_text": evidence_text,
            "page_recovered": page_recovered,
            "visual_state": visual_state,
            "reason": reason,
            "confidence": confidence,
            "raw_response": raw_response,
            "schema_version": "observation_v1" if has_new_schema else "legacy_decision_v0",
            "model_load_failed": legacy_load_failed,
            "model_failure_type": legacy_failure_type,
        }

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
                "after_timestamp_sec": after_ts,
            }
        )
        probe_result = self.evaluator.evaluate_json(
            before_image=before_image,
            after_image=center_image,
            task_id=f"{task_id}_load_failure_probe_{candidate_index:04d}",
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            required_fields={
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
        return self._coerce_probe_observation(
            parsed=probe_result.parsed_json,
            raw_response=probe_result.raw_response,
            candidate_index=candidate_index,
            before_ts=before_ts,
            center_ts=center_ts,
            after_ts=after_ts,
        )

    def detect(self, sampled_frames: list[Any], task_id: str) -> LoadFailureDetectionResult:
        """扫描少量候选帧，判断是否出现加载失败提示或弹窗。"""
        if len(sampled_frames) < 2:
            raise ValueError("页面加载失败检测至少需要 2 帧")

        t_start = time.perf_counter()
        candidate_indices = self._build_candidate_indices(sampled_frames)
        prefilter_scores = self._score_candidates(sampled_frames, candidate_indices)
        probe_indices = self._select_probe_indices(prefilter_scores)
        visual_failure = self._detect_persistent_visual_failure(sampled_frames)

        hits: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []
        probe_errors: list[dict[str, Any]] = []
        raw_parts: list[str] = []
        t_vlm_start = time.perf_counter()
        for idx in probe_indices:
            before_idx = max(0, idx - 1)
            after_idx = min(len(sampled_frames) - 1, idx + 1)
            before_frame = sampled_frames[before_idx]
            center_frame = sampled_frames[idx]
            after_frame = sampled_frames[after_idx]
            try:
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
            except Exception as exc:
                error_result = {
                    "candidate_index": idx,
                    "before_ts": before_frame.timestamp_sec,
                    "center_ts": center_frame.timestamp_sec,
                    "after_ts": after_frame.timestamp_sec,
                    "error": str(exc),
                }
                probe_errors.append(error_result)
                raw_parts.append(f"[load_failure_probe_idx_{idx:04d}_error]\n{exc}")
                self.logger.warning(
                    "加载失败候选帧 VLM 输出无效，跳过该候选: task_id=%s, idx=%s, error=%s",
                    task_id,
                    idx,
                    exc,
                )
                continue
            result = self._normalize_probe_result(result)
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

        recovery_after_hit = None
        if best_hit is not None and not bool(visual_failure.get("detected")):
            # Failure text/toast hits remain bugs even if a later frame looks recovered.
            # Only non-text visual hits may be cleared by later recovery.
            hit_had_failure_text = bool(best_hit.get("failure_text_visible"))
            if not hit_had_failure_text:
                later_recovery_results = [
                    item
                    for item in all_results
                    if int(item["candidate_index"]) > int(best_hit["candidate_index"])
                    and bool(item.get("page_recovered"))
                    and not bool(item.get("failure_text_visible"))
                    and str(item.get("visual_state", "")).strip().lower() == "normal"
                    and not bool(item["load_failed"])
                ]
                if later_recovery_results:
                    recovery_after_hit = sorted(
                        later_recovery_results,
                        key=lambda item: (int(item["candidate_index"]), float(item["center_ts"])),
                        reverse=True,
                    )[0]
                    best_hit = None

        visual_failure_detected = bool(visual_failure.get("detected"))
        bug_detected = best_hit is not None or visual_failure_detected
        failure_type = (
            best_hit["failure_type"]
            if best_hit
            else str(visual_failure.get("failure_type", "none") if visual_failure_detected else "none")
        )
        evidence_text = best_hit["evidence_text"] if best_hit else ""
        confidence = best_hit["confidence"] if best_hit else None
        reason = (
            f"页面加载失败提示命中：{best_hit['reason']}（证据文案：{evidence_text}，候选帧={best_hit['candidate_index']}）"
            if best_hit
            else (
                str(visual_failure.get("reason", ""))
                if visual_failure_detected
                else (
                    "候选帧曾出现加载态，但后续候选帧已恢复业务内容，按短暂加载过程处理。"
                    if recovery_after_hit
                    else "未在候选帧中发现明确的加载失败提示或失败弹窗。"
                )
            )
        )
        cv_metrics = {
            "candidate_indices": candidate_indices,
            "probe_indices": probe_indices,
            "prefilter_scores": prefilter_scores,
            "scan_count": len(all_results),
            "probe_error_count": len(probe_errors),
            "probe_errors": probe_errors,
            "hit_count": len(hits),
            "hits": hits,
            "best_hit": best_hit,
            "recovery_after_hit": recovery_after_hit,
            "visual_failure": visual_failure,
            "inconsistent_probe_count": len([item for item in all_results if item.get("inconsistent_output")]),
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
