"""内容列表刷新检测模块：判定目标区域内容是否刷新。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from aichecker.vision.checkers.count_change import ControlBounds
from aichecker.vision.evaluator import VisionEvaluator
from aichecker.vision.prompt_builders import build_prompt_for_type


@dataclass
class ListRefreshResult:
    """列表刷新检测结果。"""

    bug_detected: bool
    expectation_met: bool
    task_intent: str
    list_refreshed: bool
    still_loading: bool
    target_region: str
    target_region_box: dict[str, int]
    roi_mean_abs_diff: float
    roi_changed_pixel_ratio: float
    reason: str
    confidence: float | None
    raw_response: str
    timing: dict[str, float] | None = None
    preprocess_evidence: dict[str, Any] | None = None


class ListRefreshDetector:
    """基于目标区域 bounds + 前后截图的列表刷新检测器。"""

    # ROI 几乎无变化时直接判未刷新，跳过 VLM。
    _NO_CHANGE_RATIO_THRESHOLD = 1e-4
    _NO_CHANGE_MEAN_ABS_DIFF_THRESHOLD = 0.5

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        crops_dir: Path | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.crops_dir = crops_dir

    @staticmethod
    def _read_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return image

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

    @classmethod
    def _is_roi_unchanged(cls, mean_abs_diff: float, changed_pixel_ratio: float) -> bool:
        return (
            changed_pixel_ratio <= cls._NO_CHANGE_RATIO_THRESHOLD
            and mean_abs_diff <= cls._NO_CHANGE_MEAN_ABS_DIFF_THRESHOLD
        )

    @staticmethod
    def _resolve_target_region(
        image_width: int,
        image_height: int,
        target_bounds: ControlBounds | None,
    ) -> tuple[dict[str, int], str]:
        if target_bounds is None:
            top_margin = int(image_height * 0.10)
            bottom_margin = int(image_height * 0.10)
            return (
                {"x1": 0, "y1": top_margin, "x2": image_width, "y2": image_height - bottom_margin},
                "主内容区域（未提供目标区域 bounds，自动退化）",
            )

        x1 = max(0, min(image_width - 1, int(target_bounds.x)))
        y1 = max(0, min(image_height - 1, int(target_bounds.y)))
        x2 = max(1, min(image_width, int(target_bounds.x + target_bounds.width)))
        y2 = max(1, min(image_height, int(target_bounds.y + target_bounds.height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"目标区域 bounds 超出图片范围或面积无效: {target_bounds}")
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "目标区域 bounds 指定的内容区域"

    def _save_roi_artifacts(
        self,
        before_roi: Any,
        after_roi: Any,
        diff_gray: Any,
        task_id: str,
    ) -> list[Path]:
        if self.crops_dir is None:
            return []
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for image, name in [
            (before_roi, f"{task_id}_pp_list_before_roi.png"),
            (after_roi, f"{task_id}_pp_list_after_roi.png"),
            (diff_gray, f"{task_id}_pp_list_diff_gray.png"),
        ]:
            out_path = self.crops_dir / name
            if cv2.imwrite(str(out_path), image):
                paths.append(out_path)
        return paths

    def _build_result(
        self,
        *,
        list_refreshed: bool,
        still_loading: bool,
        expected_list_refresh: bool,
        task_intent: str,
        target_region: str,
        list_region: dict[str, int],
        roi_mean_abs_diff: float,
        roi_changed_pixel_ratio: float,
        reason: str,
        confidence: float | None,
        raw_response: str,
        detect_start: float,
        vlm_elapsed_ms: float,
        preprocess_evidence: dict[str, Any],
    ) -> ListRefreshResult:
        expectation_met = (list_refreshed == bool(expected_list_refresh)) and not still_loading
        return ListRefreshResult(
            bug_detected=not expectation_met,
            expectation_met=expectation_met,
            task_intent=task_intent,
            list_refreshed=list_refreshed,
            still_loading=still_loading,
            target_region=target_region,
            target_region_box=list_region,
            roi_mean_abs_diff=round(roi_mean_abs_diff, 3),
            roi_changed_pixel_ratio=round(roi_changed_pixel_ratio, 4),
            reason=reason,
            confidence=confidence,
            raw_response=raw_response,
            timing={
                "detect_elapsed_ms": round((time.perf_counter() - detect_start) * 1000.0, 2),
                "vlm_elapsed_ms": round(vlm_elapsed_ms, 2),
            },
            preprocess_evidence=preprocess_evidence,
        )

    def detect(
        self,
        before_image: Path,
        after_image: Path,
        target_bounds: ControlBounds | None,
        expected_list_refresh: bool = True,
        expected_result_text: str | None = None,
        control_name_hint: str | None = None,
        task_id: str = "list_refresh_task",
    ) -> ListRefreshResult:
        """检测目标区域在控件响应后是否发生刷新。"""
        if not before_image.exists() or not after_image.exists():
            raise FileNotFoundError("输入前后页面截图不存在")
        if target_bounds is not None:
            target_bounds.validate()
        detect_start = time.perf_counter()

        before = self._read_image(before_image)
        after = self._read_image(after_image)
        h, w = before.shape[:2]
        if after.shape[:2] != (h, w):
            raise ValueError("前后截图分辨率不一致，无法执行列表刷新检测")

        list_region, list_region_desc = self._resolve_target_region(
            image_width=w,
            image_height=h,
            target_bounds=target_bounds,
        )
        x1, y1, x2, y2 = list_region["x1"], list_region["y1"], list_region["x2"], list_region["y2"]
        before_roi = before[y1:y2, x1:x2]
        after_roi = after[y1:y2, x1:x2]
        roi_mean_abs_diff, roi_changed_pixel_ratio, diff_gray = self._calc_diff_metrics(before_roi, after_roi)
        preprocess_structured = {
            "evidence_type": "list_refresh_roi_diff",
            "list_region_desc": list_region_desc,
            "list_region_box": list_region,
            "roi_mean_abs_diff": round(roi_mean_abs_diff, 3),
            "roi_changed_pixel_ratio": round(roi_changed_pixel_ratio, 4),
            "notes": "该指标用于衡量目标列表区域在前后截图中的变化强度，辅助判定是否已刷新。",
        }
        task_intent = "检测控件触发后，目标内容列表区域是否发生刷新变化。"

        # CV 分流：ROI 几乎一致 → 未刷新，跳过 VLM。
        if self._is_roi_unchanged(roi_mean_abs_diff, roi_changed_pixel_ratio):
            self._save_roi_artifacts(before_roi, after_roi, diff_gray, task_id=task_id)
            return self._build_result(
                list_refreshed=False,
                still_loading=False,
                expected_list_refresh=expected_list_refresh,
                task_intent=task_intent,
                target_region=list_region_desc,
                list_region=list_region,
                roi_mean_abs_diff=roi_mean_abs_diff,
                roi_changed_pixel_ratio=roi_changed_pixel_ratio,
                reason=(
                    "CV short-circuit: 目标列表 ROI 前后几乎无变化，判定为未刷新，跳过 VLM。"
                    f" mean_abs_diff={roi_mean_abs_diff:.3f}, changed_ratio={roi_changed_pixel_ratio:.4f}"
                ),
                confidence=1.0,
                raw_response="",
                detect_start=detect_start,
                vlm_elapsed_ms=0.0,
                preprocess_evidence=preprocess_structured,
            )

        preprocess_extra_images = self._save_roi_artifacts(before_roi, after_roi, diff_gray, task_id=task_id)
        preprocess_summary = (
            "前处理证据(list_refresh): "
            f"region={list_region_desc}, box=({x1},{y1})-({x2},{y2}), "
            f"mean_abs_diff={roi_mean_abs_diff:.3f}, changed_ratio={roi_changed_pixel_ratio:.4f}"
        )
        prompt_pack = build_prompt_for_type(
            task_type="list_refresh",
            context={
                "expected_list_refresh": expected_list_refresh,
                "expected_result_text": expected_result_text or "未提供",
                "control_name_hint": control_name_hint or "未提供",
                "target_bounds_x": target_bounds.x if target_bounds else None,
                "target_bounds_y": target_bounds.y if target_bounds else None,
                "target_bounds_width": target_bounds.width if target_bounds else None,
                "target_bounds_height": target_bounds.height if target_bounds else None,
                "preprocess_summary": preprocess_summary,
                "preprocess_structured": preprocess_structured,
            },
        )
        if self.debug:
            self.logger.info(
                "ListRefresh 调试: task_id=%s, expected_list_refresh=%s, list_region=%s",
                task_id,
                expected_list_refresh,
                list_region,
            )

        t_vlm_start = time.perf_counter()
        eval_result = self.evaluator.evaluate_json(
            before_image=before_image,
            after_image=after_image,
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            task_id=task_id,
            required_fields={
                "list_refreshed": bool,
                "still_loading": bool,
                "target_region": str,
                "reason": str,
            },
            extra_image_paths=preprocess_extra_images,
        )
        vlm_elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        parsed = eval_result.parsed_json
        return self._build_result(
            list_refreshed=bool(parsed.get("list_refreshed", False)),
            still_loading=bool(parsed.get("still_loading", False)),
            expected_list_refresh=expected_list_refresh,
            task_intent=prompt_pack.task_intent,
            target_region=str(parsed.get("target_region", list_region_desc)),
            list_region=list_region,
            roi_mean_abs_diff=roi_mean_abs_diff,
            roi_changed_pixel_ratio=roi_changed_pixel_ratio,
            reason=str(parsed.get("reason", "")),
            confidence=float(parsed["confidence"]) if parsed.get("confidence") is not None else None,
            raw_response=eval_result.raw_response,
            detect_start=detect_start,
            vlm_elapsed_ms=vlm_elapsed_ms,
            preprocess_evidence=preprocess_structured,
        )
