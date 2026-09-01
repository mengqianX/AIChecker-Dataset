from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..models import Bounds, CheckResult, ControlInfo
from ..vision.checkers.count_change import ControlBounds, CountChangeDetector
from ..vision.evaluator import VisionEvaluator, resolve_vlm_config
from ..vision.preprocessor import GuiPreprocessor


def _resolve_expected_change(payload: Dict[str, Any]) -> str:
    if payload.get("expected_count_change") is not None:
        return str(payload["expected_count_change"])
    if payload.get("expected_passed") is not None:
        return "any_change" if bool(payload["expected_passed"]) else "no_change"
    return "any_change"


def _is_auth_error(error: BaseException) -> bool:
    error_text = str(error)
    error_type = type(error).__name__
    return (
        "401" in error_text
        or "invalid_api_key" in error_text.lower()
        or "incorrect api key" in error_text.lower()
        or "authentication" in error_text.lower()
        or "unauthorized" in error_text.lower()
        or error_type in {"AuthenticationError", "InvalidAPIKeyError"}
    )


def _build_api_error_result(bounds: Bounds, error: BaseException) -> CheckResult:
    error_text = str(error)
    return CheckResult(
        passed=False,
        basis=f"API调用失败: {error_text}",
        control_info=ControlInfo(bounds=bounds, source="api_error"),
        details={
            "error": error_text,
            "error_type": type(error).__name__,
            "is_api_error": True,
            "is_auth_error": _is_auth_error(error),
        },
    )


def check_count_change(
    payload: Dict[str, Any],
    debug_dir: Path | None = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> CheckResult:
    """Detect whether a target control caused the expected related count change."""
    screenshot_a = payload.get("screenshot_a") or payload.get("before_image")
    screenshot_b = payload.get("screenshot_b") or payload.get("after_image") or payload.get("screenshot")
    if not screenshot_a or not screenshot_b:
        raise ValueError("Both screenshot_a and screenshot_b (or before_image/after_image) must be provided")

    bounds = Bounds.from_sequence(payload["bounds"])
    control_bounds = ControlBounds.from_list(list(bounds.as_box()))
    debug_root = Path(debug_dir) if debug_dir else None
    logger = logging.getLogger("aichecker.vision.count_change")
    vlm_config = resolve_vlm_config(
        api_key=api_key or payload.get("api_key"),
        model=model or payload.get("model"),
        base_url=base_url or payload.get("base_url"),
    )
    resolved_api_key = vlm_config.api_key
    resolved_model = vlm_config.model
    resolved_base_url = vlm_config.base_url

    evaluator = VisionEvaluator(
        api_key=resolved_api_key,
        model=resolved_model or "gpt-4o",
        base_url=resolved_base_url,
        logger=logger,
        debug=bool(payload.get("debug", False)),
        prompt_log_path=(debug_root / "vlm_prompts.jsonl") if debug_root else None,
        prompt_text_path=(debug_root / "full_prompts.txt") if debug_root else None,
    )
    preprocessor = GuiPreprocessor(
        artifact_dir=(debug_root / "preprocess") if debug_root else None,
        logger=logger,
    )
    detector = CountChangeDetector(
        evaluator=evaluator,
        logger=logger,
        debug=bool(payload.get("debug", False)),
        crops_dir=(debug_root / "crops") if debug_root else None,
        preprocessor=preprocessor,
        enable_preprocess=bool(payload.get("enable_preprocess", True)),
    )

    try:
        result = detector.detect(
            before_image=Path(str(screenshot_a)),
            after_image=Path(str(screenshot_b)),
            control_bounds=control_bounds,
            expected_change=_resolve_expected_change(payload),
            metric_hints=[str(x) for x in payload.get("metric_hints", [])] if payload.get("metric_hints") else None,
            control_name_hint=(
                str(payload.get("control_name_hint"))
                if payload.get("control_name_hint")
                else (str(payload.get("label")) if payload.get("label") else None)
            ),
            task_id=str(payload.get("task_id", "count_change_task")),
        )
    except ImportError:
        raise
    except Exception as exc:
        if _is_auth_error(exc):
            raise ValueError(f"API authentication failed (401): {exc}") from exc
        return _build_api_error_result(bounds=bounds, error=exc)

    control_info = ControlInfo(
        bounds=bounds,
        semantics=result.semantic_target or None,
        source="vision_count_change",
        extras={
            "linked_metric": result.linked_metric,
            "before_value": result.before_value,
            "after_value": result.after_value,
            "value_changed": result.value_changed,
            "change_direction": result.change_direction,
            "confidence": result.confidence,
        },
    )
    details: Dict[str, Any] = {
        "method": "vision_count_change",
        "model": resolved_model or "gpt-4o",
        "base_url": resolved_base_url,
        "semantic_target": result.semantic_target,
        "linked_metric": result.linked_metric,
        "before_value": result.before_value,
        "after_value": result.after_value,
        "value_changed": result.value_changed,
        "change_direction": result.change_direction,
        "expectation_met": result.expectation_met,
        "confidence": result.confidence,
        "raw_response": result.raw_response,
        "timing": result.timing,
        "preprocess_evidence": result.preprocess_evidence,
    }
    return CheckResult(
        passed=result.expectation_met,
        basis=result.reason,
        control_info=control_info,
        details=details,
    )
