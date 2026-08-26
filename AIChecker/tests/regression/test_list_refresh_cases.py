"""
基于 TestAgent/testcase/content_list_refresh 的用例，验证 AIChecker ListRefreshDetector。

该检测依赖 VLM 语义判定，需要通过 .env 或环境变量配置 VLM backend。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_list_refresh_cases.py -q

外部用例目录（默认 ../TestAgent/testcase/content_list_refresh）可通过 TESTAGENT_ROOT 覆盖：
  TESTAGENT_ROOT=/Users/xmq/GitHubRepo/TestAgent AIChecker/.venv/bin/python -m pytest ...

说明：
  TestAgent 用例中的 bounds 多为触发控件（Tab/筛选按钮），而当前 AIChecker
  list_refresh 的 target_bounds 语义是“列表内容 ROI”。
  因此默认不把 case.bounds 当作列表 ROI；若未提供 target_region_bounds，
  则传 target_bounds=None，由检测器退化为主内容区域。
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pytest

from aichecker.vision.checkers.count_change import ControlBounds
from aichecker.vision.checkers.list_refresh import ListRefreshDetector
from aichecker.vision.evaluator import VisionEvaluator, resolve_vlm_backend_config
from testagent_case_utils import (
    REPO_ROOT,
    collect_case_jsons,
    extract_expected_passed,
    record_evaluator_token_usage,
    require_testagent_root,
    resolve_path,
    set_checker_report_meta,
)

CHECKER = "list_refresh"
CASE_CATEGORY = "content_list_refresh"
APP_FROM_NAME = re.compile(r"content-list-refresh-([a-z0-9]+)-")


def _app_from_case_id(case_id: str) -> str:
    match = APP_FROM_NAME.search(case_id)
    return match.group(1) if match else "unknown"


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


def _parse_target_bounds(payload: dict[str, Any]) -> ControlBounds | None:
    """解析列表 ROI；忽略控件级 bounds，避免把 Tab 当成列表区域。"""
    if payload.get("target_region_bounds") is not None:
        raw = payload["target_region_bounds"]
        if isinstance(raw, list):
            return ControlBounds.from_list(raw)
        if isinstance(raw, dict):
            return ControlBounds.from_payload(raw)
        raise ValueError(f"Unsupported target_region_bounds type: {type(raw)}")

    # 显式允许用环境变量强制把 case.bounds 当列表 ROI（兼容实验）
    if _truthy_env("VGA_LIST_REFRESH_USE_CASE_BOUNDS", "0") and payload.get("bounds") is not None:
        raw = payload["bounds"]
        if isinstance(raw, list):
            return ControlBounds.from_list(raw)
        if isinstance(raw, dict):
            return ControlBounds.from_payload(raw)
    return None


def load_list_refresh_case(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("screenshot_a", "screenshot_b", "before_image", "after_image"):
        if data.get(key):
            data[key] = str(resolve_path(json_path, str(data[key])))
    data.setdefault("task_type", "list_refresh")
    data.setdefault("mode", "targeted")
    data.setdefault("expected_list_refresh", True)
    return data


def _resolve_image_pair(payload: dict[str, Any]) -> tuple[Path, Path]:
    before = payload.get("before_image") or payload.get("screenshot_a")
    after = payload.get("after_image") or payload.get("screenshot_b")
    if not before or not after:
        raise ValueError("Missing before/after screenshots")
    return Path(str(before)), Path(str(after))


def _build_list_refresh_detector(case_id: str) -> ListRefreshDetector:
    backend_config = resolve_vlm_backend_config()
    if not backend_config.api_key:
        pytest.skip("VLM API key is required for list_refresh regression tests")

    logger = logging.getLogger("test_list_refresh")
    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "list_refresh_regression" / case_id
    evaluator = VisionEvaluator(
        api_key=backend_config.api_key,
        model=backend_config.model,
        base_url=backend_config.base_url,
        backend=backend_config.backend,
        logger=logger,
        debug=_truthy_env("VGA_DEBUG", "0"),
    )
    return ListRefreshDetector(
        evaluator=evaluator,
        logger=logger,
        debug=_truthy_env("VGA_DEBUG", "0"),
        crops_dir=debug_dir / "crops",
    )


@pytest.mark.parametrize("case_json_path", collect_case_jsons(CASE_CATEGORY), ids=lambda p: p.name)
def test_list_refresh_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
    require_testagent_root()

    case_id = case_json_path.stem
    set_checker_report_meta(
        request,
        checker=CHECKER,
        app=_app_from_case_id(case_id),
        case_id=case_id,
        case_file=str(case_json_path),
    )

    payload = load_list_refresh_case(case_json_path)
    before_image, after_image = _resolve_image_pair(payload)
    if not before_image.exists():
        pytest.skip(f"Before image not found: {before_image}")
    if not after_image.exists():
        pytest.skip(f"After image not found: {after_image}")

    expected_passed = extract_expected_passed(payload, case_json_path)
    set_checker_report_meta(
        request,
        expected_passed=expected_passed,
        template_image=before_image.name,
        target_image=after_image.name,
        preview_template_image=str(before_image.resolve()),
        preview_target_image=str(after_image.resolve()),
    )

    target_bounds = _parse_target_bounds(payload)
    detector = _build_list_refresh_detector(case_id)
    try:
        result = detector.detect(
            before_image=before_image,
            after_image=after_image,
            target_bounds=target_bounds,
            expected_list_refresh=bool(payload.get("expected_list_refresh", True)),
            expected_result_text=payload.get("expected_result") or payload.get("expected_result_text"),
            control_name_hint=payload.get("control_name_hint"),
            task_id=case_id,
        )
    except Exception as exc:  # pylint: disable=broad-except
        err = str(exc)
        api_markers = (
            "Gateway Timeout",
            "Error code: 524",
            "Error code: 502",
            "Error code: 503",
            "InternalServerError",
            "APITimeoutError",
            "RateLimitError",
            "Connection error",
        )
        if any(marker in err or marker in type(exc).__name__ for marker in api_markers):
            pytest.skip(f"{case_id}: VLM API call failed. Error: {exc}")
        raise

    actual_passed = not bool(result.bug_detected)
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        list_refreshed=result.list_refreshed,
        still_loading=result.still_loading,
        expectation_met=result.expectation_met,
        roi_changed_pixel_ratio=result.roi_changed_pixel_ratio,
        detect_elapsed_ms=(result.timing or {}).get("detect_elapsed_ms", ""),
    )
    record_evaluator_token_usage(request, detector.evaluator)

    print(
        json.dumps(
            {
                "case_id": case_id,
                "expected_passed": expected_passed,
                "actual_passed": actual_passed,
                "list_refreshed": result.list_refreshed,
                "still_loading": result.still_loading,
                "expectation_met": result.expectation_met,
                "target_region": result.target_region,
                "target_region_box": result.target_region_box,
                "roi_mean_abs_diff": result.roi_mean_abs_diff,
                "roi_changed_pixel_ratio": result.roi_changed_pixel_ratio,
                "confidence": result.confidence,
                "reason": result.reason,
                "timing": result.timing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"bug_detected={result.bug_detected}, list_refreshed={result.list_refreshed}, "
        f"still_loading={result.still_loading}, expectation_met={result.expectation_met}, "
        f"reason={result.reason!r}"
    )
