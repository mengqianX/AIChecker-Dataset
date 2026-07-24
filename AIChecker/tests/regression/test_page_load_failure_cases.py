"""
基于 TestAgent/testcase/page_load_failure 的用例，验证 AIChecker LoadFailurePromptDetector。

该检测依赖 VLM 语义探测，需要通过 .env 或环境变量配置 VLM backend。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_page_load_failure_cases.py -q
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aichecker.vision.checkers.load_failure import LoadFailurePromptDetector
from aichecker.vision.evaluator import VisionEvaluator, resolve_vlm_backend_config
from testagent_case_utils import (
    collect_case_jsons,
    extract_expected_passed,
    extract_video_frames,
    load_case,
    require_testagent_root,
    set_checker_report_meta,
)

CHECKER = "page_load_failure"
APP = "jd"


def _build_load_failure_detector() -> LoadFailurePromptDetector:
    backend_config = resolve_vlm_backend_config()
    if not backend_config.api_key:
        pytest.skip("VLM API key is required for page_load_failure regression tests")
    evaluator = VisionEvaluator(
        api_key=backend_config.api_key,
        model=backend_config.model,
        base_url=backend_config.base_url,
        backend=backend_config.backend,
        logger=logging.getLogger("test_page_load_failure"),
        debug=False,
    )
    return LoadFailurePromptDetector(evaluator=evaluator)


@pytest.mark.parametrize("case_json_path", collect_case_jsons("page_load_failure"), ids=lambda p: p.name)
def test_page_load_failure_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
    require_testagent_root()

    set_checker_report_meta(
        request,
        checker=CHECKER,
        app=APP,
        case_id=case_json_path.stem,
        case_file=str(case_json_path),
    )

    payload = load_case(case_json_path)
    video_path = Path(str(payload.get("video_file", "")))
    if not video_path.exists():
        pytest.skip(f"Video not found: {video_path}")

    expected_passed = extract_expected_passed(payload, case_json_path)
    set_checker_report_meta(request, expected_passed=expected_passed)

    detector = _build_load_failure_detector()
    with extract_video_frames(
        video_path=video_path,
        prefix=case_json_path.stem,
        sample_interval_sec=float(payload.get("sample_interval_sec", 1.0)),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
    ) as frames:
        assert len(frames) >= 2, f"Need at least 2 frames for {video_path}"
        result = detector.detect(frames, task_id=case_json_path.stem)

    actual_passed = not bool(result.bug_detected)
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        anomaly_type=result.anomaly_type,
    )

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"bug_detected={result.bug_detected}, anomaly_type={result.anomaly_type}, reason={result.reason!r}"
    )
