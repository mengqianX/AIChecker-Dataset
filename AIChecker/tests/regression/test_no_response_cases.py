"""
基于 TestAgent/testcase/no_response 的用例，验证 AIChecker LoadingDetector 的无响应检测。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_no_response_cases.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aichecker.vision.checkers.loading import LoadingDetector
from testagent_case_utils import (
    collect_case_jsons,
    extract_expected_passed,
    extract_video_frames,
    load_case,
    no_vlm_evaluator,
    require_testagent_root,
    set_checker_report_meta,
)

SIGNAL_KEY = "no_response"
CHECKER = "no_response"
APP = "zhihu"


@pytest.mark.parametrize("case_json_path", collect_case_jsons("no_response"), ids=lambda p: p.name)
def test_no_response_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
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

    detector = LoadingDetector(evaluator=no_vlm_evaluator())
    with extract_video_frames(
        video_path=video_path,
        prefix=case_json_path.stem,
        sample_interval_sec=float(payload.get("sample_interval_sec", 1.0)),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
    ) as frames:
        assert len(frames) >= 2, f"Need at least 2 frames for {video_path}"
        result = detector.detect(frames, task_id=case_json_path.stem)
        signal = result.cv_metrics["signals"][SIGNAL_KEY]

    actual_passed = not bool(signal["detected"])
    set_checker_report_meta(request, actual_passed=actual_passed, anomaly_type=signal.get("status", ""))

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"signal_detected={signal['detected']}, status={signal.get('status')}, reason={signal.get('reason')!r}"
    )
