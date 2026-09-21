"""
基于 testcase/black_white_screen 目录的测试用例，
验证 AIChecker aichecker.vision.checkers.loading 的黑白屏/花屏检测能力。

JSON 与视频默认在本仓库 testcase/；可通过环境变量 TESTAGENT_ROOT 覆盖根目录。

运行示例：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_black_white_screen_cases.py -q
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

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

CHECKER = "black_white_screen"
APP = "xhs"


def _resolve_black_white_detected(anomaly_type: str | None) -> bool:
    return anomaly_type in {"black_screen", "white_screen", "garbled_screen"}


@pytest.mark.parametrize("case_json_path", collect_case_jsons("black_white_screen"), ids=lambda p: p.name)
def test_black_white_screen_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
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
        assert len(frames) >= 2, f"Need at least 2 frames, got {len(frames)} for {video_path}"
        anomaly_type, reason, metrics = detector._detect_black_white_screen(frames)
        signal: dict[str, Any] = {
            "detected": _resolve_black_white_detected(anomaly_type),
            "anomaly_type": anomaly_type or "none",
            "reason": reason,
            "evidence": metrics,
        }
        actual_passed = not signal["detected"]

    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        anomaly_type=signal["anomaly_type"],
    )

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, "
        f"actual_passed={actual_passed}, black_white_detected={signal['detected']}, "
        f"anomaly_type={signal['anomaly_type']}, reason={signal['reason']!r}"
    )
