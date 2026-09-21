"""
基于 testcase/video_peek 的用例，验证 AIChecker SeekPlaybackDetector。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_seek_playback_cases.py -q

用例目录默认本仓库 testcase/video_peek；可通过 TESTAGENT_ROOT 覆盖根目录。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from aichecker.vision.checkers.seek_playback import SeekPlaybackDetector
from testagent_case_utils import (
    collect_case_jsons,
    extract_expected_passed,
    extract_video_frames,
    load_case,
    require_testagent_root,
    set_checker_report_meta,
)

CHECKER = "seek_playback"
CASE_CATEGORY = "video_peek"
APP_FROM_NAME = re.compile(r"video-peek-([a-z0-9]+)-")


def _app_from_case_id(case_id: str) -> str:
    match = APP_FROM_NAME.search(case_id)
    return match.group(1) if match else "unknown"


@pytest.mark.parametrize("case_json_path", collect_case_jsons(CASE_CATEGORY), ids=lambda p: p.name)
def test_seek_playback_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
    require_testagent_root()

    case_id = case_json_path.stem
    set_checker_report_meta(
        request,
        checker=CHECKER,
        app=_app_from_case_id(case_id),
        case_id=case_id,
        case_file=str(case_json_path),
    )

    payload = load_case(case_json_path)
    video_path = Path(str(payload.get("video_file", "")))
    if not video_path.exists():
        pytest.skip(f"Video not found: {video_path}")

    expected_passed = extract_expected_passed(payload, case_json_path)
    set_checker_report_meta(request, expected_passed=expected_passed)

    seek_timestamp_sec = payload.get("seek_timestamp_sec")
    if seek_timestamp_sec is None:
        seek_timestamp_sec = payload.get("seek_time_sec")

    detector = SeekPlaybackDetector()
    with extract_video_frames(
        video_path=video_path,
        prefix=case_id,
        sample_interval_sec=float(payload.get("sample_interval_sec", 1.0)),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
    ) as frames:
        assert len(frames) >= 2, f"Need at least 2 frames for {video_path}"
        result = detector.detect(
            frames,
            task_id=case_id,
            seek_timestamp_sec=float(seek_timestamp_sec) if seek_timestamp_sec is not None else None,
        )

    actual_passed = not bool(result.bug_detected)
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        anomaly_type=result.anomaly_type,
        seek_detected=result.seek_detected,
        seek_timestamp_sec=result.seek_timestamp_sec,
        detect_elapsed_ms=(result.timing or {}).get("elapsed_ms", ""),
        prompt_call_count=0,
        total_tokens=0,
    )

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"bug_detected={result.bug_detected}, anomaly_type={result.anomaly_type}, "
        f"seek_detected={result.seek_detected}, seek_timestamp_sec={result.seek_timestamp_sec}, "
        f"reason={result.reason!r}"
    )
