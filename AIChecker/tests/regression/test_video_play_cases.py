"""
基于 TestAgent/testcase/video_play 的用例，验证 AIChecker VideoPlayDetector 与 vision CLI。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_video_play_cases.py -q

外部用例目录（默认 ../TestAgent/testcase/video_play）可通过 TESTAGENT_ROOT 覆盖。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from aichecker.vision.checkers.video_play import VideoPlayDetector
from testagent_case_utils import (
    PROJECT_ROOT,
    collect_case_jsons,
    extract_expected_passed,
    extract_video_frames,
    load_case,
    require_testagent_root,
    set_checker_report_meta,
)

CHECKER = "video_play"
CASE_CATEGORY = "video_play"
APP_FROM_NAME = re.compile(r"video-play-([a-z0-9]+)-")


def _app_from_case_id(case_id: str) -> str:
    match = APP_FROM_NAME.search(case_id)
    return match.group(1) if match else "unknown"


def _cli_payload(payload: dict, case_id: str) -> dict:
    cli_payload = dict(payload)
    cli_payload.setdefault("task_id", case_id)
    cli_payload.setdefault("mode", "targeted")
    cli_payload.setdefault("task_type", "video_play")
    return cli_payload


def _run_vision_cli(payload: dict) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "aichecker.vision.cli",
        "--input-json",
        json.dumps(payload, ensure_ascii=False),
    ]
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    return proc.wait(), "".join(output_lines)


def _resolve_video_play_pass(report: dict) -> tuple[bool, str]:
    for seg in report.get("segment_results", []):
        if not isinstance(seg, dict):
            continue
        if seg.get("detector") == "video_play_detector":
            return (not bool(seg.get("bug_detected", False))), "video_play_detector.bug_detected"
    bug_detected = bool(report.get("video_level_result", {}).get("bug_detected", False))
    return (not bug_detected), "video_level_result.bug_detected"


@pytest.mark.parametrize("case_json_path", collect_case_jsons(CASE_CATEGORY), ids=lambda p: p.name)
def test_video_play_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
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

    play_timestamp_sec = payload.get("play_timestamp_sec")
    if play_timestamp_sec is None:
        play_timestamp_sec = payload.get("play_time_sec")

    detector = VideoPlayDetector()
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
            play_timestamp_sec=float(play_timestamp_sec) if play_timestamp_sec is not None else None,
        )

    actual_passed = not bool(result.bug_detected)
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        anomaly_type=result.anomaly_type,
    )

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"bug_detected={result.bug_detected}, anomaly_type={result.anomaly_type}, "
        f"play_timestamp_sec={result.play_timestamp_sec}, reason={result.reason!r}"
    )


@pytest.mark.parametrize("case_json_path", collect_case_jsons(CASE_CATEGORY), ids=lambda p: p.name)
def test_video_play_from_testagent_case_via_cli(case_json_path: Path, request: pytest.FixtureRequest) -> None:
    require_testagent_root()

    case_id = case_json_path.stem
    set_checker_report_meta(
        request,
        checker=CHECKER,
        app=_app_from_case_id(case_id),
        case_id=f"{case_id}_cli",
        case_file=str(case_json_path),
    )

    payload = load_case(case_json_path)
    video_path = Path(str(payload.get("video_file", "")))
    if not video_path.exists():
        pytest.skip(f"Video not found: {video_path}")

    expected_passed = extract_expected_passed(payload, case_json_path)
    set_checker_report_meta(request, expected_passed=expected_passed)

    code, output = _run_vision_cli(_cli_payload(payload, case_id))
    assert code == 0, f"aichecker.vision.cli exited with code={code}\n{output}"

    report_marker = "测试报告已生成:"
    report_path = None
    for line in output.splitlines():
        if report_marker in line:
            report_path = Path(line.split(report_marker, 1)[1].strip())
    assert report_path is not None and report_path.exists(), f"Report path not found in CLI output:\n{output}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("input", {}).get("task_type") == "video_play"
    actual_passed, source = _resolve_video_play_pass(report)
    set_checker_report_meta(request, actual_passed=actual_passed, anomaly_type=source)

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"source={source}, report={report_path}"
    )
