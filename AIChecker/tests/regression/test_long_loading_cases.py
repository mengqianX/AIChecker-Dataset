"""
基于 TestAgent/testcase/long_loading 的用例，验证 AIChecker LoadingDetector 的长时间加载检测。

CV 明确时只走帧间变化信号；CV 不确定时会回退到真实 VLM。
需要通过 .env 或环境变量配置 VLM backend。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_long_loading_cases.py -q
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from aichecker.vision.checkers.loading import LoadingDetector
from aichecker.vision.evaluator import VisionEvaluator, resolve_vlm_backend_config
from testagent_case_utils import (
    collect_case_jsons,
    extract_expected_passed,
    extract_video_frames,
    load_case,
    record_evaluator_token_usage,
    require_testagent_root,
    set_checker_report_meta,
)

SIGNAL_KEY = "long_loading"
CHECKER = "long_loading"
APP = "qqmusic"


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


def _build_loading_detector() -> LoadingDetector:
    backend_config = resolve_vlm_backend_config()
    if not backend_config.api_key:
        pytest.skip("VLM API key is required for long_loading regression tests")

    logger = logging.getLogger("test_long_loading")
    evaluator = VisionEvaluator(
        api_key=backend_config.api_key,
        model=backend_config.model,
        base_url=backend_config.base_url,
        backend=backend_config.backend,
        logger=logger,
        debug=_truthy_env("VGA_DEBUG", "0"),
    )
    return LoadingDetector(
        evaluator=evaluator,
        logger=logger,
        debug=_truthy_env("VGA_DEBUG", "0"),
    )


@pytest.mark.parametrize("case_json_path", collect_case_jsons("long_loading"), ids=lambda p: p.name)
def test_long_loading_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
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

    detector = _build_loading_detector()
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

    # CV 明确的 long_loading 信号，或 VLM 兜底判为 long_loading，都算检测到。
    long_loading_detected = bool(signal["detected"]) or "long_loading" in (result.anomaly_types or [])
    actual_passed = not long_loading_detected
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        anomaly_type=signal.get("status", "") or result.anomaly_type,
        detect_elapsed_ms=(result.timing or {}).get("detect_elapsed_ms", ""),
    )
    record_evaluator_token_usage(request, detector.evaluator)

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"signal_detected={signal['detected']}, anomaly_types={result.anomaly_types}, "
        f"decision_source={result.decision_source}, status={signal.get('status')}, "
        f"reason={signal.get('reason') or result.reason!r}"
    )
