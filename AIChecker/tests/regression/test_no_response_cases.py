"""
基于 TestAgent/testcase/no_response 的用例，验证 AIChecker LoadingDetector 的无响应检测。

策略（CV 先定、VLM 兜底；no_response 与 long_loading 分离）：
1. 先对 ROI/全帧做纯 CV（不调用 VLM）
2. CV 明确无响应 → 直接采用
3. CV 明确稳定响应（roi/full stable）→ 直接通过
4. 其余不确定（含 CV 的 long_loading）→ 对全帧调用 **no_response 专用 VLM**
5. 不把 long_loading 由 CV/VLM 混进 no_response 结论

需要通过 .env 或环境变量配置 VLM backend。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_no_response_cases.py -q
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from aichecker.vision.checkers.loading import LoadingDetector
from aichecker.vision.evaluator import VisionEvaluator, resolve_vlm_backend_config
from testagent_case_utils import (
    collect_case_jsons,
    crop_frames_to_bounds,
    extract_expected_passed,
    extract_video_frames,
    load_case,
    record_evaluator_token_usage,
    require_testagent_root,
    set_checker_report_meta,
)

SIGNAL_KEY = "no_response"
CHECKER = "no_response"
APP = "zhihu"


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


def _build_loading_detector() -> LoadingDetector:
    backend_config = resolve_vlm_backend_config()
    if not backend_config.api_key:
        pytest.skip("VLM API key is required for no_response regression tests")

    logger = logging.getLogger("test_no_response")
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


def _signal(result: Any) -> dict[str, Any]:
    return result.cv_metrics["signals"][SIGNAL_KEY]


def _is_no_response_hit(result: Any) -> bool:
    signal = _signal(result)
    return bool(signal["detected"]) or "no_response" in (result.anomaly_types or [])


def _needs_vlm(result: Any) -> bool:
    return bool((result.cv_metrics.get("vlm_fallback") or {}).get("needed"))


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

    detector = _build_loading_detector()
    with extract_video_frames(
        video_path=video_path,
        prefix=case_json_path.stem,
        sample_interval_sec=float(payload.get("sample_interval_sec", 1.0)),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
    ) as frames:
        assert len(frames) >= 2, f"Need at least 2 frames for {video_path}"

        bounds = payload.get("bounds")
        use_roi = isinstance(bounds, list) and len(bounds) == 4 and bounds != [0, 0, 0, 0]
        if use_roi:
            with tempfile.TemporaryDirectory(prefix="no_response_roi_") as roi_tmp:
                roi_frames = crop_frames_to_bounds(frames, bounds, Path(roi_tmp))
                # 1) 先纯 CV，避免 ROI uncertain/long_loading 过早打 VLM 误杀。
                roi_cv = detector.detect(
                    roi_frames,
                    task_id=f"{case_json_path.stem}_roi_cv",
                    enable_vlm_fallback=False,
                )
                full_cv = detector.detect(
                    frames,
                    task_id=f"{case_json_path.stem}_full_cv",
                    enable_vlm_fallback=False,
                )
                roi_stable = bool(roi_cv.cv_metrics.get("stable_baseline_progress"))
                full_stable = bool(full_cv.cv_metrics.get("stable_baseline_progress"))

                if _is_no_response_hit(roi_cv):
                    result, signal = roi_cv, _signal(roi_cv)
                elif _is_no_response_hit(full_cv):
                    result, signal = full_cv, _signal(full_cv)
                elif roi_stable:
                    result, signal = roi_cv, _signal(roi_cv)
                elif full_stable:
                    result, signal = full_cv, _signal(full_cv)
                elif _needs_vlm(full_cv) or _needs_vlm(roi_cv):
                    # 2) 不确定：对全帧问 no_response 专用 VLM（与 long_loading 分离）。
                    result = detector.detect(
                        frames,
                        task_id=f"{case_json_path.stem}_full_vlm",
                        enable_vlm_fallback=True,
                        vlm_prompt_type="no_response",
                    )
                    signal = _signal(result)
                else:
                    result, signal = roi_cv, _signal(roi_cv)
        else:
            result = detector.detect(
                frames,
                task_id=case_json_path.stem,
                enable_vlm_fallback=True,
                vlm_prompt_type="no_response",
            )
            signal = _signal(result)

    no_response_detected = _is_no_response_hit(result)
    actual_passed = not no_response_detected
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        anomaly_type=signal.get("status", "") or getattr(result, "anomaly_type", ""),
        detect_elapsed_ms=(getattr(result, "timing", None) or {}).get("detect_elapsed_ms", ""),
    )
    record_evaluator_token_usage(request, detector.evaluator)

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"signal_detected={signal['detected']}, anomaly_types={result.anomaly_types}, "
        f"decision_source={result.decision_source}, status={signal.get('status')}, "
        f"reason={signal.get('reason') or result.reason!r}"
    )
