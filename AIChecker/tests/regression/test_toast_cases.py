"""
基于 TestAgent/testcase/toast 的用例，验证 AIChecker ToastMessageDetector。

该检测依赖 VLM 语义判定，需要通过 .env 或环境变量配置 VLM。

运行：
  AIChecker/.venv/bin/python -m pytest AIChecker/tests/regression/test_toast_cases.py -q

外部用例目录（默认 ../TestAgent/testcase/toast）可通过 TESTAGENT_ROOT 覆盖：
  TESTAGENT_ROOT=/Users/xmq/GitHubRepo/TestAgent AIChecker/.venv/bin/python -m pytest ...
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pytest

from aichecker.vision.checkers.toast import ToastMessageDetector
from aichecker.vision.evaluator import VisionEvaluator, resolve_vlm_config
from aichecker.vision.preprocessor import GuiPreprocessor
from testagent_case_utils import (
    REPO_ROOT,
    collect_case_jsons,
    extract_expected_passed,
    extract_video_frames,
    record_evaluator_token_usage,
    require_testagent_root,
    resolve_path,
    set_checker_report_meta,
)

CHECKER = "toast"
CASE_CATEGORY = "toast"
APP_FROM_NAME = re.compile(r"toast_([a-z0-9]+)")


def _app_from_case_id(case_id: str) -> str:
    match = APP_FROM_NAME.search(case_id)
    return match.group(1) if match else "unknown"


def _truthy_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


def resolve_toast_video(json_path: Path, video_file: str) -> Path:
    """Toast 用例视频通常在 ../normal/ 目录，JSON 里可能只写文件名。"""
    rel = video_file.strip()
    if not rel:
        raise ValueError(f"Empty video_file in JSON {json_path}")

    candidates = [
        resolve_path(json_path, rel),
        (json_path.parent / ".." / "normal" / Path(rel).name).resolve(),
        (json_path.parent / ".." / "abnormal" / Path(rel).name).resolve(),
        (json_path.parent.parent / "normal" / Path(rel).name).resolve(),
        (json_path.parent.parent / "abnormal" / Path(rel).name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_toast_case(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("video_file"):
        data["video_file"] = str(resolve_toast_video(json_path, str(data["video_file"])))
    data.setdefault("sample_interval_sec", 0.5)
    data.setdefault("start_sec", 0.0)
    return data


def _build_toast_detector() -> ToastMessageDetector:
    vlm_config = resolve_vlm_config()
    if not vlm_config.api_key:
        pytest.skip("VLM API key is required for toast regression tests")

    logger = logging.getLogger("test_toast")
    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "toast_regression"
    evaluator = VisionEvaluator(
        api_key=vlm_config.api_key,
        model=vlm_config.model,
        base_url=vlm_config.base_url,
        logger=logger,
        debug=_truthy_env("VGA_DEBUG", "0"),
    )
    preprocessor = GuiPreprocessor(
        artifact_dir=debug_dir / "preprocess",
        logger=logger,
        max_extra_images=4,
    )
    return ToastMessageDetector(
        evaluator=evaluator,
        logger=logger,
        debug=_truthy_env("VGA_DEBUG", "0"),
        preprocessor=preprocessor,
        enable_preprocess=_truthy_env("VGA_ENABLE_PREPROCESS", "1"),
        top_k_candidates=int(os.getenv("VGA_TOAST_TOP_K_CANDIDATES", "3")),
        prompt_version=os.getenv("VGA_TOAST_PROMPT_VERSION", "current"),
    )


@pytest.mark.parametrize("case_json_path", collect_case_jsons(CASE_CATEGORY), ids=lambda p: p.name)
def test_toast_from_testagent_case(case_json_path: Path, request: pytest.FixtureRequest) -> None:
    require_testagent_root()

    case_id = case_json_path.stem
    set_checker_report_meta(
        request,
        checker=CHECKER,
        app=_app_from_case_id(case_id),
        case_id=case_id,
        case_file=str(case_json_path),
    )

    payload = load_toast_case(case_json_path)
    video_path = Path(str(payload.get("video_file", "")))
    if not video_path.exists():
        pytest.skip(f"Video not found: {video_path}")

    expected_passed = extract_expected_passed(payload, case_json_path)
    set_checker_report_meta(request, expected_passed=expected_passed)

    detector = _build_toast_detector()
    with extract_video_frames(
        video_path=video_path,
        prefix=case_id,
        sample_interval_sec=float(payload.get("sample_interval_sec", 0.5)),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
    ) as frames:
        assert len(frames) >= 2, f"Need at least 2 frames for {video_path}"
        result = detector.detect(
            frames,
            task_id=case_id,
            expected_toast_keywords=payload.get("expected_toast_keywords"),
        )

    actual_passed = not bool(result.bug_detected)
    set_checker_report_meta(
        request,
        actual_passed=actual_passed,
        toast_text=result.toast_text,
        expectation_met=result.expectation_met,
        key_frame_timestamp=result.key_frame_timestamp,
    )
    record_evaluator_token_usage(request, detector.evaluator)

    assert actual_passed == expected_passed, (
        f"{case_json_path.name}: expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"bug_detected={result.bug_detected}, expectation_met={result.expectation_met}, "
        f"toast_text={result.toast_text!r}, inferred_expected={result.inferred_expected_toast_text!r}, "
        f"action_semantic={result.action_semantic!r}, reason={result.reason!r}"
    )
