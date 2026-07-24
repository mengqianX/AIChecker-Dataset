"""TestAgent testcase 回归测试的共享工具。"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from aichecker.vision.perception import ExtractedFrame, FrameExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROJECT_ROOT = REPO_ROOT / "AIChecker"
TESTAGENT_ROOT = Path(os.getenv("TESTAGENT_ROOT", str(REPO_ROOT.parent / "TestAgent"))).resolve()


class _NoVlmEvaluator:
    """避免 loading 检测在 CV 不确定时触发真实 VLM 调用。"""

    def evaluate_json(self, **_kwargs: Any) -> Any:
        class _Result:
            parsed_json = {
                "bug_detected": False,
                "reason": "mock_no_vlm",
                "decision_basis": "mock_no_vlm",
                "anomaly_type": "none",
                "load_failed": False,
                "failure_type": "none",
                "evidence_text": "",
                "confidence": 0.0,
            }
            raw_response = "{}"

        return _Result()


def set_checker_report_meta(request: pytest.FixtureRequest, **kwargs: Any) -> None:
    current = getattr(request.node, "_checker_report_meta", {})
    current.update(kwargs)
    request.node._checker_report_meta = current


def testcase_json_dir(category: str) -> Path:
    return TESTAGENT_ROOT / "testcase" / category / "json"


def resolve_path(json_path: Path, rel_path: str) -> Path:
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()


def load_case(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("video_file"):
        data["video_file"] = str(resolve_path(json_path, str(data["video_file"])))
    data.setdefault("sample_interval_sec", 1.0)
    data.setdefault("start_sec", 0.0)
    return data


def collect_case_jsons(category: str) -> list[Path]:
    json_dir = testcase_json_dir(category)
    if not json_dir.exists():
        return []
    return sorted(json_dir.glob("*.json"), key=lambda p: p.name)


def extract_expected_passed(payload: dict[str, Any], json_path: Path) -> bool:
    if "expected_passed" in payload:
        return bool(payload["expected_passed"])
    raise AssertionError(
        f"No groundtruth found in {json_path}. "
        "Test case must include 'expected_passed' (boolean) field."
    )


def get_frame_extractor() -> FrameExtractor:
    auto_crop = os.getenv("VGA_AUTO_CROP_BLACK_BORDERS", "0").strip() in {
        "1",
        "true",
        "True",
        "YES",
        "yes",
    }
    return FrameExtractor(auto_crop_black_borders=auto_crop)


@contextmanager
def extract_video_frames(
    *,
    video_path: Path,
    prefix: str,
    sample_interval_sec: float,
    start_sec: float,
    end_sec: float | None = None,
) -> Iterator[list[ExtractedFrame]]:
    extractor = get_frame_extractor()
    with tempfile.TemporaryDirectory(prefix="testagent_frames_") as tmp:
        frames = extractor.extract_frames_by_interval(
            video_path=video_path,
            output_dir=Path(tmp),
            prefix=prefix,
            interval_sec=sample_interval_sec,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        yield frames


def require_testagent_root() -> None:
    if not TESTAGENT_ROOT.exists():
        pytest.skip(f"TestAgent root not found: {TESTAGENT_ROOT}")


def no_vlm_evaluator() -> _NoVlmEvaluator:
    return _NoVlmEvaluator()
