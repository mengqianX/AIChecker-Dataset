from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aichecker.vision.baseline_health import VideoBaselineHealthOrchestrator
from aichecker.vision.checkers.load_failure import LoadFailurePromptDetector
from aichecker.vision.checkers.loading import LoadingDetector
from aichecker.vision.cli import parse_task_from_payload


class _NoVlmEvaluator:
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


def _frame(path: Path, timestamp_sec: float) -> Any:
    return SimpleNamespace(image_path=path, timestamp_sec=timestamp_sec)


def test_parse_video_baseline_defaults_to_targeted_mode() -> None:
    task = parse_task_from_payload(
        {
            "task_type": "video_baseline",
            "video_file": "demo.mp4",
        },
        base_dir=Path("."),
    )

    assert task.mode == "targeted"
    assert task.task_type == "video_baseline"
    assert task.video_file == "demo.mp4"


def test_video_baseline_orchestrator_returns_four_independent_checks(tmp_path: Path) -> None:
    frames = [
        _frame(tmp_path / "f0.png", 0.0),
        _frame(tmp_path / "f1.png", 1.0),
    ]
    for idx, frame in enumerate(frames):
        from PIL import Image

        Image.new("RGB", (40, 40), color=(20 + idx * 10, 20, 20)).save(frame.image_path)

    orchestrator = VideoBaselineHealthOrchestrator(
        loading_detector=LoadingDetector(evaluator=_NoVlmEvaluator()),
        load_failure_detector=LoadFailurePromptDetector(evaluator=_NoVlmEvaluator()),
    )
    segments, summary, video_bug_detected, task_intent = orchestrator.run(
        task_id="baseline_unit",
        sampled_frames=frames,
    )

    check_ids = [item["check_id"] for item in segments]
    assert check_ids == [
        "no_response",
        "black_white_screen",
        "long_loading",
        "load_failure",
    ]
    assert len(summary["checks"]) == 4
    assert "四项基础检测" in task_intent
    assert isinstance(video_bug_detected, bool)
    assert all("bug_detected" in item for item in segments)
