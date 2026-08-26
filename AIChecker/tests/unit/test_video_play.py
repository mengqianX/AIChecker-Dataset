from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aichecker.vision.checkers.video_play import VideoPlayDetector
from aichecker.vision.cli import parse_task_from_payload


def test_parse_video_play_accepts_unified_video_input() -> None:
    task = parse_task_from_payload(
        {
            "task_id": "video_play_demo",
            "mode": "targeted",
            "task_type": "video_play",
            "video_file": "play.mp4",
            "sample_interval_sec": 1.0,
            "start_sec": 0.0,
            "play_timestamp_sec": 2.5,
        },
        base_dir=Path("."),
    )

    assert task.task_id == "video_play_demo"
    assert task.mode == "targeted"
    assert task.task_type == "video_play"
    assert task.video_file == "play.mp4"
    assert task.sample_interval_sec == 1.0
    assert task.start_sec == 0.0
    assert task.play_timestamp_sec == 2.5


def test_parse_video_play_defaults_to_targeted_mode() -> None:
    task = parse_task_from_payload(
        {
            "task_type": "video_play",
            "video_file": "play.mp4",
        },
        base_dir=Path("."),
    )

    assert task.mode == "targeted"
    assert task.task_type == "video_play"


class _Frame:
    def __init__(self, image_path: Path, timestamp_sec: float) -> None:
        self.image_path = str(image_path)
        self.timestamp_sec = timestamp_sec


def _write_scene(path: Path, *, with_center_play: bool, noise: int = 0) -> None:
    height, width = 640, 360
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[80:420, 60:300] = (90, 90, 90)
    if with_center_play:
        cv2.circle(image, (width // 2, height // 2), 48, (220, 220, 220), -1)
        triangle = np.array(
            [
                (width // 2 - 16, height // 2 - 24),
                (width // 2 - 16, height // 2 + 24),
                (width // 2 + 28, height // 2),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, triangle, (40, 40, 40))
    if noise:
        rng = np.random.default_rng(noise)
        image = np.clip(image.astype(np.int16) + rng.integers(-3, 4, image.shape), 0, 255).astype(
            np.uint8
        )
    cv2.imwrite(str(path), image)


def test_center_play_then_freeze_is_detected_as_bug(tmp_path: Path) -> None:
    """中心播放按钮消失后画面卡住，应判定播放失败。"""
    frames: list[_Frame] = []
    for idx, (ts, with_play) in enumerate(
        [
            (0.0, True),
            (1.0, True),
            (2.0, False),
            (3.0, False),
            (4.0, False),
            (5.0, False),
        ]
    ):
        path = tmp_path / f"f{idx}.png"
        _write_scene(path, with_center_play=with_play)
        frames.append(_Frame(path, ts))

    result = VideoPlayDetector().detect(frames, task_id="center_play_freeze")
    assert result.play_action_detected is True
    assert result.play_timestamp_sec == 2.0
    assert result.bug_detected is True


def test_single_ui_spike_without_sustained_motion_is_not_passed(tmp_path: Path) -> None:
    """仅一次 UI 尖峰、后续全静止时，全片兜底不应判为播放正常。"""
    frames: list[_Frame] = []
    for idx, (ts, with_play) in enumerate(
        [
            (0.0, True),
            (1.0, False),
            (2.0, False),
            (3.0, False),
            (4.0, False),
        ]
    ):
        path = tmp_path / f"f{idx}.png"
        _write_scene(path, with_center_play=with_play)
        frames.append(_Frame(path, ts))

    detector = VideoPlayDetector(play_content_change_threshold=1.0)  # force miss play locate
    result = detector.detect(frames, task_id="ui_spike_only")
    assert result.play_action_detected is False
    assert result.bug_detected is True
    assert result.decision_source in {"cv_play_locator", "cv_whole_video_motion"}
