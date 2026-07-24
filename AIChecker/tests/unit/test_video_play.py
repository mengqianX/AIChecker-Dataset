from __future__ import annotations

from pathlib import Path

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
