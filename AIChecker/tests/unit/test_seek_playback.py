from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from aichecker.vision.checkers.seek_playback import SeekPlaybackDetector
from aichecker.vision.cli import parse_task_from_payload


def _write_frame(
    path: Path,
    *,
    progress_ratio: float,
    content_x: int = 16,
    content_color: tuple[int, int, int] = (80, 150, 220),
) -> Path:
    image = Image.new("RGB", (80, 80), color=(24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle((content_x, 18, content_x + 18, 36), fill=content_color)
    draw.rectangle((8, 70, 72, 73), fill=(80, 80, 80))
    draw.rectangle((8, 70, 8 + int(64 * progress_ratio), 73), fill=(230, 230, 230))
    image.save(path)
    return path


def _frame(path: Path, timestamp_sec: float) -> Any:
    return SimpleNamespace(image_path=path, timestamp_sec=timestamp_sec)


def test_seek_playback_auto_detects_seek_and_passes_when_content_moves(tmp_path: Path) -> None:
    frames = [
        _frame(_write_frame(tmp_path / "f0.png", progress_ratio=0.15, content_x=16), 0.0),
        _frame(_write_frame(tmp_path / "f1.png", progress_ratio=0.15, content_x=16), 1.0),
        _frame(_write_frame(tmp_path / "f2.png", progress_ratio=0.75, content_x=16), 2.0),
        _frame(_write_frame(tmp_path / "f3.png", progress_ratio=0.76, content_x=18), 3.0),
        _frame(_write_frame(tmp_path / "f4.png", progress_ratio=0.77, content_x=24), 4.0),
        _frame(_write_frame(tmp_path / "f5.png", progress_ratio=0.78, content_x=30), 5.0),
    ]

    result = SeekPlaybackDetector().detect(frames, task_id="unit_seek")

    assert result.result is True
    assert result.bug_detected is False
    assert result.seek_detected is True
    assert result.seek_source == "auto_cv"
    assert result.seek_timestamp_sec == 2.0
    assert result.anomaly_type == "none"
    assert "恢复变化" in result.decision_basis


def test_seek_playback_fails_when_post_seek_content_is_static(tmp_path: Path) -> None:
    frames = [
        _frame(_write_frame(tmp_path / "f0.png", progress_ratio=0.15, content_x=16), 0.0),
        _frame(_write_frame(tmp_path / "f1.png", progress_ratio=0.15, content_x=16), 1.0),
        _frame(_write_frame(tmp_path / "f2.png", progress_ratio=0.75, content_x=16), 2.0),
        _frame(_write_frame(tmp_path / "f3.png", progress_ratio=0.75, content_x=16), 3.0),
        _frame(_write_frame(tmp_path / "f4.png", progress_ratio=0.75, content_x=16), 4.0),
        _frame(_write_frame(tmp_path / "f5.png", progress_ratio=0.75, content_x=16), 5.0),
        _frame(_write_frame(tmp_path / "f6.png", progress_ratio=0.75, content_x=16), 6.0),
        _frame(_write_frame(tmp_path / "f7.png", progress_ratio=0.75, content_x=16), 7.0),
        _frame(_write_frame(tmp_path / "f8.png", progress_ratio=0.75, content_x=16), 8.0),
    ]

    result = SeekPlaybackDetector().detect(frames, task_id="unit_seek")

    assert result.result is False
    assert result.bug_detected is True
    assert result.seek_detected is True
    assert result.anomaly_type == "no_response"
    assert "长期近乎静止" in result.decision_basis


def test_seek_playback_accepts_optional_seek_timestamp(tmp_path: Path) -> None:
    frames = [
        _frame(_write_frame(tmp_path / "f0.png", progress_ratio=0.15, content_x=16), 0.0),
        _frame(_write_frame(tmp_path / "f1.png", progress_ratio=0.15, content_x=16), 1.0),
        _frame(_write_frame(tmp_path / "f2.png", progress_ratio=0.15, content_x=18), 2.0),
        _frame(_write_frame(tmp_path / "f3.png", progress_ratio=0.16, content_x=24), 3.0),
        _frame(_write_frame(tmp_path / "f4.png", progress_ratio=0.17, content_x=30), 4.0),
    ]

    result = SeekPlaybackDetector().detect(
        frames,
        task_id="unit_seek",
        seek_timestamp_sec=1.0,
    )

    assert result.result is True
    assert result.seek_source == "input"
    assert result.seek_timestamp_sec == 1.0


def test_parse_seek_playback_accepts_unified_video_input() -> None:
    task = parse_task_from_payload(
        {
            "task_id": "seek_demo",
            "mode": "targeted",
            "task_type": "video_seek_playback",
            "video_file": "seek.mp4",
            "sample_interval_sec": 0.5,
            "start_sec": 0.0,
        },
        base_dir=Path("."),
    )

    assert task.task_id == "seek_demo"
    assert task.mode == "targeted"
    assert task.task_type == "video_seek_playback"
    assert task.video_file == "seek.mp4"
    assert task.sample_interval_sec == 0.5
    assert task.start_sec == 0.0
