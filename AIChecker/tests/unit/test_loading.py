from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image, ImageDraw

from aichecker.vision.checkers.loading import LoadingDetector


def _write_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (24, 24), color=color).save(path)
    return path


def _write_page_with_dynamic_ad(path: Path, ad_color: tuple[int, int, int]) -> Path:
    image = Image.new("RGB", (100, 100), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 42, 42), fill=ad_color)
    draw.rectangle((56, 72, 90, 84), fill=(40, 40, 40))
    image.save(path)
    return path


def _frame(path: Path, timestamp_sec: float) -> Any:
    return SimpleNamespace(image_path=path, timestamp_sec=timestamp_sec)


def _detector() -> LoadingDetector:
    return LoadingDetector(evaluator=object())


def test_black_white_screen_ignores_single_terminal_black_frame(tmp_path: Path) -> None:
    frames = [
        _frame(_write_image(tmp_path / "frame_0.png", (20, 90, 180)), 0.0),
        _frame(_write_image(tmp_path / "frame_1.png", (40, 120, 210)), 1.0),
        _frame(_write_image(tmp_path / "frame_2.png", (80, 160, 120)), 2.0),
        _frame(_write_image(tmp_path / "frame_3.png", (120, 40, 80)), 3.0),
        _frame(_write_image(tmp_path / "frame_4.png", (0, 0, 0)), 4.0),
    ]

    anomaly_type, reason, metrics = _detector()._detect_black_white_screen(frames)

    assert anomaly_type is None
    assert reason == ""
    assert metrics["black_terminal_run"] == 1
    assert metrics["black_match_count"] == 1


def test_black_white_screen_requires_sustained_terminal_black_frames(tmp_path: Path) -> None:
    frames = [
        _frame(_write_image(tmp_path / "frame_0.png", (20, 90, 180)), 0.0),
        _frame(_write_image(tmp_path / "frame_1.png", (40, 120, 210)), 1.0),
        _frame(_write_image(tmp_path / "frame_2.png", (0, 0, 0)), 2.0),
        _frame(_write_image(tmp_path / "frame_3.png", (0, 0, 0)), 3.0),
        _frame(_write_image(tmp_path / "frame_4.png", (0, 0, 0)), 4.0),
    ]

    anomaly_type, reason, metrics = _detector()._detect_black_white_screen(frames)

    assert anomaly_type == "black_screen"
    assert "持续为近纯黑" in reason
    assert metrics["black_terminal_run"] == 3
    assert metrics["black_window_ratio"] >= 0.6


def test_black_white_screen_requires_sustained_terminal_white_frames(tmp_path: Path) -> None:
    frames = [
        _frame(_write_image(tmp_path / "frame_0.png", (20, 90, 180)), 0.0),
        _frame(_write_image(tmp_path / "frame_1.png", (40, 120, 210)), 1.0),
        _frame(_write_image(tmp_path / "frame_2.png", (255, 255, 255)), 2.0),
        _frame(_write_image(tmp_path / "frame_3.png", (255, 255, 255)), 3.0),
        _frame(_write_image(tmp_path / "frame_4.png", (255, 255, 255)), 4.0),
    ]

    anomaly_type, reason, metrics = _detector()._detect_black_white_screen(frames)

    assert anomaly_type == "white_screen"
    assert "持续为近纯白" in reason
    assert metrics["white_terminal_run"] == 3
    assert metrics["white_window_ratio"] >= 0.6


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "loading"
GARBLED_FIXTURE = FIXTURES_DIR / "garbled_screen_sample.jpg"


def test_garbled_screen_requires_consecutive_corrupted_frames(tmp_path: Path) -> None:
    if not GARBLED_FIXTURE.exists():
        pytest.skip(f"Missing garbled fixture: {GARBLED_FIXTURE}")

    frames = [
        _frame(_write_image(tmp_path / "frame_0.png", (220, 220, 220)), 0.0),
        _frame(GARBLED_FIXTURE, 1.0),
        _frame(GARBLED_FIXTURE, 2.0),
    ]

    anomaly_type, reason, metrics = _detector()._detect_black_white_screen(frames)

    assert anomaly_type == "garbled_screen"
    assert "花屏" in reason
    assert metrics["garbled"]["garbled_max_consecutive"] >= 2


def test_garbled_screen_ignores_single_corrupted_frame(tmp_path: Path) -> None:
    if not GARBLED_FIXTURE.exists():
        pytest.skip(f"Missing garbled fixture: {GARBLED_FIXTURE}")

    frames = [
        _frame(_write_image(tmp_path / "frame_0.png", (220, 220, 220)), 0.0),
        _frame(GARBLED_FIXTURE, 1.0),
        _frame(_write_image(tmp_path / "frame_2.png", (220, 220, 220)), 2.0),
    ]

    anomaly_type, reason, metrics = _detector()._detect_black_white_screen(frames)

    assert anomaly_type is None
    assert reason == ""
    assert metrics["garbled"]["garbled_max_consecutive"] == 1


def test_long_loading_candidate_requires_minimum_duration() -> None:
    detector = _detector()

    short_result, short_reason, short_type, short_metrics = detector._cv_decide(
        [0.02, 0.021, 0.019],
        duration_sec=3.0,
    )
    long_result, long_reason, long_type, long_metrics = detector._cv_decide(
        [0.02, 0.021, 0.019],
        duration_sec=6.0,
    )

    assert short_result is None
    assert short_type == "unknown"
    assert "视频跨度不足" in short_reason
    assert short_metrics["long_loading_candidate"] is True
    assert long_result is True
    assert long_type == "long_loading"
    assert "持续小幅规律变化" in long_reason
    assert long_metrics["duration_sec"] == 6.0


def test_detect_exposes_structured_loading_signals(tmp_path: Path) -> None:
    frames = [
        _frame(_write_image(tmp_path / "frame_0.png", (10, 20, 30)), 0.0),
        _frame(_write_image(tmp_path / "frame_1.png", (210, 80, 40)), 1.0),
        _frame(_write_image(tmp_path / "frame_2.png", (30, 200, 90)), 2.0),
    ]

    result = _detector().detect(frames, task_id="unit_loading")

    assert result.bug_detected is False
    assert result.anomaly_types == ["none"]
    assert set(result.cv_metrics["signals"]) == {
        "no_response",
        "black_white_screen",
        "load_failed",
        "long_loading",
    }
    assert result.cv_metrics["signals"]["no_response"]["detected"] is False
    assert result.cv_metrics["signals"]["black_white_screen"]["detected"] is False
    assert result.cv_metrics["signals"]["load_failed"]["status"] == "delegated_to_load_failure_prompt_detector"
    assert result.cv_metrics["signals"]["long_loading"]["detected"] is False
    assert result.cv_metrics["stages"]["early_frame_indices"] == [0]
    assert result.cv_metrics["stages"]["tail_frame_indices"] == [2]


def test_detect_ignores_persistent_dynamic_noise_for_no_response(tmp_path: Path) -> None:
    colors = [
        (220, 20, 20),
        (20, 220, 20),
        (20, 20, 220),
        (220, 220, 20),
        (220, 20, 220),
        (20, 220, 220),
    ]
    frames = [
        _frame(_write_page_with_dynamic_ad(tmp_path / f"frame_{idx}.png", color), float(idx))
        for idx, color in enumerate(colors)
    ]

    result = _detector().detect(frames, task_id="unit_dynamic_noise")

    assert result.bug_detected is True
    assert result.anomaly_type == "no_response"
    assert result.cv_metrics["motion_noise"]["has_dynamic_noise"] is True
    assert result.cv_metrics["raw_change_stats"]["max"] >= 0.08
    assert result.cv_metrics["max"] <= 0.015
