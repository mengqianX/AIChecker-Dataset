from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from aichecker.vision.motion_noise import MotionNoiseAnalyzer


def _frame(path: Path, timestamp_sec: float) -> Any:
    return SimpleNamespace(image_path=path, timestamp_sec=timestamp_sec)


def _write_frame(path: Path, ad_color: tuple[int, int, int], *, full_screen: bool = False) -> Path:
    image = Image.new("RGB", (100, 100), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    if full_screen:
        draw.rectangle((0, 0, 99, 99), fill=ad_color)
    else:
        draw.rectangle((10, 10, 44, 44), fill=ad_color)
        draw.rectangle((60, 70, 90, 84), fill=(30, 30, 30))
    image.save(path)
    return path


def test_motion_noise_masks_persistent_local_region(tmp_path: Path) -> None:
    colors = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20), (220, 20, 220)]
    frames = [
        _frame(_write_frame(tmp_path / f"frame_{idx}.png", color), float(idx))
        for idx, color in enumerate(colors)
    ]

    result = MotionNoiseAnalyzer().analyze(frames)

    assert result.has_dynamic_noise is True
    assert result.metrics["dynamic_noise_area_ratio"] > 0.05
    assert result.metrics["masked_change_mean"] < result.metrics["raw_change_mean"]
    assert len(result.regions) == 1


def test_motion_noise_rejects_full_screen_motion(tmp_path: Path) -> None:
    colors = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)]
    frames = [
        _frame(_write_frame(tmp_path / f"frame_{idx}.png", color, full_screen=True), float(idx))
        for idx, color in enumerate(colors)
    ]

    result = MotionNoiseAnalyzer().analyze(frames)

    assert result.has_dynamic_noise is False
    assert result.metrics["reason"] == "not_detected"
