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


def test_transient_button_pulse_is_noise_not_clear_progress(tmp_path: Path) -> None:
    """中间短脉冲大跳变后回到原状，应视为噪音并判无响应，不能仅凭 max 判有响应。"""
    base = (30, 30, 30)
    flash = (220, 220, 220)
    frames = []
    for idx in range(12):
        color = flash if idx in {6, 7} else base
        frames.append(_frame(_write_image(tmp_path / f"frame_{idx}.png", color), float(idx)))

    result = _detector().detect(frames, task_id="transient_pulse")
    nr = result.cv_metrics["signals"]["no_response"]

    assert result.cv_metrics["stable_baseline_progress"] is False
    assert result.cv_metrics.get("transient_spike_filter", {}).get("applied") is True
    assert nr["detected"] is True
    assert "no_response" in result.anomaly_types
    assert result.decision_source != "cv_clear_progress"


def test_stable_state_change_is_clear_progress(tmp_path: Path) -> None:
    """相对起始帧切换到新状态并短时保持，应视为真正响应（静止时长不超过 long_loading 阈值）。"""
    before = (30, 30, 30)
    after = (220, 80, 40)
    frames = []
    # 总时长约 3.5s，切换后静止不足 5s，避免被低变化长跑误判为 long_loading。
    for idx in range(8):
        color = before if idx < 2 else after
        frames.append(_frame(_write_image(tmp_path / f"frame_{idx}.png", color), float(idx) * 0.5))

    result = _detector().detect(frames, task_id="stable_toggle")
    nr = result.cv_metrics["signals"]["no_response"]

    assert result.cv_metrics["stable_baseline_progress"] is True
    assert result.bug_detected is False
    assert nr["detected"] is False
    assert result.anomaly_types == ["none"]
    assert "稳定变化" in result.reason or result.decision_source == "cv_clear_progress"


def test_stable_baseline_with_long_low_run_is_long_loading(tmp_path: Path) -> None:
    """切换到新外观后长时间低变化停留，即使相对起始态已稳定变化，也应判 long_loading。"""
    before = (30, 30, 30)
    after = (220, 80, 40)
    frames = []
    for idx in range(12):
        color = before if idx < 2 else after
        frames.append(_frame(_write_image(tmp_path / f"frame_{idx}.png", color), float(idx)))

    result = _detector().detect(frames, task_id="stable_then_stuck", enable_vlm_fallback=False)
    ll = result.cv_metrics["signals"]["long_loading"]

    assert result.cv_metrics["stable_baseline_progress"] is True
    assert ll["detected"] is True
    assert "long_loading" in result.anomaly_types
    assert result.decision_source != "cv_clear_progress"


def test_long_loading_does_not_auto_promote_to_no_response(tmp_path: Path) -> None:
    """long_loading 不应被 CV 强行提升为 no_response（避免正常点击后局部静止误报）。"""
    frames = [
        _frame(_write_image(tmp_path / f"frame_{idx}.png", (30 + idx, 30, 30)), float(idx))
        for idx in range(8)
    ]
    detector = _detector()

    def _force_long_loading(self, ratios, duration_sec=None, **_kwargs):  # noqa: ANN001
        metrics = {
            "frame_change_ratios": list(ratios),
            **self._build_stats(ratios),
            "duration_sec": duration_sec,
            "min_long_loading_duration_sec": self.cv_min_long_loading_duration_sec,
        }
        return True, "CV判定长时间加载", "long_loading", metrics

    original = LoadingDetector._cv_decide
    LoadingDetector._cv_decide = _force_long_loading  # type: ignore[method-assign]
    try:
        result = detector.detect(frames, task_id="ll_no_promote", enable_vlm_fallback=False)
    finally:
        LoadingDetector._cv_decide = original  # type: ignore[method-assign]

    nr = result.cv_metrics["signals"]["no_response"]
    ll = result.cv_metrics["signals"]["long_loading"]
    assert result.anomaly_type == "long_loading"
    assert ll["detected"] is True
    assert nr["detected"] is False
    assert result.cv_metrics.get("vlm_fallback", {}).get("needed") is True
    assert result.cv_metrics.get("vlm_fallback", {}).get("skipped") is True


def test_no_response_hit_is_inherited_by_long_loading_signal(tmp_path: Path) -> None:
    frames = [
        _frame(_write_image(tmp_path / f"frame_{idx}.png", (30, 30, 30)), float(idx))
        for idx in range(8)
    ]

    result = _detector().detect(frames, task_id="inherit_nr")
    nr = result.cv_metrics["signals"]["no_response"]
    ll = result.cv_metrics["signals"]["long_loading"]

    assert result.anomaly_type == "no_response"
    assert nr["detected"] is True
    assert ll["detected"] is True
    assert ll.get("source") == "inherited_from_no_response" or ll["evidence"].get("inherited_from") == "no_response"


def test_vlm_uncertain_updates_no_response_signal(tmp_path: Path) -> None:
    """CV 不确定时，VLM 若判 no_response，应回写信号。"""

    class _VlmNoResponse:
        def evaluate_json(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                parsed_json={
                    "bug_detected": True,
                    "reason": "页面长时间无反馈",
                    "decision_basis": "首尾帧几乎不变",
                    "anomaly_type": "no_response",
                },
                raw_response='{"bug_detected":true,"anomaly_type":"no_response"}',
            )

    frames = [
        _frame(_write_image(tmp_path / f"frame_{idx}.png", (30 + idx * 12, 40, 80 - idx * 5)), float(idx))
        for idx in range(4)
    ]
    detector = LoadingDetector(evaluator=_VlmNoResponse())

    def _force_uncertain(self, ratios, duration_sec=None, **_kwargs):  # noqa: ANN001
        metrics = {
            "frame_change_ratios": list(ratios),
            **self._build_stats(ratios),
            "duration_sec": duration_sec,
            "min_long_loading_duration_sec": self.cv_min_long_loading_duration_sec,
        }
        return None, "CV判定不确定：变化特征介于静止与明显进展之间，转交VLM语义判定。", "unknown", metrics

    original = LoadingDetector._cv_decide
    LoadingDetector._cv_decide = _force_uncertain  # type: ignore[method-assign]
    try:
        result = detector.detect(frames, task_id="vlm_uncertain_nr")
    finally:
        LoadingDetector._cv_decide = original  # type: ignore[method-assign]

    assert result.cv_metrics["vlm_fallback"]["needed"] is True
    assert result.cv_metrics["vlm_fallback"]["trigger"] == "cv_uncertain"
    assert result.cv_metrics["signals"]["no_response"]["detected"] is True
    assert result.cv_metrics["signals"]["no_response"]["source"] == "vlm_fallback"
    assert "no_response" in result.anomaly_types



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
