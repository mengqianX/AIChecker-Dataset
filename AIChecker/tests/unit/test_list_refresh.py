from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from aichecker.vision.checkers.count_change import ControlBounds
from aichecker.vision.checkers.list_refresh import ListRefreshDetector
from aichecker.vision.cli import parse_task_from_payload


class _FakeEvaluator:
    def __init__(self, parsed_json: dict[str, Any]) -> None:
        self.parsed_json = parsed_json
        self.calls: list[dict[str, Any]] = []

    def evaluate_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_json=self.parsed_json, raw_response="{}")


def _write_page(path: Path, text: str) -> Path:
    image = Image.new("RGB", (120, 100), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 20, 112, 88), outline=(120, 120, 120), width=1)
    draw.text((16, 44), text, fill=(20, 20, 20))
    image.save(path)
    return path


def test_list_refresh_without_bounds_marks_loading_as_failed(tmp_path: Path) -> None:
    before = _write_page(tmp_path / "before.png", "old list")
    after = _write_page(tmp_path / "after.png", "loading")
    evaluator = _FakeEvaluator(
        {
            "list_refreshed": False,
            "still_loading": True,
            "target_region": "主内容列表区域",
            "expectation_met": False,
            "confidence": 0.91,
            "reason": "后页仍显示加载中，列表刷新未完成。",
        }
    )

    result = ListRefreshDetector(evaluator=evaluator).detect(
        before_image=before,
        after_image=after,
        target_bounds=None,
        expected_list_refresh=True,
        expected_result_text="点击刷新后列表应加载完成并展示新内容",
    )

    assert result.expectation_met is False
    assert result.bug_detected is True
    assert result.list_refreshed is False
    assert result.still_loading is True
    assert result.target_region_box == {"x1": 0, "y1": 10, "x2": 120, "y2": 90}
    assert evaluator.calls[0]["required_fields"]["still_loading"] is bool


def test_list_refresh_uses_bounds_as_target_region(tmp_path: Path) -> None:
    before = _write_page(tmp_path / "before.png", "old list")
    after = _write_page(tmp_path / "after.png", "new list")
    evaluator = _FakeEvaluator(
        {
            "list_refreshed": True,
            "still_loading": False,
            "target_region": "目标列表区域",
            "expectation_met": True,
            "confidence": 0.94,
            "reason": "目标区域内容发生变化。",
        }
    )

    result = ListRefreshDetector(evaluator=evaluator).detect(
        before_image=before,
        after_image=after,
        target_bounds=ControlBounds(x=8, y=20, width=104, height=68),
        expected_list_refresh=True,
        expected_result_text="目标列表区域应刷新",
    )

    assert result.expectation_met is True
    assert result.target_region_box == {"x1": 8, "y1": 20, "x2": 112, "y2": 88}
    prompt = evaluator.calls[0]["user_prompt"]
    assert "目标区域 bounds" in prompt
    assert "x=8" in prompt


def test_parse_list_refresh_accepts_natural_language_expectation_without_bounds() -> None:
    task = parse_task_from_payload(
        {
            "mode": "targeted",
            "task_type": "list_refresh",
            "before_image": "before.png",
            "after_image": "after.png",
            "expected_result": "点击刷新后列表应加载完成并展示新内容",
        },
        base_dir=Path("."),
    )

    assert task.control_bounds is None
    assert task.expected_list_refresh is True
    assert task.expected_result_text == "点击刷新后列表应加载完成并展示新内容"


def test_parse_list_refresh_rejects_video_only_input() -> None:
    try:
        parse_task_from_payload(
            {
                "task_type": "list_refresh",
                "video_file": "list-refresh.mp4",
            },
            base_dir=Path("."),
        )
    except ValueError as exc:
        assert "before_image|screenshot_a" in str(exc)
        assert "after_image|screenshot_b" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("list_refresh should require before/after images")


def test_parse_list_refresh_accepts_target_region_bounds() -> None:
    task = parse_task_from_payload(
        {
            "task_type": "list_refresh",
            "before_image": "before.png",
            "after_image": "after.png",
            "target_region_bounds": [8, 20, 112, 88],
            "expected_result": "目标列表区域应刷新",
        },
        base_dir=Path("."),
    )

    assert task.mode == "targeted"
    assert task.control_bounds is not None
    assert task.control_bounds.x == 8
    assert task.control_bounds.y == 20
    assert task.control_bounds.width == 104
    assert task.control_bounds.height == 68


def test_parse_list_refresh_negative_natural_language_expectation() -> None:
    task = parse_task_from_payload(
        {
            "mode": "targeted",
            "task_type": "list_refresh",
            "before_image": "before.png",
            "after_image": "after.png",
            "expected_result": "点击后列表应保持不变，不发生刷新",
        },
        base_dir=Path("."),
    )

    assert task.expected_list_refresh is False
