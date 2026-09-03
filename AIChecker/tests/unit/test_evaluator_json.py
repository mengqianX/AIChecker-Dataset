from __future__ import annotations

import pytest

from aichecker.vision.evaluator import VisionEvaluator
from aichecker.vision.vlm_json import recover_vlm_json


def test_plain_json_does_not_need_repair() -> None:
    recovered = recover_vlm_json('{"toast_visible": true, "reason": "ok"}')
    assert recovered.payload["toast_visible"] is True
    assert recovered.repaired is False
    assert recovered.method == "raw"


def test_markdown_fence_is_rewrapped() -> None:
    raw = """```json
{
  "toast_visible": true,
  "toast_text": "1 已回收 撤销",
  "action_semantic": "选择标签",
  "expectation_met": false
}
```"""
    recovered = recover_vlm_json(raw)
    assert recovered.repaired is True
    assert recovered.method == "fence"
    assert recovered.payload["action_semantic"] == "选择标签"
    assert recovered.payload["expectation_met"] is False


def test_prose_wrapped_object_is_extracted() -> None:
    recovered = recover_vlm_json('分析如下：\n{"bug_detected": false, "reason": "无异常"}')
    assert recovered.repaired is True
    assert recovered.payload["bug_detected"] is False


def test_trailing_comma_is_rewrapped() -> None:
    recovered = recover_vlm_json('{"reason": "ok", "bug_detected": false,}')
    assert recovered.payload["bug_detected"] is False
    assert recovered.repaired is True
    assert "trailing_comma" in recovered.method


def test_bool_string_is_coerced_to_schema() -> None:
    recovered = recover_vlm_json(
        '{"toast_visible": "true", "reason": 1}',
        required_fields={"toast_visible": bool, "reason": str},
    )
    assert recovered.payload["toast_visible"] is True
    assert recovered.payload["reason"] == "1"
    assert recovered.repaired is True


def test_optional_null_string_coerces() -> None:
    recovered = recover_vlm_json(
        '{"expectation_met": "null"}',
        required_fields={"expectation_met": (bool, type(None))},
    )
    assert recovered.payload["expectation_met"] is None


def test_parse_empty_output_raises() -> None:
    with pytest.raises(ValueError, match="为空"):
        recover_vlm_json("   ")


def test_parse_invalid_json_still_raises() -> None:
    with pytest.raises(ValueError, match="不是合法 JSON"):
        recover_vlm_json("```json\nnot json\n```")


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValueError, match="缺少必需字段"):
        recover_vlm_json('{"reason": "ok"}', required_fields={"bug_detected": bool})


def test_evaluator_wrapper_entry_still_works() -> None:
    parsed = VisionEvaluator._parse_json_object('```json\n{"ok": true}\n```')
    assert parsed == {"ok": True}
