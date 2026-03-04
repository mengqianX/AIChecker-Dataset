import json
from pathlib import Path

import pytest
from aichecker.checkers import check_button_color
from aichecker.utils import _encode, get_image_color


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_PAYLOAD = REPO_ROOT / "AIChecker" / "tests" / "fixtures" / "button" / "sample_button_payload.json"
DEBUG_DIR = REPO_ROOT / "AIChecker" / "debug" / "crops"


def _load_payload() -> dict:
    data = json.loads(SAMPLE_PAYLOAD.read_text(encoding="utf-8"))
    base_dir = SAMPLE_PAYLOAD.parent
    for key in ("screenshot_a", "screenshot_b"):
        raw = data.get(key)
        if raw:
            if str(raw).startswith("./tests/"):
                data[key] = str((REPO_ROOT / "AIChecker" / str(raw).lstrip("./")).resolve())
            else:
                data[key] = str((base_dir / raw).resolve())
    return data


def test_button_color_pass():
    data = _load_payload()
    result = check_button_color(data, debug_dir=DEBUG_DIR)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is True


def test_button_color_fail():
    data = _load_payload()
    data["expected_color"] = "255,0,0"
    result = check_button_color(data, debug_dir=DEBUG_DIR)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is False


def test_button_color_pass_1():
    data = _load_payload()
    data["screenshot_b"] = str(REPO_ROOT / "AIChecker" / "tests" / "screens" / "button_after1.png")
    data["expected_color"] = "80,113,246"
    data["bounds"] = [130, 2577, 200, 2647]
    result = check_button_color(data, debug_dir=DEBUG_DIR)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is True


def test_button_color_auto_change_pass():
    data = _load_payload()
    data["expected_color"] = None
    result = check_button_color(data, debug_dir=DEBUG_DIR)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is True


def test_button_color_auto_change_fail():
    data = _load_payload()
    data["expected_color"] = None
    data["screenshot_b"] = str(REPO_ROOT / "AIChecker" / "tests" / "screens" / "button_before.png")
    result = check_button_color(data, debug_dir=DEBUG_DIR)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is False


@pytest.mark.skip(reason="暂时跳过")
def test_get_image_color():
    print("Testing get_image_color...")
    print(get_image_color(str(REPO_ROOT / "AIChecker" / "tests" / "screens" / "button_after.png")))
