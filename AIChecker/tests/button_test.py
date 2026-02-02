    

from doctest import debug
import json
from pathlib import Path

import pytest
from aichecker.checkers import check_button_color
from aichecker.utils import _encode, get_image_color


def test_button_color_pass():
    file="./tests/jsons/sample_button_payload.json"
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    debug_dir = Path("./debug/crops")
    result = check_button_color(data, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed == True

def test_button_color_fail():
    file="./tests/jsons/sample_button_payload.json"
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    data["expected_color"] = "255,0,0"  # 故意设置错误的颜色
    debug_dir = Path("./debug/crops")
    result = check_button_color(data, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed == False

def test_button_color_pass_1():
    file="./tests/jsons/sample_button_payload.json"
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    data["screenshot_b"] = "./tests/screens/button_after1.png"  # 明确指定正确的图片
    data["expected_color"] = "80,113,246"
    data["bounds"] = [130,2577, 200,2647]
    debug_dir = Path("./debug/crops")
    result = check_button_color(data, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed == True

def test_button_color_auto_change_pass():
    file="./tests/jsons/sample_button_payload.json"
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    data["expected_color"] = None    # 启用自动颜色变化检测
    debug_dir = Path("./debug/crops")
    result = check_button_color(data, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed == True

def test_button_color_auto_change_fail():
    file="./tests/jsons/sample_button_payload.json"
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    data["expected_color"] = None    # 启用自动颜色变化检测
    data["screenshot_b"] = "./tests/screens/button_before.png"  # 使用没有变化的图片
    debug_dir = Path("./debug/crops")
    result = check_button_color(data, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed == False

@pytest.mark.skip(reason="暂时跳过")
def test_get_image_color():
    print("Testing get_image_color...")
    print(get_image_color("./tests/screens/button_after.png"))

