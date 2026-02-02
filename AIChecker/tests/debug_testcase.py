#!/usr/bin/env python3
"""
诊断脚本：检查 testcase 测试失败的原因
"""
import json
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aichecker.checkers import check_button_color
from aichecker.utils import button_base_color, dominant_button_color
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "button_color_change"
JSONS_DIR = TESTCASE_DIR / "jsons"

def _resolve_path(json_path: Path, rel_path: str) -> Path:
    return (json_path.parent / rel_path.strip()).resolve()

def test_single_case(json_path: Path):
    """测试单个用例并输出详细信息"""
    print(f"\n{'='*60}")
    print(f"Testing: {json_path.name}")
    print(f"{'='*60}")
    
    data = json.loads(json_path.read_text(encoding="utf-8"))
    
    # 解析路径
    screenshot_a = str(_resolve_path(json_path, data["screenshot_a"]))
    screenshot_b = str(_resolve_path(json_path, data["screenshot_b"]))
    
    print(f"Screenshot A: {screenshot_a}")
    print(f"Screenshot B: {screenshot_b}")
    print(f"Bounds: {data['bounds']}")
    print(f"Expected passed: {data.get('expected_passed', 'N/A')}")
    print(f"Label: {data.get('label', 'N/A')}")
    
    # 检查文件是否存在
    if not Path(screenshot_a).exists():
        print(f"❌ Screenshot A not found!")
        return
    if not Path(screenshot_b).exists():
        print(f"❌ Screenshot B not found!")
        return
    
    # 加载图片
    img_before = Image.open(screenshot_a)
    img_after = Image.open(screenshot_b)
    
    # 裁剪
    bounds = data["bounds"]
    crop_before = img_before.crop(bounds)
    crop_after = img_after.crop(bounds)
    
    print(f"\nCrop size: {crop_before.size}")
    
    # 测试新方法
    try:
        base_before = button_base_color(crop_before)
        base_after = button_base_color(crop_after)
        base_diff = tuple(abs(base_after[i] - base_before[i]) for i in range(3))
        base_max_diff = max(base_diff)
        print(f"\n✅ button_base_color:")
        print(f"   Before: {base_before}")
        print(f"   After:  {base_after}")
        print(f"   Diff:   {base_diff}, max={base_max_diff}")
    except Exception as e:
        print(f"❌ button_base_color failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试旧方法（对比）
    try:
        dom_before = dominant_button_color(crop_before)
        dom_after = dominant_button_color(crop_after)
        dom_diff = tuple(abs(dom_after[i] - dom_before[i]) for i in range(3))
        dom_max_diff = max(dom_diff)
        print(f"\n📊 dominant_button_color (old method):")
        print(f"   Before: {dom_before}")
        print(f"   After:  {dom_after}")
        print(f"   Diff:   {dom_diff}, max={dom_max_diff}")
    except Exception as e:
        print(f"❌ dominant_button_color failed: {e}")
    
    # 运行完整检测
    payload = {
        "screenshot_a": screenshot_a,
        "screenshot_b": screenshot_b,
        "bounds": bounds,
        "expected_color": None if data.get("expected_color") == "" else data.get("expected_color"),
        "tolerance": data.get("tolerance", 20),
    }
    
    try:
        result = check_button_color(payload)
        print(f"\n🔍 check_button_color result:")
        print(f"   Passed: {result.passed}")
        print(f"   Basis:  {result.basis}")
        print(f"   Details: {result.details}")
        
        expected_pass = data.get("expected_passed", data.get("label") == "pass")
        if result.passed != expected_pass:
            print(f"\n❌ MISMATCH!")
            print(f"   Expected: {expected_pass}")
            print(f"   Got:      {result.passed}")
        else:
            print(f"\n✅ Match expected result")
    except Exception as e:
        print(f"❌ check_button_color failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试所有用例
    for subdir in ("Android", "HarmonyOS"):
        dir_path = JSONS_DIR / subdir
        if not dir_path.is_dir():
            continue
        for json_file in sorted(dir_path.glob("*.json")):
            test_single_case(json_file)
