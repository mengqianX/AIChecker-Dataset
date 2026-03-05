#!/usr/bin/env python3
"""
诊断脚本：检查 button testcase 失败原因
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from aichecker.checkers import check_button_color
from aichecker.utils import button_base_color, dominant_button_color


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "button_color_change"
JSONS_DIR = TESTCASE_DIR / "jsons"


def _resolve_path(json_path: Path, rel_path: str) -> Path:
    return (json_path.parent / rel_path.strip()).resolve()


def test_single_case(json_path: Path):
    print(f"\n{'='*60}\nTesting: {json_path.name}\n{'='*60}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    screenshot_a = str(_resolve_path(json_path, data["screenshot_a"]))
    screenshot_b = str(_resolve_path(json_path, data["screenshot_b"]))
    if not Path(screenshot_a).exists() or not Path(screenshot_b).exists():
        print("❌ screenshot missing")
        return

    img_before = Image.open(screenshot_a)
    img_after = Image.open(screenshot_b)
    bounds = data["bounds"]
    crop_before = img_before.crop(bounds)
    crop_after = img_after.crop(bounds)
    print(f"Crop size: {crop_before.size}")

    try:
        base_before = button_base_color(crop_before)
        base_after = button_base_color(crop_after)
        print(f"button_base_color: before={base_before}, after={base_after}")
    except Exception as e:
        print(f"button_base_color failed: {e}")

    try:
        dom_before = dominant_button_color(crop_before)
        dom_after = dominant_button_color(crop_after)
        print(f"dominant_button_color: before={dom_before}, after={dom_after}")
    except Exception as e:
        print(f"dominant_button_color failed: {e}")

    payload = {
        "screenshot_a": screenshot_a,
        "screenshot_b": screenshot_b,
        "bounds": bounds,
        "expected_color": None if data.get("expected_color") == "" else data.get("expected_color"),
        "tolerance": data.get("tolerance", 20),
    }
    result = check_button_color(payload)
    print(f"passed={result.passed}\nbasis={result.basis}\ndetails={result.details}")


if __name__ == "__main__":
    for json_file in sorted(JSONS_DIR.rglob("*.json")):
        if "sample" in json_file.stem.lower():
            continue
        test_single_case(json_file)
