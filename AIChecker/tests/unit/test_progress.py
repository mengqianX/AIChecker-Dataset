from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from aichecker.checkers import check_progress_change



def test_progress_change_real_baidudisk_case():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    before = repo_root / "testcase" / "progress_bar_change" / "screens" / "baidudisk" / "baidudisk_before_1.png"
    after = repo_root / "testcase" / "progress_bar_change" / "screens" / "baidudisk" / "baidudisk_after_1.png"
    if not before.exists() or not after.exists():
        pytest.skip("baidudisk screenshots not found")

    payload = {
        "screenshot_a": str(before),
        "screenshot_b": str(after),
        "bounds": [120, 1880, 960, 2015],
        "expected_change": True,
        "change_ratio_threshold": 0.01,
    }
    result = check_progress_change(payload)
    assert result.passed is True
