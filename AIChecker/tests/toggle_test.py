import json
from pathlib import Path

import pytest
from aichecker.checkers import check_toggle
from aichecker.utils import _encode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "toggle_state"
JSONS_DIR = TESTCASE_DIR / "jsons"

def _resolve_path(json_path: Path, rel_path: str) -> Path:
    """将 JSON 中的相对路径解析为绝对路径（相对于该 JSON 所在目录）。"""
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()

def _load_and_resolve_payload(json_path: Path) -> dict:
    """加载 JSON 并解析其中的 screenshot 路径。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("screenshot_a", "screenshot_b"):
        if data.get(key):
            data[key] = str(_resolve_path(json_path, data[key]))
    return data

def _collect_testcase_jsons():
    """
    收集测试 JSON。
    优先使用 jsons/Android、jsons/HarmonyOS；否则回退到 testcase/toggle_state/*.json
    跳过名字包含 sample 的 JSON。
    """
    cases = []
    for subdir in ("Android", "HarmonyOS"):
        dir_path = JSONS_DIR / subdir
        if not dir_path.is_dir():
            continue
        for j in dir_path.glob("*.json"):
            if "sample" in j.stem.lower():
                continue
            cases.append((subdir, j.stem, j))
    return cases

@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_toggle_state_from_testcase(platform: str, app_name: str, json_path: Path):
    """使用 testcase/toggle_state 中各应用的 JSON 跑开关状态检测。"""
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")

    payload = _load_and_resolve_payload(json_path)

    # UI Tree 可用时无需截图；CV 分支必须有截图
    if not payload.get("ui_tree"):
        if not payload.get("screenshot_a") or not payload.get("screenshot_b"):
            pytest.skip(f"Missing screenshots in {json_path}")
        for p in (payload.get("screenshot_a"), payload.get("screenshot_b")):
            if p and not Path(p).exists():
                pytest.skip(f"Screenshot not found: {p}")

    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "crops" / f"{platform}_{app_name}"
    result = check_toggle(payload, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))

    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
        groundtruth_source = f"expected_passed={expected_pass}"
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
        groundtruth_source = f"label={payload.get('label')}"
    else:
        pytest.skip(f"No groundtruth found in {json_path} (missing 'expected_passed' or 'label')")

    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: expected passed={expected_pass} ({groundtruth_source}), "
        f"got passed={result.passed}, basis={result.basis}"
    )
