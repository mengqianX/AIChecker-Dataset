"""
基于 testcase/count_change 目录的测试用例，验证 check_count_change 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析。
"""
import json
from pathlib import Path

import pytest
from aichecker.checkers import check_count_change
from aichecker.utils import _encode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "count_change"
JSONS_DIR = TESTCASE_DIR / "jsons"


def _resolve_path(json_path: Path, rel_path: str) -> Path:
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()


def _load_and_resolve_payload(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("screenshot_a", "screenshot_b"):
        if data.get(key):
            data[key] = str(_resolve_path(json_path, data[key]))
    return data


def _collect_testcase_jsons():
    cases = []
    sample_path = JSONS_DIR / "sample.json"
    if sample_path.exists():
        cases.append(("sample", "sample", sample_path))
    for subdir in ("Android", "HarmonyOS"):
        dir_path = JSONS_DIR / subdir
        if not dir_path.is_dir():
            continue
        for j in dir_path.glob("*.json"):
            cases.append((subdir, j.stem, j))
    return cases


@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_count_change_from_testcase(platform: str, app_name: str, json_path: Path):
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")

    payload = _load_and_resolve_payload(json_path)

    import os

    backend_override = os.getenv("COUNT_CHANGE_BACKEND")
    if backend_override:
        payload["backend"] = backend_override.strip().lower()
        if payload["backend"] == "ui-tars":
            payload["base_url"] = payload.get("base_url") or os.getenv("UI_TARS_BASE_URL", "http://localhost:8000/v1")
            payload["model"] = payload.get("model") or os.getenv("UI_TARS_MODEL", "ui-tars")
            payload["api_key"] = payload.get("api_key") or os.getenv("UI_TARS_API_KEY") or os.getenv("HF_TOKEN") or "dummy"
        elif payload["backend"] == "qwen":
            if os.getenv("QWEN_MODEL"):
                payload["model"] = os.getenv("QWEN_MODEL")
            if os.getenv("QWEN_BASE_URL"):
                payload["base_url"] = os.getenv("QWEN_BASE_URL")
                payload["model"] = payload.get("model") or os.getenv("QWEN_MODEL") or "Qwen/Qwen2.5-VL-7B-Instruct"
                payload["api_key"] = payload.get("api_key") or os.getenv("QWEN_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

    bounds = payload.get("bounds", [])
    if bounds == [0, 0, 0, 0]:
        pytest.skip(f"Skipping placeholder test case: {json_path} (bounds not set)")
    if not payload.get("screenshot_a") or not payload.get("screenshot_b"):
        pytest.skip(f"Missing screenshots in {json_path}")
    for p in (payload.get("screenshot_a"), payload.get("screenshot_b")):
        if p and not Path(p).exists():
            pytest.skip(f"Screenshot not found: {p}")

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and not payload.get("api_key"):
        pytest.skip(
            f"API key not configured. Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable, "
            f"or add api_key to {json_path}"
        )

    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
    else:
        pytest.fail(
            f"No groundtruth found in {json_path}. "
            f"Test case must include 'expected_passed' (boolean) or 'label' (string) field."
        )

    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "count_change" / f"{platform}_{app_name}"
    try:
        result = check_count_change(payload, debug_dir=debug_dir)
    except ImportError as e:
        pytest.skip(f"{platform}/{app_name}: Missing required dependency. Error: {e}")
    except ValueError as e:
        if "API authentication failed" in str(e):
            pytest.skip(f"{platform}/{app_name}: API authentication failed (invalid API key). Error: {e}")
        raise

    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))

    if result.details.get("is_api_error"):
        error_msg = result.details.get("error", "")
        is_auth_error = result.details.get("is_auth_error", False)
        if is_auth_error:
            pytest.skip(f"{platform}/{app_name}: API authentication failed (invalid API key). Error: {error_msg}")
        pytest.skip(f"{platform}/{app_name}: API call failed. Error: {result.basis}. Details: {error_msg}")

    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: Groundtruth expected_passed={expected_pass}, "
        f"but model detected passed={result.passed}. Model basis: {result.basis}"
    )
