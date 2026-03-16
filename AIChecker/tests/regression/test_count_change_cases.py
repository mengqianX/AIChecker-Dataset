"""
基于 testcase/count_change 目录的测试用例，验证 check_count_change 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析。
"""
import json
from pathlib import Path
from pyexpat import model

import pytest
from aichecker.checkers import check_count_change
from aichecker.utils import _encode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "count_change"
JSONS_DIR = TESTCASE_DIR / "jsons"


def _set_checker_report_meta(request: pytest.FixtureRequest, **kwargs) -> None:
    current = getattr(request.node, "_checker_report_meta", {})
    current.update(kwargs)
    request.node._checker_report_meta = current


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
def test_count_change_from_testcase(
    platform: str, app_name: str, json_path: Path, request: pytest.FixtureRequest
):
    _set_checker_report_meta(
        request,
        checker="count_change",
        app=platform,
        case_id=app_name,
        case_file=str(json_path),
    )
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")

    payload = _load_and_resolve_payload(json_path)
    _set_checker_report_meta(
        request,
        template_image=Path(payload.get("screenshot_a", "N/A")).name,
        target_image=Path(payload.get("screenshot_b", "N/A")).name,
    )

    import os

    backend = payload.get("backend") or os.getenv("COUNT_CHANGE_BACKEND") or "qwen"
    if backend == "ui-tars":
        base_url = payload.get("base_url") or os.getenv("UI_TARS_BASE_URL", "https://zcammjkko6k15eg7.us-east-1.aws.endpoints.huggingface.cloud/v1")
        model = payload.get("model") or os.getenv("UI_TARS_MODEL") or "ByteDance-Seed/UI-TARS-1.5-7B"
        api_key = payload.get("api_key") or os.getenv("UI_TARS_API_KEY")
    elif backend == "qwen":
        base_url = payload.get("base_url") or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = payload.get("model") or os.getenv("QWEN_MODEL") or "qwen-vl-max"
        api_key = payload.get("api_key") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    elif backend == "mai-ui":
        base_url = payload.get("base_url") or os.getenv("MAI_UI_BASE_URL") or "https://bgkgog9p754r0x2j.us-east-1.aws.endpoints.huggingface.cloud/v1"
        model = payload.get("model") or os.getenv("MAI_UI_MODEL") or "Tongyi-MAI/MAI-UI-8B"
        api_key = payload.get("api_key") or os.getenv("MAI_UI_API_KEY") or os.getenv("HF_TOKEN") 
    else:
        pytest.skip(f"Unsupported backend: {backend}. Supported: qwen, ui-tars, mai-ui")
    bounds = payload.get("bounds", [])
    if bounds == [0, 0, 0, 0]:
        pytest.skip(f"Skipping placeholder test case: {json_path} (bounds not set)")
    if not payload.get("screenshot_a") or not payload.get("screenshot_b"):
        pytest.skip(f"Missing screenshots in {json_path}")
    for p in (payload.get("screenshot_a"), payload.get("screenshot_b")):
        if p and not Path(p).exists():
            pytest.skip(f"Screenshot not found: {p}")

    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
    else:
        pytest.fail(
            f"No groundtruth found in {json_path}. "
            f"Test case must include 'expected_passed' (boolean) or 'label' (string) field."
        )
    _set_checker_report_meta(request, expected_passed=expected_pass)

    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "count_change" / f"{platform}_{app_name}"
    try:
        result = check_count_change(payload, debug_dir=debug_dir, api_key=api_key, model=model, base_url=base_url)
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
    _set_checker_report_meta(request, actual_passed=bool(result.passed))