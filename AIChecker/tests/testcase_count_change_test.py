"""
基于 testcase/count_change 目录的测试用例，验证 check_count_change 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析。
"""
import json
from pathlib import Path

import pytest
from aichecker.checkers import check_count_change
from aichecker.utils import _encode

# 仓库根目录（AIChecker/tests -> AIChecker -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "count_change"
JSONS_DIR = TESTCASE_DIR / "jsons"


def _resolve_path(json_path: Path, rel_path: str) -> Path:
    """将 JSON 中的相对路径解析为绝对路径（相对于该 JSON 所在目录）。"""
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()


def _load_and_resolve_payload(json_path: Path) -> dict:
    """加载 JSON 并解析其中的 screenshot_a、screenshot_b 路径。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("screenshot_a", "screenshot_b"):
        if data.get(key):
            data[key] = str(_resolve_path(json_path, data[key]))
    return data


def _collect_testcase_jsons():
    """收集 jsons 目录下 Android/、HarmonyOS/ 中的 JSON（不含 sample.json）。"""
    cases = []
    # 包含 sample.json
    sample_path = JSONS_DIR / "sample.json"
    if sample_path.exists():
        cases.append(("sample", "sample", sample_path))
    
    # 收集子目录中的JSON
    for subdir in ("Android", "HarmonyOS"):
        dir_path = JSONS_DIR / subdir
        if not dir_path.is_dir():
            continue
        for j in dir_path.glob("*.json"):
            cases.append((subdir, j.stem, j))
    return cases


@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_count_change_from_testcase(platform: str, app_name: str, json_path: Path):
    """
    使用 testcase/count_change 中各应用的 JSON 跑数量变化检测。
    
    重要说明：
    - expected_passed 是人工标注的groundtruth，表示该测试用例期望的检测结果
    - 模型只负责检测数量是否变化，返回 passed（True/False）
    - 测试脚本会比较模型的检测结果与groundtruth，验证模型准确性
    """
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")
    
    payload = _load_and_resolve_payload(json_path)

    # 环境变量覆盖后端（便于切换 qwen / ui-tars / qwen-open 而不改 JSON）
    import os
    backend_override = os.getenv("COUNT_CHANGE_BACKEND")
    if backend_override:
        payload["backend"] = backend_override.strip().lower()
        if payload["backend"] == "ui-tars":
            payload["base_url"] = payload.get("base_url") or os.getenv("UI_TARS_BASE_URL", "http://localhost:8000/v1")
            payload["model"] = payload.get("model") or os.getenv("UI_TARS_MODEL", "ui-tars")
            payload["api_key"] = payload.get("api_key") or os.getenv("UI_TARS_API_KEY") or os.getenv("HF_TOKEN") or "dummy"
        elif payload["backend"] == "qwen":
            # QWEN_MODEL：覆盖模型名（如 qwen2.5-vl-72b、qwen-vl-max），可用 DashScope 或自部署
            # QWEN_BASE_URL：自部署端点时才需设置，不设则用默认 DashScope
            if os.getenv("QWEN_MODEL"):
                payload["model"] = os.getenv("QWEN_MODEL")
            if os.getenv("QWEN_BASE_URL"):
                payload["base_url"] = os.getenv("QWEN_BASE_URL")
                payload["model"] = payload.get("model") or os.getenv("QWEN_MODEL") or "Qwen/Qwen2.5-VL-7B-Instruct"
                payload["api_key"] = payload.get("api_key") or os.getenv("QWEN_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

    # 检查是否有占位符数据（bounds为[0,0,0,0]）
    bounds = payload.get("bounds", [])
    if bounds == [0, 0, 0, 0]:
        pytest.skip(f"Skipping placeholder test case: {json_path} (bounds not set)")
    
    # 检查截图文件是否存在
    if not payload.get("screenshot_a") or not payload.get("screenshot_b"):
        pytest.skip(f"Missing screenshots in {json_path}")
    
    for p in (payload.get("screenshot_a"), payload.get("screenshot_b")):
        if p and not Path(p).exists():
            pytest.skip(f"Screenshot not found: {p}")
    
    # 检查API密钥是否配置
    import os
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and not payload.get("api_key"):
        pytest.skip(
            f"API key not configured. Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable, "
            f"or add api_key to {json_path}"
        )
    
    # 读取人工标注的groundtruth（必须在调用模型之前读取）
    # expected_passed 是人工标注的期望结果，不是模型判断的结果
    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
        groundtruth_source = f"expected_passed={expected_pass}"
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
        groundtruth_source = f"label={payload.get('label')}"
    else:
        pytest.fail(
            f"No groundtruth found in {json_path}. "
            f"Test case must include 'expected_passed' (boolean) or 'label' (string) field "
            f"as manually annotated groundtruth."
        )
    
    # 调用模型进行检测（模型只负责检测，不依赖groundtruth）
    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "count_change" / f"{platform}_{app_name}"
    
    try:
        result = check_count_change(payload, debug_dir=debug_dir)
    except ImportError as e:
        # ImportError（如缺少openai包）应该跳过，这是环境配置问题
        pytest.skip(
            f"{platform}/{app_name}: Missing required dependency. "
            f"Error: {e}"
        )
    except ValueError as e:
        # ValueError可能是API认证错误（401）
        if "API authentication failed" in str(e):
            pytest.skip(
                f"{platform}/{app_name}: API authentication failed (invalid API key). "
                f"This is a configuration issue. Please check your API key. "
                f"Error: {e}"
            )
        else:
            raise
    
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    
    # 检查是否是API错误（环境配置问题）
    if result.details.get("is_api_error"):
        error_msg = result.details.get("error", "")
        is_auth_error = result.details.get("is_auth_error", False)
        
        if is_auth_error:
            # API密钥错误应该跳过测试，这是配置问题
            pytest.skip(
                f"{platform}/{app_name}: API authentication failed (invalid API key). "
                f"This is a configuration issue. Please check your API key. "
                f"Error: {error_msg}"
            )
        else:
            # 其他API错误（如网络错误）也应该跳过，而不是失败
            pytest.skip(
                f"{platform}/{app_name}: API call failed. "
                f"This is an environment/configuration issue, not a test failure. "
                f"Error: {result.basis}. Details: {error_msg}"
            )
    
    # 比较模型的检测结果与人工标注的groundtruth
    # 如果两者不一致，说明模型检测有误
    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: "
        f"Groundtruth (manually annotated) expected_passed={expected_pass}, "
        f"but model detected passed={result.passed}. "
        f"Model basis: {result.basis}"
    )


def test_sample_count_change():
    """测试示例用例（不跳过占位符，用于演示）。"""
    sample_path = JSONS_DIR / "sample.json"
    if not sample_path.exists():
        pytest.skip(f"Sample JSON not found: {sample_path}")
    
    payload = _load_and_resolve_payload(sample_path)
    
    # 检查截图文件是否存在
    screenshot_a = payload.get("screenshot_a")
    screenshot_b = payload.get("screenshot_b")
    
    if screenshot_a and screenshot_b and Path(screenshot_a).exists() and Path(screenshot_b).exists():
        # 检查API密钥
        import os
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key and not payload.get("api_key"):
            pytest.skip("API key not configured")
        
        result = check_count_change(payload)
        print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
        print(f"\n检测结果: {'✅ 通过' if result.passed else '❌ 未通过'}")
        print(f"判定依据: {result.basis}")
    else:
        pytest.skip(f"Screenshots not found or placeholder: {sample_path}")
