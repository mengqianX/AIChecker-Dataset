from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from PIL import Image

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import button_base_color, load_image


def _encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _get_image_size(image_path: str) -> Tuple[int, int]:
    """获取图片尺寸"""
    img = load_image(image_path)
    return img.size


# 支持的后端：qwen（千问）| ui-tars（字节跳动 UI-TARS，需 vLLM 或 HuggingFace 部署）
SUPPORTED_BACKENDS = ("qwen", "ui-tars")


def _resolve_backend_config(
    backend: str,
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> Tuple[str, str, str]:
    """
    根据 backend 解析 base_url、model、api_key。
    返回 (base_url, model, api_key)
    """
    import os

    backend = (backend or "qwen").strip().lower()

    if backend == "qwen":
        base_url = base_url or os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model = model or "qwen-vl-max"
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided for Qwen. Set DASHSCOPE_API_KEY or OPENAI_API_KEY, or pass api_key."
            )
        return base_url, model, api_key

    if backend == "ui-tars":
        # UI-TARS 需通过 vLLM 或 HuggingFace Inference Endpoints 部署，提供 OpenAI 兼容 API
        # 本地 vLLM: base_url=http://localhost:8000/v1, api_key 可为空
        # HuggingFace: base_url=xxx.inference.endpoints.huggingface.cloud, api_key=HF_TOKEN
        base_url = base_url or os.getenv("UI_TARS_BASE_URL")
        if not base_url:
            raise ValueError(
                "UI-TARS backend requires base_url. Set UI_TARS_BASE_URL (e.g. http://localhost:8000/v1 for vLLM) "
                "or pass base_url in payload."
            )
        model = model or os.getenv("UI_TARS_MODEL", "UI-TARS-7B-DPO")
        api_key = api_key or os.getenv("UI_TARS_API_KEY") or os.getenv("HF_TOKEN") or "dummy"
        return base_url, model, api_key

    raise ValueError(f"Unsupported backend: {backend}. Supported: {SUPPORTED_BACKENDS}")


def _build_count_change_prompt(bounds: Bounds, img_width: int, img_height: int) -> str:
    """构建数量变化检测的 prompt（与后端无关）"""
    return f"""你是一个UI自动化测试专家。请分析操作前后的两张页面截图，检测目标按钮响应后相关的数量是否发生变化。

**任务要求：**
1. 识别页面中与数量相关的控件（如评论数、点赞数、商品数、收藏数、关注数、消息数等）
2. 比较操作前后这些数量的变化
3. 判断目标按钮的操作是否导致了预期的数量变化
4. 提取目标控件（bounds: {bounds.as_box()}）的相关信息

**目标控件位置：**
- bounds: {bounds.as_box()} (left, top, right, bottom)
- 图片尺寸: {img_width} x {img_height}

**请按以下JSON格式返回结果：**
{{
    "passed": true/false,  // 是否检测到数量变化（True表示检测到变化，False表示未检测到变化）
    "basis": "判定依据的详细说明",  // 说明检测到或未检测到数量变化的原因
    "count_changes": [  // 检测到的数量变化列表（before/after 将用于规则校验，请尽量填可解析的数值）
        {{
            "type": "评论数/点赞数/商品数等",  // 数量类型
            "before": 数值或null,  // 操作前的数量，请填整数、小数或 "1.2万"/"1k" 等可解析形式，无法识别则 null
            "after": 数值或null,  // 操作后的数量，同上
            "changed": true/false,  // 是否发生变化
            "bounds": [left, top, right, bottom],  // 该数量控件的bounds（如果可识别）
            "text": "显示的文本内容"  // 该数量控件的文本内容
        }}
    ],
    "target_control": {{
        "bounds": [left, top, right, bottom],  // 目标控件的bounds
        "text": "文本内容或null",  // 如果是文字按钮，提取文字；如果是图片按钮，则为null
        "semantics": "语义描述或null",  // 如果是图片按钮，描述其语义（如"点赞图标"、"收藏图标"等）
        "main_color": [r, g, b]  // 主背景色的RGB值
    }}
}}

**注意事项：**
- before/after 请尽量填可解析的数值（整数、小数或 "1.2万"/"1k"），以便规则校验使用；无法识别时填 null。
- 若无法识别数量变化，passed 填 false，basis 说明原因。
- 若页面中没有明显的数量相关控件，也返回 false。
- main_color 从目标控件区域提取主要背景色。
- 请确保返回的 JSON 格式正确，可直接被解析。
"""


def _call_multimodal_api(
    image_before_path: str,
    image_after_path: str,
    bounds: Bounds,
    base_url: str,
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    调用多模态 API（OpenAI 兼容格式），进行数量变化检测。
    适用于千问、UI-TARS（vLLM/HuggingFace 部署）等 OpenAI 兼容后端。
    """
    if OpenAI is None:
        raise ImportError(
            "openai package is required. Install it with: pip install openai"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    image_before_b64 = _encode_image_to_base64(image_before_path)
    image_after_b64 = _encode_image_to_base64(image_after_path)
    img_width, img_height = _get_image_size(image_after_path)
    prompt = _build_count_change_prompt(bounds, img_width, img_height)

    # 调用 API（OpenAI 兼容格式）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_before_b64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_after_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ],
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except Exception as api_error:
        # OpenAI SDK可能抛出各种异常，包括认证错误
        # 检查是否是401认证错误
        error_str = str(api_error)
        error_type = type(api_error).__name__
        
        # 检查错误码或错误消息
        is_auth_error = (
            "401" in error_str or 
            "invalid_api_key" in error_str.lower() or
            "Incorrect API key" in error_str.lower() or
            "authentication" in error_str.lower() or
            "unauthorized" in error_str.lower() or
            error_type in ("AuthenticationError", "InvalidAPIKeyError")
        )
        
        # 重新抛出，让上层处理
        if is_auth_error:
            raise ValueError(f"API authentication failed (401): {error_str}") from api_error
        else:
            raise
    
    # 解析响应
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty response from API")
    
    result_text = response.choices[0].message.content
    
    if not result_text:
        raise ValueError("Empty response from API")
    
    # 尝试从响应中提取JSON
    # 模型可能返回markdown格式的JSON（如```json ... ```），需要提取
    json_start = result_text.find("{")
    json_end = result_text.rfind("}") + 1
    
    if json_start == -1 or json_end == 0:
        # 尝试查找markdown代码块
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            raise ValueError(f"Failed to extract JSON from response: {result_text[:500]}")
    else:
        json_str = result_text[json_start:json_end]
    
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nJSON string: {json_str[:500]}")
    
    return result


def check_count_change(
    payload: Dict[str, Any],
    debug_dir: Path | None = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[str] = None,
) -> CheckResult:
    """
    检测目标按钮响应后相关数量是否变化（如评论数、点赞数、商品数等）

    支持多后端：qwen（千问）、ui-tars（字节跳动 UI-TARS，需 vLLM 或 HuggingFace 部署）。
    
    Args:
        payload: 包含以下字段的字典：
            - screenshot_a: 操作前的截图路径
            - screenshot_b: 操作后的截图路径
            - bounds: [left, top, right, bottom] 目标控件的bounds
            - backend (可选): 后端选择，"qwen" 或 "ui-tars"，默认 "qwen"
            - api_key (可选): API密钥
            - model (可选): 模型名称（qwen 默认 qwen-vl-max，ui-tars 默认 UI-TARS-7B-DPO）
            - base_url (可选): API基础URL
        debug_dir: 调试输出目录（可选）
        api_key: API密钥（可选，优先级高于payload）
        model: 模型名称（可选，优先级高于payload）
        base_url: API基础URL（可选，优先级高于payload）
        backend: 后端选择（可选，优先级高于payload）
    
    Returns:
        CheckResult对象，包含：
            - passed: 是否检测到数量变化
            - basis: 判定依据
            - control_info: 控件相关信息
            - details: 详细信息（包含 count_changes、backend、model 等）
    """
    # 解析输入
    screenshot_a = payload.get("screenshot_a")
    screenshot_b = payload.get("screenshot_b") or payload.get("screenshot")

    if not screenshot_a or not screenshot_b:
        raise ValueError(
            "Both screenshot_a and screenshot_b (or screenshot) must be provided"
        )

    bounds = Bounds.from_sequence(payload["bounds"])

    # 后端与配置（优先级：参数 > payload > 环境变量）
    backend = backend or payload.get("backend", "qwen")
    api_key = api_key or payload.get("api_key")
    model = model or payload.get("model")
    base_url = base_url or payload.get("base_url")

    base_url, model, api_key = _resolve_backend_config(backend, api_key, model, base_url)

    # 调用多模态 API
    try:
        result = _call_multimodal_api(
            screenshot_a,
            screenshot_b,
            bounds,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
    except ImportError as e:
        # ImportError（如缺少openai包）应该直接抛出，不应该被捕获为"正常"的检测结果
        # 这是环境配置问题，不是检测逻辑问题
        raise
    except ValueError as e:
        # ValueError可能是API认证错误（401），应该重新抛出
        if "API authentication failed" in str(e):
            raise
        # 其他ValueError继续处理
        return CheckResult(
            passed=False,
            basis=f"API调用失败: {str(e)}",
            control_info=ControlInfo(bounds=bounds, source="api_error"),
            details={
                "error": str(e), 
                "error_type": type(e).__name__,
                "is_api_error": True,
                "is_auth_error": "authentication" in str(e).lower() or "401" in str(e),
            },
        )
    except Exception as e:
        # 其他API调用错误（如网络错误等）可以返回失败结果
        # 但应该在details中明确标记这是API错误，让调用者可以区分
        error_str = str(e)
        is_auth_error = (
            "401" in error_str or 
            "invalid_api_key" in error_str.lower() or
            "Incorrect API key" in error_str.lower() or
            "authentication" in error_str.lower() or
            "unauthorized" in error_str.lower()
        )
        
        return CheckResult(
            passed=False,
            basis=f"API调用失败: {error_str}",
            control_info=ControlInfo(bounds=bounds, source="api_error"),
            details={
                "error": error_str, 
                "error_type": type(e).__name__,
                "is_api_error": True,  # 标记这是API错误
                "is_auth_error": is_auth_error,  # 标记是否是认证错误（API密钥问题）
            },
        )
    
    # 提取结果
    passed = result.get("passed", False)
    basis = result.get("basis", "未提供判定依据")
    count_changes = result.get("count_changes", [])
    target_control = result.get("target_control", {})
    
    # 构建ControlInfo
    target_bounds = target_control.get("bounds", bounds.as_box())
    if isinstance(target_bounds, list) and len(target_bounds) == 4:
        control_bounds = Bounds.from_sequence(target_bounds)
    else:
        control_bounds = bounds
    
    # 提取主背景色
    main_color = None
    if target_control.get("main_color"):
        color = target_control["main_color"]
        if isinstance(color, list) and len(color) == 3:
            main_color = tuple(color)
    
    # 如果API没有返回颜色，尝试从图片中提取
    if main_color is None:
        try:
            img_after = load_image(screenshot_b)
            crop_after = img_after.crop(bounds.as_box())
            main_color = button_base_color(crop_after)
        except Exception:
            pass
    
    control_info = ControlInfo(
        bounds=control_bounds,
        text=target_control.get("text"),
        semantics=target_control.get("semantics"),
        main_color=main_color,
        source="multimodal",
        extras={
            "count_changes": count_changes,
            "model": model,
        },
    )
    
    # 构建details
    details: Dict[str, Any] = {
        "method": "multimodal_count_change",
        "backend": backend,
        "model": model,
        "count_changes": count_changes,
        "target_control_raw": target_control,
        "api_response": result,
    }
    
    return CheckResult(
        passed=passed,
        basis=basis,
        control_info=control_info,
        details=details,
    )
