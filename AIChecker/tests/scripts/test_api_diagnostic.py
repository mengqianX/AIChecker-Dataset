#!/usr/bin/env python3
"""
通用 API 诊断脚本
支持测试 Qwen、UI-TARS 等 OpenAI 兼容的多模态 API。

用法:
  # 使用 Qwen (DashScope)
  python api_diagnostic.py --backend qwen --model qwen-vl-max

  # 使用 UI-TARS (HuggingFace Inference Endpoints)
  python api_diagnostic.py --backend ui-tars \
    --base-url "https://xxx.inference.endpoints.huggingface.cloud/v1" \
    --model "ByteDance-Seed/UI-TARS-1.5-7B"

  # 使用 MAI-UI (Tongyi-MAI)
  python test_api_diagnostic.py --backend mai-ui

  # 使用 payload JSON 文件
  python test_api_diagnostic.py --payload testcase/count_change/jsons/Android/sample_comment.json

  # 简单连接测试（不测试 count_change）
  python test_api_diagnostic.py --backend qwen --test-connection-only
"""
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("❌ 请安装 openai 包: pip install openai")
    sys.exit(1)


def encode_image(path: str) -> str:
    """将图片编码为 base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_size(path: str) -> tuple[int, int]:
    """获取图片尺寸"""
    from PIL import Image
    img = Image.open(path)
    return img.size


def build_count_change_prompt(bounds: list[int], width: int, height: int) -> str:
    """构建数量变化检测的 prompt"""
    box = f"({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]})"
    return f"""你是一个UI自动化测试专家。请分析操作前后的两张页面截图，检测目标按钮响应后相关的数量是否发生变化。

**任务要求：**
1. 识别页面中与数量相关的控件（如评论数、点赞数、商品数、收藏数、关注数、消息数等）
2. 比较操作前后这些数量的变化
3. 判断目标按钮的操作是否导致了预期的数量变化
4. 提取目标控件（bounds: {box}）的相关信息

**目标控件位置：**
- bounds: {box} (left, top, right, bottom)
- 图片尺寸: {width} x {height}

**请按以下JSON格式返回结果：**
{{
    "passed": true/false,
    "basis": "判定依据的详细说明",
    "count_changes": [
        {{
            "type": "评论数/点赞数/商品数等",
            "before": 数值或null,
            "after": 数值或null,
            "changed": true/false,
            "bounds": [left, top, right, bottom],
            "text": "显示的文本内容"
        }}
    ],
    "target_control": {{
        "bounds": [left, top, right, bottom],
        "text": "文本内容或null",
        "semantics": "语义描述或null",
        "main_color": [r, g, b]
    }}
}}

请确保返回的 JSON 格式正确，可直接被解析。"""


def build_image_presence_prompt() -> str:
    """构建图像包含关系检测 prompt（判断图片 B 是否包含图片 A）"""
    return """你是一个图像匹配与定位专家。现在有两张图片：
- 第一张是模板图 A（小图，目标）
- 第二张是目标图 B（大图，待搜索）

任务：判断图片 B 中是否存在与图片 A 对应的区域（允许缩放、轻微压缩失真与亮度变化）。

请按以下 JSON 格式返回（不要输出额外解释）：
{
  "exists": true/false,
  "confidence": 0.0-1.0,
  "basis": "判定依据，简明说明纹理/形状/颜色等线索",
  "best_match_bbox": [left, top, right, bottom] 或 null,
  "candidate_count": 候选区域数量（整数）
}

要求：
1. 如果无法确认，优先返回 exists=false，并在 basis 说明不确定原因；
2. 坐标基于目标图 B 像素坐标系；
3. 保证输出是合法 JSON，可直接解析。"""


def extract_json(text: str) -> dict:
    """从响应文本中提取 JSON"""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise ValueError(f"Cannot extract JSON from: {text[:300]}...")
    return json.loads(text[start:end])


def resolve_backend_config(
    backend: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str, str]:
    """根据 backend 解析配置
    
    支持的 qwen 模型：
    - qwen-vl-max (默认，最强模型)
    - qwen-vl-plus
    - qwen-vl-7b (小模型)
    可通过 QWEN_MODEL 环境变量或 --model 参数指定
    """
    backend = (backend or "qwen").strip().lower()

    if backend == "qwen":
        base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 默认使用 qwen-vl-max，可通过 QWEN_MODEL 或 --model 指定其他模型（如 qwen-vl-7b）
        model = model or os.getenv("QWEN_MODEL", "qwen-vl-max")
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        return base_url, model, api_key

    elif backend == "ui-tars":
        base_url = base_url or os.getenv("UI_TARS_BASE_URL")
        if not base_url:
            raise ValueError("请设置 UI_TARS_BASE_URL 或 --base-url")
        model = model or os.getenv("UI_TARS_MODEL", "ui-tars")
        api_key = api_key or os.getenv("UI_TARS_API_KEY") or os.getenv("HF_TOKEN", "dummy")
        return base_url, model, api_key

    elif backend == "mai-ui":
        base_url = base_url or os.getenv("MAI_UI_BASE_URL", "https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-36ace45a-ca9d-4038-a1ee-31005934c852/user-9cf8e66e-8726-42ab-9eeb-3a458295c571/vscode/86a2108b-57cd-4e9d-8b5b-c0b5ae5cb45a/6af74476-37cd-47bb-98dd-7eeca5cba313/proxy/8000/v1")
        model = model or os.getenv("MAI_UI_MODEL", "Tongyi-MAI/MAI-UI-8B")
        api_key = api_key or os.getenv("MAI_UI_API_KEY") or os.getenv("HF_TOKEN", "dummy")
        return base_url, model, api_key

    else:
        raise ValueError(f"不支持的 backend: {backend}. 支持: qwen, ui-tars, mai-ui")


def test_connection(base_url: str, api_key: str, model: str) -> bool:
    """测试简单的文本 API 连接"""
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "你好，请回复'测试成功'"}],
            max_tokens=50,
        )
        if response.choices and response.choices[0].message.content:
            print(f"✅ 文本 API 连接成功: {response.choices[0].message.content}")
            return True
        print("❌ API 调用成功但响应为空")
        return False
    except Exception as e:
        print(f"❌ 文本 API 连接失败: {e}")
        return False


def test_multimodal_connection(base_url: str, api_key: str, model: str, image_path: str) -> bool:
    """测试多模态 API 连接"""
    if not Path(image_path).exists():
        print(f"⚠️ 测试图片不存在: {image_path}")
        return False
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    {"type": "text", "text": "请描述这张图片的主要内容，用一句话回答。"},
                ],
            }],
            max_tokens=100,
        )
        if response.choices and response.choices[0].message.content:
            print(f"✅ 多模态 API 连接成功: {response.choices[0].message.content[:100]}...")
            return True
        print("❌ 多模态 API 调用成功但响应为空")
        return False
    except Exception as e:
        print(f"❌ 多模态 API 连接失败: {e}")
        return False


def test_count_change(
    base_url: str,
    api_key: str,
    model: str,
    image_a: str,
    image_b: str,
    bounds: list[int],
    stream: bool = False,
) -> dict:
    """测试完整的 count_change 功能"""
    client = OpenAI(base_url=base_url, api_key=api_key)
    img_a_b64 = encode_image(image_a)
    img_b_b64 = encode_image(image_b)
    width, height = get_image_size(image_b)
    prompt = build_count_change_prompt(bounds, width, height)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    kwargs = {"model": model, "messages": messages, "max_tokens": 1024, "temperature": 0.0}

    if stream:
        resp = client.chat.completions.create(**kwargs, stream=True)
        text = ""
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
    else:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content

    return extract_json(text)


def test_image_presence(
    base_url: str,
    api_key: str,
    model: str,
    image_a: str,
    image_b: str,
    stream: bool = False,
) -> dict:
    """测试图像包含关系：图片 B 是否存在图片 A"""
    client = OpenAI(base_url=base_url, api_key=api_key)
    img_a_b64 = encode_image(image_a)
    img_b_b64 = encode_image(image_b)
    prompt = build_image_presence_prompt()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    kwargs = {"model": model, "messages": messages, "max_tokens": 512, "temperature": 0.0}

    if stream:
        resp = client.chat.completions.create(**kwargs, stream=True)
        text = ""
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
    else:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content

    return extract_json(text)


def parse_bounds(s: str) -> list[int]:
    """解析 bounds 字符串"""
    return [int(x.strip()) for x in s.split(",")]


def resolve_image_path(json_path: Path, rel_path: str) -> str:
    """解析图片路径（相对于 JSON 文件或当前目录）"""
    if Path(rel_path).is_absolute():
        return rel_path
    json_dir = json_path.parent
    cwd = Path.cwd()
    p1 = (json_dir / rel_path).resolve()
    if p1.exists():
        return str(p1)
    p2 = (cwd / rel_path.lstrip("./")).resolve()
    if p2.exists():
        return str(p2)
    return str(p1)


def main():
    parser = argparse.ArgumentParser(description="通用 API 诊断脚本")
    parser.add_argument(
        "--task",
        type=str,
        default="count_change",
        choices=["count_change", "image_match"],
        help="测试任务类型: count_change(默认) 或 image_match",
    )
    parser.add_argument("--backend", type=str, default=os.getenv("COUNT_CHANGE_BACKEND", "qwen"),
                       help="后端类型: qwen (默认 qwen-vl-max), ui-tars, 或 mai-ui")
    parser.add_argument("--base-url", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--model", type=str, help="模型名")
    parser.add_argument("--payload", type=str, help="测试用例 JSON 文件路径")
    parser.add_argument("--img-a", type=str, help="操作前截图路径")
    parser.add_argument("--img-b", type=str, help="操作后截图路径")
    parser.add_argument("--bounds", type=str, help="bounds: left,top,right,bottom")
    parser.add_argument("--test-connection-only", action="store_true",
                       help="仅测试 API 连接，不测试 count_change")
    parser.add_argument("--stream", action="store_true", help="使用流式响应")
    args = parser.parse_args()

    # 解析后端配置
    try:
        base_url, model, api_key = resolve_backend_config(
            backend=args.backend,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)

    print(f"后端: {args.backend}")
    print(f"base_url: {base_url}")
    print(f"model: {model}")
    print(f"api_key: {'*' * 10 if api_key else '未设置'}")
    print()

    # 简单连接测试
    if args.test_connection_only:
        print("=== 测试 API 连接 ===")
        text_ok = test_connection(base_url, api_key, model)
        if text_ok:
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            test_img = repo_root / "testcase" / "count_change" / "screens" / "Android" / "sample" / "like_before.png"
            if test_img.exists():
                test_multimodal_connection(base_url, api_key, model, str(test_img))
        return

    # 完整 count_change 测试
    if args.payload:
        with open(args.payload) as f:
            p = json.load(f)
        json_path = Path(args.payload)
        if args.task == "image_match":
            img_a_key = "template_image" if "template_image" in p else "screenshot_a"
            img_b_key = "target_image" if "target_image" in p else "screenshot_b"
            if img_a_key not in p or img_b_key not in p:
                parser.error("image_match payload 需包含 template_image/target_image（或 screenshot_a/screenshot_b）")
            img_a = resolve_image_path(json_path, p[img_a_key])
            img_b = resolve_image_path(json_path, p[img_b_key])
            bounds = None
        else:
            img_a = resolve_image_path(json_path, p["screenshot_a"])
            img_b = resolve_image_path(json_path, p["screenshot_b"])
            bounds = p["bounds"]
        base_url = args.base_url or p.get("base_url") or base_url
        model = args.model or p.get("model") or model
        api_key = args.api_key or p.get("api_key") or api_key
    else:
        if args.task == "image_match":
            if not all([args.img_a, args.img_b]):
                parser.error("image_match 需指定 --payload 或同时指定 --img-a, --img-b")
            bounds = None
        else:
            if not all([args.img_a, args.img_b, args.bounds]):
                parser.error("count_change 需指定 --payload 或同时指定 --img-a, --img-b, --bounds")
            bounds = parse_bounds(args.bounds)
        img_a, img_b = args.img_a, args.img_b

    print(f"=== 测试 {args.task} 功能 ===")
    print(f"img_a: {img_a}")
    print(f"img_b: {img_b}")
    if bounds is not None:
        print(f"bounds: {bounds}")
    print()

    try:
        if args.task == "image_match":
            result = test_image_presence(
                base_url=base_url,
                api_key=api_key,
                model=model,
                image_a=img_a,
                image_b=img_b,
                stream=args.stream,
            )
        else:
            assert bounds is not None
            result = test_count_change(
                base_url=base_url,
                api_key=api_key,
                model=model,
                image_a=img_a,
                image_b=img_b,
                bounds=bounds,
                stream=args.stream,
            )
        print(f"✅ {args.task} 测试成功")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"❌ {args.task} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
