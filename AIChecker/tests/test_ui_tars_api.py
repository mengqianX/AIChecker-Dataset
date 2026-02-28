#!/usr/bin/env python3
"""
UI-TARS API 独立测试脚本
用于验证 UI-TARS 服务是否可用，无需依赖 AIChecker 其他模块。

用法:
  # 环境变量配置
  export UI_TARS_BASE_URL="https://xxx.inference.endpoints.huggingface.cloud/v1"
  export UI_TARS_API_KEY="hf_xxx"
  export UI_TARS_MODEL="ByteDance-Seed/UI-TARS-7B-DPO"

  # 运行
  python scripts/test_ui_tars_api.py [--payload payload.json]

  # 或直接指定参数
  python scripts/test_ui_tars_api.py \
    --base-url "http://localhost:8000/v1" \
    --model ui-tars \
    --img-a ./path/to/before.png \
    --img-b ./path/to/after.png \
    --bounds "100,200,300,400"
"""
import argparse
import base64
import json
import os
import re
from pathlib import Path

# pip install openai
from openai import OpenAI


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    img = Image.open(path)
    return img.size


def build_count_change_prompt(bounds: list[int], width: int, height: int) -> str:
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


def test_ui_tars(
    base_url: str,
    api_key: str,
    model: str,
    image_a: str,
    image_b: str,
    bounds: list[int],
    stream: bool = False,
) -> dict:
    client = OpenAI(base_url=base_url, api_key=api_key)
    img_a_b64 = encode_image(image_a)
    img_b_b64 = encode_image(image_b)
    width, height = get_image_size(image_b)
    prompt = build_count_change_prompt(bounds, width, height)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.0,
    }

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
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="测试 UI-TARS API")
    parser.add_argument("--payload", type=str, help="JSON payload 文件路径（含 screenshot_a, screenshot_b, bounds）")
    parser.add_argument("--base-url", type=str, default=os.getenv("UI_TARS_BASE_URL"), help="API base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("UI_TARS_API_KEY", "dummy"), help="API key")
    parser.add_argument("--model", type=str, default=os.getenv("UI_TARS_MODEL", "ui-tars"), help="模型名")
    parser.add_argument("--img-a", type=str, help="操作前截图路径")
    parser.add_argument("--img-b", type=str, help="操作后截图路径")
    parser.add_argument("--bounds", type=str, help="bounds: left,top,right,bottom")
    parser.add_argument("--stream", action="store_true", help="使用流式响应")
    args = parser.parse_args()

    if args.payload:
        with open(args.payload) as f:
            p = json.load(f)
        json_dir = Path(args.payload).parent
        cwd = Path.cwd()

        def resolve_img(path: str) -> str:
            if Path(path).is_absolute():
                return path
            p1 = (json_dir / path).resolve()
            if p1.exists():
                return str(p1)
            p2 = (cwd / path.lstrip("./")).resolve()
            if p2.exists():
                return str(p2)
            return str(p1)

        img_a = resolve_img(p["screenshot_a"])
        img_b = resolve_img(p["screenshot_b"])
        bounds = p["bounds"]
        base_url = args.base_url or p.get("base_url")
        model = args.model or p.get("model", "ui-tars")
        api_key = args.api_key or p.get("api_key", "dummy")
    else:
        if not all([args.img_a, args.img_b, args.bounds]):
            parser.error("需指定 --payload 或同时指定 --img-a, --img-b, --bounds")
        img_a = args.img_a
        img_b = args.img_b
        bounds = parse_bounds(args.bounds)
        base_url = args.base_url
        model = args.model
        api_key = args.api_key

    if not base_url:
        parser.error("请设置 UI_TARS_BASE_URL 或 --base-url")

    print("调用 UI-TARS API...")
    print(f"  base_url: {base_url}")
    print(f"  model: {model}")
    print(f"  img_a: {img_a}")
    print(f"  img_b: {img_b}")
    print(f"  bounds: {bounds}")
    print()

    try:
        result = test_ui_tars(
            base_url=base_url,
            api_key=api_key,
            model=model,
            image_a=img_a,
            image_b=img_b,
            bounds=bounds,
            stream=args.stream,
        )
        print("响应 JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()
        print(f"passed: {result.get('passed')}")
        print(f"basis: {result.get('basis', '')[:200]}...")
    except Exception as e:
        print(f"API 调用失败: {e}")
        raise


if __name__ == "__main__":
    main()
