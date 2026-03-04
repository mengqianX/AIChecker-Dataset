#!/usr/bin/env python3
"""
UI-TARS API 独立测试脚本
用于验证 UI-TARS 服务是否可用，无需依赖 AIChecker 其他模块。
"""
import argparse
import base64
import json
import os
import re
from pathlib import Path

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
目标控件 bounds: {box}, 图片尺寸: {width}x{height}
请按JSON返回：{{"passed":true/false,"basis":"...","count_changes":[],"target_control":{{}}}}"""


def extract_json(text: str) -> dict:
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


def parse_bounds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="测试 UI-TARS API")
    parser.add_argument("--payload", type=str)
    parser.add_argument("--base-url", type=str, default=os.getenv("UI_TARS_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=os.getenv("UI_TARS_API_KEY", "dummy"))
    parser.add_argument("--model", type=str, default=os.getenv("UI_TARS_MODEL", "ui-tars"))
    parser.add_argument("--img-a", type=str)
    parser.add_argument("--img-b", type=str)
    parser.add_argument("--bounds", type=str)
    parser.add_argument("--stream", action="store_true")
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
        img_a, img_b, bounds = args.img_a, args.img_b, parse_bounds(args.bounds)
        base_url, model, api_key = args.base_url, args.model, args.api_key

    if not base_url:
        parser.error("请设置 UI_TARS_BASE_URL 或 --base-url")

    result = test_ui_tars(
        base_url=base_url,
        api_key=api_key,
        model=model,
        image_a=img_a,
        image_b=img_b,
        bounds=bounds,
        stream=args.stream,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
