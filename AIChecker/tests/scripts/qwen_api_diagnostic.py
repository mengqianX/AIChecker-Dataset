"""
测试千问API连接
用于诊断API调用问题
"""
import os
import sys
from pathlib import Path


def test_openai_import():
    try:
        from openai import OpenAI  # noqa: F401
        print("✅ openai包导入成功")
        return True
    except ImportError as e:
        print(f"❌ openai包导入失败: {e}")
        return False


def test_api_key():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        masked_key = f"{api_key[:10]}...{api_key[-10:]}" if len(api_key) > 20 else "***"
        print(f"✅ 找到API密钥: {masked_key}")
        return api_key
    print("❌ 未找到API密钥")
    return None


def test_simple_text_api(api_key: str):
    try:
        from openai import OpenAI
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "你好，请回复'测试成功'"}],
            max_tokens=50,
        )
        if response.choices and response.choices[0].message.content:
            print(f"✅ API调用成功: {response.choices[0].message.content}")
            return True
        print("❌ API调用成功但响应为空")
        return False
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return False


def test_multimodal_api(api_key: str):
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    test_image = repo_root / "testcase" / "count_change" / "screens" / "Android" / "sample" / "like_before.png"
    if not test_image.exists():
        print(f"⚠️ 测试图片不存在: {test_image}")
        return None
    try:
        from openai import OpenAI
        import base64
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        with open(test_image, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="qwen-vl-max",
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
            print(f"✅ 多模态API调用成功: {response.choices[0].message.content}")
            return True
        print("❌ 多模态API调用成功但响应为空")
        return False
    except Exception as e:
        print(f"❌ 多模态API调用失败: {e}")
        return False


def main():
    if not test_openai_import():
        sys.exit(1)
    api_key = test_api_key()
    if not api_key:
        sys.exit(1)
    text_success = test_simple_text_api(api_key)
    multimodal_success = test_multimodal_api(api_key)
    print(f"文本API: {'✅ 成功' if text_success else '❌ 失败'}")
    print(f"多模态API: {'✅ 成功' if multimodal_success else ('❌ 失败' if multimodal_success is False else '⚠️ 跳过')}")


if __name__ == "__main__":
    main()
