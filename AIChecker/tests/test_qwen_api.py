"""
测试千问API连接
用于诊断API调用问题
"""
import os
import sys
from pathlib import Path

def test_openai_import():
    """测试openai包是否能正常导入"""
    print("=" * 60)
    print("1. 测试openai包导入")
    print("=" * 60)
    try:
        from openai import OpenAI
        print("✅ openai包导入成功")
        return True
    except ImportError as e:
        print(f"❌ openai包导入失败: {e}")
        print("请运行: pip install openai")
        return False

def test_api_key():
    """测试API密钥是否配置"""
    print("\n" + "=" * 60)
    print("2. 测试API密钥配置")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        # 只显示前10个字符和后10个字符，保护密钥
        masked_key = f"{api_key[:10]}...{api_key[-10:]}" if len(api_key) > 20 else "***"
        print(f"✅ 找到API密钥: {masked_key}")
        return api_key
    else:
        print("❌ 未找到API密钥")
        print("请设置环境变量:")
        print("  export DASHSCOPE_API_KEY='your-api-key'")
        print("  或")
        print("  export OPENAI_API_KEY='your-api-key'")
        return None

def test_simple_text_api(api_key: str):
    """测试简单的文本API调用"""
    print("\n" + "=" * 60)
    print("3. 测试简单文本API调用")
    print("=" * 60)
    
    try:
        from openai import OpenAI
        
        # 使用通义千问的兼容端点
        base_url = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 默认使用北京端点
        )
        
        print(f"使用API端点: {base_url}")
        print(f"使用模型: qwen-plus (测试用)")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": "你好，请回复'测试成功'"}
            ],
            max_tokens=50,
        )
        
        if response.choices and response.choices[0].message.content:
            result = response.choices[0].message.content
            print(f"✅ API调用成功!")
            print(f"响应: {result}")
            return True
        else:
            print("❌ API调用成功但响应为空")
            return False
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 详细错误信息
        error_str = str(e)
        if "401" in error_str or "invalid_api_key" in error_str.lower():
            print("\n⚠️  这是API密钥错误（401）")
            print("可能的原因:")
            print("  1. API密钥无效或已过期")
            print("  2. API密钥没有权限访问该服务")
            print("  3. API密钥格式不正确")
        elif "network" in error_str.lower() or "connection" in error_str.lower():
            print("\n⚠️  这是网络连接错误")
            print("可能的原因:")
            print("  1. 网络连接问题")
            print("  2. 防火墙阻止了连接")
            print("  3. 代理设置问题")
        elif "timeout" in error_str.lower():
            print("\n⚠️  这是超时错误")
            print("可能的原因:")
            print("  1. 网络速度慢")
            print("  2. API服务响应慢")
        
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

def test_multimodal_api(api_key: str):
    """测试多模态API调用（使用图片）"""
    print("\n" + "=" * 60)
    print("4. 测试多模态API调用（图片）")
    print("=" * 60)
    
    # 查找测试图片
    repo_root = Path(__file__).resolve().parent.parent.parent
    test_image = repo_root / "testcase" / "count_change" / "screens" / "Android" / "sample" / "like_before.png"
    
    if not test_image.exists():
        print(f"⚠️  测试图片不存在: {test_image}")
        print("跳过多模态API测试")
        return None
    
    try:
        from openai import OpenAI
        import base64
        
        base_url = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 默认使用北京端点
        )
        
        print(f"使用API端点: {base_url}")
        print(f"使用模型: qwen-vl-max")
        print(f"测试图片: {test_image}")
        
        # 编码图片
        with open(test_image, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请描述这张图片的主要内容，用一句话回答。"
                        }
                    ],
                }
            ],
            max_tokens=100,
        )
        
        if response.choices and response.choices[0].message.content:
            result = response.choices[0].message.content
            print(f"✅ 多模态API调用成功!")
            print(f"响应: {result}")
            return True
        else:
            print("❌ 多模态API调用成功但响应为空")
            return False
            
    except Exception as e:
        print(f"❌ 多模态API调用失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        error_str = str(e)
        if "401" in error_str or "invalid_api_key" in error_str.lower():
            print("\n⚠️  这是API密钥错误（401）")
        elif "model" in error_str.lower() and "not found" in error_str.lower():
            print("\n⚠️  模型不存在或不可用")
            print("可能的原因:")
            print("  1. qwen-vl-max模型不可用")
            print("  2. API密钥没有权限访问该模型")
            print("  3. 需要使用其他模型名称")
        
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("千问API连接测试")
    print("=" * 60)
    
    # 1. 测试导入
    if not test_openai_import():
        sys.exit(1)
    
    # 2. 测试API密钥
    api_key = test_api_key()
    if not api_key:
        sys.exit(1)
    
    # 3. 测试简单文本API
    text_success = test_simple_text_api(api_key)
    
    # 4. 测试多模态API
    multimodal_success = test_multimodal_api(api_key)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"文本API: {'✅ 成功' if text_success else '❌ 失败'}")
    print(f"多模态API: {'✅ 成功' if multimodal_success else ('❌ 失败' if multimodal_success is False else '⚠️  跳过')}")
    
    if text_success:
        print("\n✅ 基本API连接正常，可以继续使用")
    else:
        print("\n❌ API连接有问题，请检查:")
        print("  1. 网络连接")
        print("  2. API密钥有效性")
        print("  3. API端点配置")

if __name__ == "__main__":
    main()
