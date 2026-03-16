#!/usr/bin/env python3
"""
图片匹配Checker实际使用示例

场景：UI自动化测试中查找搜索图标

假设场景：
- 你有一个完整的应用主页面截图（大图）：1080x1920像素
- 你需要找到页面上的"搜索图标"（小图）：50x50像素
- 找到后，你需要点击这个图标来测试搜索功能

为什么需要小图匹配大图？
1. 不同分辨率适配：测试可能在1080p或4K手机上运行，图标大小不同
2. 元素位置不确定：UI可能因设备、主题、版本而变化，不能硬编码坐标
3. 视觉识别更可靠：比通过ID、XPath等定位更直观
"""

from pathlib import Path
import json
from aichecker.checkers import check_image_match
from aichecker.utils import _encode


def example_1_basic_match():
    """示例1：基本图片匹配"""
    print("=" * 60)
    print("示例1：基本图片匹配 - 在应用主页面中查找搜索图标")
    print("=" * 60)
    
    # 准备测试数据
    payload = {
        "template_image": "testcase/image_match/templates/sample/icon_search.png",  # 小图：搜索图标模板
        "target_image": "testcase/image_match/screens/sample/app_main.png",          # 大图：完整页面截图
        "similarity_threshold": 0.8,  # 相似度阈值
    }
    
    # 解析路径（相对于当前文件）
    script_dir = Path(__file__).parent.parent
    payload["template_image"] = str(script_dir / payload["template_image"])
    payload["target_image"] = str(script_dir / payload["target_image"])
    
    # 检查文件是否存在
    if not Path(payload["template_image"]).exists():
        print(f"⚠️  模板图片不存在: {payload['template_image']}")
        print("   请先准备模板图片和目标截图")
        return
    
    if not Path(payload["target_image"]).exists():
        print(f"⚠️  目标图片不存在: {payload['target_image']}")
        print("   请先准备模板图片和目标截图")
        return
    
    # 执行匹配
    debug_dir = script_dir / "debug" / "example_search_icon"
    result = check_image_match(payload, output=debug_dir)
    
    # 显示结果
    print(f"\n✅ 匹配结果: {'找到匹配' if result.passed else '未找到匹配'}")
    print(f"相似度: {result.details['similarity']:.2%}")
    print(f"匹配位置: {result.details['bounds']}")
    print(f"最佳缩放比例: {result.details['best_scale']:.2f}x")
    print(f"\n判定依据: {result.basis}")
    
    # 如果找到匹配，可以用于自动化操作
    if result.passed:
        bounds = result.details['bounds']
        center_x = (bounds[0] + bounds[2]) // 2
        center_y = (bounds[1] + bounds[3]) // 2
        print(f"\n🎯 可以点击的位置: ({center_x}, {center_y})")
        print("   # 使用Appium: driver.tap([(center_x, center_y)])")
        print("   # 使用Selenium: ActionChains(driver).move_to_element(...).click()")


def example_2_cross_resolution():
    """示例2：跨分辨率匹配（不同设备）"""
    print("\n" + "=" * 60)
    print("示例2：跨分辨率匹配 - 在4K屏幕上查找1080p截图中的图标")
    print("=" * 60)
    
    payload = {
        "template_image": "testcase/image_match/templates/sample/icon_search_1080p.png",  # 从1080p截图提取的图标
        "target_image": "testcase/image_match/screens/sample/app_main_4k.png",              # 4K截图
        "similarity_threshold": 0.8,
        "scale_min": 0.5,   # 允许缩小到50%
        "scale_max": 4.0,   # 允许放大到400%（适应4K）
        "scale_step": 0.1,
    }
    
    script_dir = Path(__file__).parent.parent
    payload["template_image"] = str(script_dir / payload["template_image"])
    payload["target_image"] = str(script_dir / payload["target_image"])
    
    if not Path(payload["template_image"]).exists() or not Path(payload["target_image"]).exists():
        print("⚠️  示例图片不存在，跳过此示例")
        return
    
    result = check_image_match(payload)
    
    print(f"\n✅ 匹配结果: {'找到匹配' if result.passed else '未找到匹配'}")
    print(f"相似度: {result.details['similarity']:.2%}")
    print(f"匹配位置: {result.details['bounds']}")
    print(f"最佳缩放比例: {result.details['best_scale']:.2f}x")
    print("\n💡 说明：即使图标在4K屏幕上更大（约4倍），也能找到匹配")


def example_3_state_verification():
    """示例3：操作前后状态验证"""
    print("\n" + "=" * 60)
    print("示例3：操作前后状态验证 - 点击通知按钮后验证红点是否出现")
    print("=" * 60)
    
    # 操作前：验证通知图标没有红点
    print("\n📋 步骤1：操作前检查（不应该有红点）")
    payload_before = {
        "template_image": "testcase/image_match/templates/sample/icon_notification_red_dot.png",
        "target_image": "testcase/image_match/screens/sample/before_click.png",
        "similarity_threshold": 0.8,
    }
    
    script_dir = Path(__file__).parent.parent
    payload_before["template_image"] = str(script_dir / payload_before["template_image"])
    payload_before["target_image"] = str(script_dir / payload_before["target_image"])
    
    if Path(payload_before["template_image"]).exists() and Path(payload_before["target_image"]).exists():
        result_before = check_image_match(payload_before)
        print(f"   结果: {'❌ 找到红点（不应该）' if result_before.passed else '✅ 没有红点（正确）'}")
        
        # 模拟点击操作
        print("\n🖱️  步骤2：点击通知按钮")
        print("   # driver.tap([(x, y)])")
        
        # 操作后：验证通知图标有红点
        print("\n📋 步骤3：操作后检查（应该有红点）")
        payload_after = {
            "template_image": "testcase/image_match/templates/sample/icon_notification_red_dot.png",
            "target_image": "testcase/image_match/screens/sample/after_click.png",
            "similarity_threshold": 0.8,
        }
        payload_after["template_image"] = str(script_dir / payload_after["template_image"])
        payload_after["target_image"] = str(script_dir / payload_after["target_image"])
        
        if Path(payload_after["target_image"]).exists():
            result_after = check_image_match(payload_after)
            print(f"   结果: {'✅ 找到红点（正确）' if result_after.passed else '❌ 没有红点（不应该）'}")
    else:
        print("⚠️  示例图片不存在，跳过此示例")


def example_4_ui_regression():
    """示例4：UI版本更新后的回归测试"""
    print("\n" + "=" * 60)
    print("示例4：UI回归测试 - 使用旧版本图标在新版本截图中查找")
    print("=" * 60)
    
    payload = {
        "template_image": "testcase/image_match/templates/sample/icon_search_v1.png",  # 旧版本图标
        "target_image": "testcase/image_match/screens/sample/app_main_v2.png",          # 新版本截图
        "similarity_threshold": 0.75,  # 降低阈值，因为可能有细微变化
    }
    
    script_dir = Path(__file__).parent.parent
    payload["template_image"] = str(script_dir / payload["template_image"])
    payload["target_image"] = str(script_dir / payload["target_image"])
    
    if Path(payload["template_image"]).exists() and Path(payload["target_image"]).exists():
        result = check_image_match(payload)
        print(f"\n✅ 匹配结果: {'找到匹配（UI兼容）' if result.passed else '未找到匹配（UI可能已改变）'}")
        print(f"相似度: {result.details['similarity']:.2%}")
        print("\n💡 说明：即使UI有细微变化，也能找到匹配（如果相似度足够）")
    else:
        print("⚠️  示例图片不存在，跳过此示例")


def example_5_automation_integration():
    """示例5：集成到自动化测试框架"""
    print("\n" + "=" * 60)
    print("示例5：集成到自动化测试 - 完整的登录流程验证")
    print("=" * 60)
    
    script_dir = Path(__file__).parent.parent
    
    # 步骤1：验证登录页面
    print("\n📋 步骤1：检查登录页面是否显示")
    payload_login_page = {
        "template_image": "testcase/image_match/templates/sample/login_title.png",
        "target_image": "testcase/image_match/screens/sample/app_launch.png",
        "similarity_threshold": 0.8,
    }
    payload_login_page["template_image"] = str(script_dir / payload_login_page["template_image"])
    payload_login_page["target_image"] = str(script_dir / payload_login_page["target_image"])
    
    if Path(payload_login_page["target_image"]).exists():
        result = check_image_match(payload_login_page)
        if result.passed:
            print(f"   ✅ 找到登录页面，位置: {result.details['bounds']}")
        else:
            print("   ❌ 未找到登录页面")
            return
        
        # 步骤2：找到登录按钮
        print("\n📋 步骤2：查找登录按钮")
        payload_login_button = {
            "template_image": "testcase/image_match/templates/sample/button_login.png",
            "target_image": "testcase/image_match/screens/sample/app_launch.png",
            "similarity_threshold": 0.8,
        }
        payload_login_button["template_image"] = str(script_dir / payload_login_button["template_image"])
        payload_login_button["target_image"] = str(script_dir / payload_login_button["target_image"])
        
        result = check_image_match(payload_login_button)
        if result.passed:
            bounds = result.details['bounds']
            button_x = (bounds[0] + bounds[2]) // 2
            button_y = (bounds[1] + bounds[3]) // 2
            print(f"   ✅ 找到登录按钮，位置: ({button_x}, {button_y})")
            print(f"   🖱️  可以点击: driver.tap([({button_x}, {button_y})])")
            
            # 步骤3：验证登录成功
            print("\n📋 步骤3：验证登录成功（查找用户头像）")
            payload_avatar = {
                "template_image": "testcase/image_match/templates/sample/avatar.png",
                "target_image": "testcase/image_match/screens/sample/after_login.png",
                "similarity_threshold": 0.8,
            }
            payload_avatar["template_image"] = str(script_dir / payload_avatar["template_image"])
            payload_avatar["target_image"] = str(script_dir / payload_avatar["target_image"])
            
            if Path(payload_avatar["target_image"]).exists():
                result = check_image_match(payload_avatar)
                if result.passed:
                    print(f"   ✅ 登录成功，找到用户头像，位置: {result.details['bounds']}")
                else:
                    print("   ❌ 登录失败，未找到用户头像")
        else:
            print("   ❌ 未找到登录按钮")
    else:
        print("⚠️  示例图片不存在，跳过此示例")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("图片匹配Checker使用示例")
    print("=" * 60)
    print("\n这些示例展示了在实际项目中如何使用图片匹配功能：")
    print("1. 基本图片匹配")
    print("2. 跨分辨率匹配")
    print("3. 操作前后状态验证")
    print("4. UI回归测试")
    print("5. 集成到自动化测试")
    print("\n注意：示例需要实际的图片文件才能运行")
    print("=" * 60)
    
    # 运行示例（如果图片不存在会跳过）
    example_1_basic_match()
    example_2_cross_resolution()
    example_3_state_verification()
    example_4_ui_regression()
    example_5_automation_integration()
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)
    print("\n💡 提示：")
    print("1. 准备实际的图片文件后，这些示例可以直接运行")
    print("2. 查看 debug/ 目录下的可视化结果，了解匹配详情")
    print("3. 根据实际情况调整 similarity_threshold 参数")
    print("4. 更多信息请查看文档：docs/IMAGE_MATCH_CONCRETE_EXAMPLE.md")


if __name__ == "__main__":
    main()
