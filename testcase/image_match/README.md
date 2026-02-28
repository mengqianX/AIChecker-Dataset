# Image Match 测试用例目录

本目录包含图片匹配检测（Image Match）的测试用例。

## 使用场景

图片匹配checker主要用于以下场景：

### 1. **UI元素定位验证**
- **场景**：验证某个图标、按钮或UI元素是否出现在屏幕上
- **示例**：检查搜索图标是否在导航栏中，登录按钮是否在页面底部
- **应用**：UI自动化测试、视觉回归测试

### 2. **跨分辨率适配验证**
- **场景**：在不同分辨率/DPI的设备上验证UI元素是否存在
- **示例**：在1080p和4K屏幕上查找同一个图标
- **应用**：响应式设计测试、多设备兼容性测试

### 3. **图标/Logo识别**
- **场景**：识别应用中的特定图标或Logo
- **示例**：验证应用启动时是否显示正确的Logo
- **应用**：品牌一致性检查、UI元素识别

### 4. **截图对比**
- **场景**：在操作前后的截图中查找特定元素
- **示例**：验证点击某个按钮后，某个图标是否出现/消失
- **应用**：交互流程验证、状态变化检测

### 5. **模板匹配**
- **场景**：使用小图（模板）在大图中查找匹配位置
- **示例**：在完整页面截图中定位某个小组件的位置
- **应用**：元素定位、坐标提取

## 目录结构

```
image_match/
├── jsons/                    # 测试用例JSON配置文件
│   ├── sample.json           # 示例测试用例
│   ├── Android/              # Android平台测试用例
│   │   ├── icon_search.json
│   │   ├── button_login.json
│   │   └── logo_app.json
│   ├── HarmonyOS/            # HarmonyOS平台测试用例
│   │   ├── icon_search.json
│   │   └── button_login.json
│   └── README.md             # JSON文件说明
└── templates/                # 模板图片目录（小图A）
    ├── sample/               # 示例模板
    ├── Android/              # Android平台模板
    └── HarmonyOS/            # HarmonyOS平台模板
└── screens/                  # 目标截图目录（大图B）
    ├── sample/               # 示例截图
    ├── Android/              # Android平台截图
    └── HarmonyOS/            # HarmonyOS平台截图
```

## 快速开始

### 1. 准备测试数据

#### 步骤1：准备模板图片（小图A）
- 从完整截图中裁剪出要查找的元素（图标、按钮等）
- 保存为PNG格式（推荐，无损）
- 放到 `templates/` 对应的子目录下

**示例**：
```bash
# 从完整截图中裁剪搜索图标
# 原始截图：screens/Android/app_main.png
# 裁剪后：templates/Android/icon_search.png
```

#### 步骤2：准备目标截图（大图B）
- 完整的页面截图
- 应该包含模板图片（用于positive测试）或不包含（用于negative测试）
- 放到 `screens/` 对应的子目录下

#### 步骤3：创建JSON配置文件
- 在 `jsons/` 目录下创建JSON文件
- 配置模板图片路径、目标截图路径、相似度阈值等
- 标注 `expected_passed`（groundtruth）

**示例JSON**：
```json
{
  "type": "image_match",
  "template_image": "../../templates/Android/icon_search.png",
  "target_image": "../../screens/Android/app_main.png",
  "similarity_threshold": 0.8,
  "expected_passed": true,
  "description": "在Android应用主页面中查找搜索图标"
}
```

### 2. 运行测试

```bash
cd AIChecker
pytest tests/testcase_image_match_test.py -v
```

### 3. 使用Python API

```python
from aichecker.checkers import check_image_match
import json
from pathlib import Path

# 加载测试用例
json_path = Path("testcase/image_match/jsons/sample.json")
with open(json_path, "r") as f:
    payload = json.load(f)

# 解析路径（相对于JSON文件所在目录）
def resolve_path(json_path, rel_path):
    return str((json_path.parent / rel_path).resolve())

payload["template_image"] = resolve_path(json_path, payload["template_image"])
payload["target_image"] = resolve_path(json_path, payload["target_image"])

# 执行检测
result = check_image_match(payload, debug_dir=Path("./debug"))

# 查看结果
print(f"检测结果: {'✅ 找到匹配' if result.passed else '❌ 未找到匹配'}")
print(f"相似度: {result.details['similarity']:.2%}")
print(f"匹配位置: {result.details['bounds']}")
```

## JSON字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 测试类型，固定为 `"image_match"` |
| `template_image` | string | 是 | 模板图片路径（小图A，相对于JSON文件） |
| `target_image` | string | 是 | 目标图片路径（大图B，相对于JSON文件） |
| `similarity_threshold` | float | 否 | 相似度阈值（0-1），默认0.8 |
| `scale_min` | float | 否 | 最小缩放比例，默认0.5 |
| `scale_max` | float | 否 | 最大缩放比例，默认2.0 |
| `scale_step` | float | 否 | 缩放步长，默认0.1 |
| `match_method` | string | 否 | 匹配方法，默认`"TM_CCOEFF_NORMED"` |
| `expected_passed` | boolean | 是 | **人工标注的groundtruth**（true=应找到匹配，false=不应找到匹配） |
| `expected_bounds` | array | 否 | 期望的匹配位置 `[left, top, right, bottom]`（可选，用于验证位置准确性） |
| `description` | string | 否 | 测试用例描述 |

## 测试场景示例

### 场景1：图标查找（Positive Case）
- **描述**：在页面中查找搜索图标
- **模板**：`templates/Android/icon_search.png`（从截图中裁剪的搜索图标）
- **目标**：`screens/Android/app_main.png`（完整页面截图）
- **预期**：`expected_passed: true`（应该找到匹配）

### 场景2：按钮查找（Positive Case）
- **描述**：在登录页面中查找登录按钮
- **模板**：`templates/Android/button_login.png`
- **目标**：`screens/Android/login_page.png`
- **预期**：`expected_passed: true`

### 场景3：元素不存在（Negative Case）
- **描述**：在首页中查找不存在的设置图标
- **模板**：`templates/Android/icon_settings.png`
- **目标**：`screens/Android/home_page.png`
- **预期**：`expected_passed: false`（不应该找到匹配）

### 场景4：跨分辨率匹配
- **描述**：在不同分辨率的截图中查找同一个图标
- **模板**：`templates/Android/icon_search.png`（1080p截图中的图标）
- **目标**：`screens/Android/app_main_4k.png`（4K截图）
- **预期**：`expected_passed: true`（应该能找到，即使分辨率不同）

## 如何收集测试用例

### 方法1：手动收集（推荐用于小规模测试）

1. **准备模板图片**：
   ```bash
   # 使用图片编辑工具（如GIMP、Photoshop）从截图中裁剪
   # 或使用Python脚本自动裁剪
   python scripts/crop_template.py \
     --input screens/Android/app_main.png \
     --bounds 100,200,150,250 \
     --output templates/Android/icon_search.png
   ```

2. **准备目标截图**：
   - 使用自动化测试工具（如Appium、Selenium）截图
   - 或手动截图保存

3. **创建JSON配置**：
   - 参考 `jsons/sample.json`
   - 填写路径和参数
   - **重要**：人工查看截图，标注 `expected_passed`

### 方法2：批量收集（推荐用于大规模测试）

使用提供的脚本批量生成测试用例：

```bash
# 从现有测试用例中提取模板
python scripts/extract_templates.py \
  --screenshot-dir testcase/button_color_change/screens \
  --bounds-file testcase/button_color_change/jsons/Android/app1.json \
  --output-dir testcase/image_match/templates/Android

# 批量生成JSON配置
python scripts/generate_image_match_cases.py \
  --template-dir testcase/image_match/templates \
  --screenshot-dir testcase/image_match/screens \
  --output-dir testcase/image_match/jsons
```

### 方法3：从UI自动化测试中收集

```python
# 在UI自动化测试中自动收集
from appium import webdriver
from PIL import Image

# 1. 截图
driver.save_screenshot("screens/Android/app_main.png")

# 2. 获取元素位置并裁剪模板
element = driver.find_element_by_id("search_icon")
bounds = element.location, element.size
crop_template("screens/Android/app_main.png", bounds, "templates/Android/icon_search.png")

# 3. 生成测试用例JSON
generate_test_case(
    template="templates/Android/icon_search.png",
    target="screens/Android/app_main.png",
    expected_passed=True
)
```

## 注意事项

1. **Groundtruth标注（重要）**：
   - `expected_passed` **必须由人工标注**，不能依赖模型判断
   - 标注人员需要查看模板图片和目标截图，人工判断是否真的存在匹配
   - 这是评估模型准确性的标准答案（groundtruth）

2. **模板图片质量**：
   - 模板图片应该清晰，避免模糊
   - 建议使用PNG格式（无损）
   - 模板图片不要太大，建议不超过目标图片的1/4

3. **相似度阈值**：
   - 默认0.8通常效果较好
   - 如果误匹配多，提高阈值（如0.85）
   - 如果漏匹配多，降低阈值（如0.75）

4. **缩放范围**：
   - 默认0.5-2.0倍通常足够
   - 如果分辨率差异很大，可以扩大范围（如0.3-5.0）
   - 注意：范围越大，搜索时间越长

5. **测试流程**：
   - 步骤1：人工查看模板和目标截图，标注 `expected_passed`（groundtruth）
   - 步骤2：模型检测匹配，返回 `passed`（模型判断）
   - 步骤3：测试脚本比较两者，如果不一致则测试失败

6. **调试输出**：
   - 如果测试失败，查看 `debug/image_match/` 目录下的可视化结果
   - `match_result.png` 会标注匹配位置，便于分析

## 相关文件

- `AIChecker/aichecker/checkers/image_match_checker.py`：检测器实现
- `AIChecker/tests/testcase_image_match_test.py`：测试脚本
- `AIChecker/aichecker/checkers/IMAGE_MATCH_CHECKER_README.md`：详细文档
