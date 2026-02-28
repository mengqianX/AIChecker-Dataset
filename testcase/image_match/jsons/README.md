# Image Match JSON配置文件说明

## template_image 和 target_image 的含义

**是的，这是 OpenCV / 图像匹配里常用的专有命名**，和日常中文「目标」的直觉可能相反，需要单独记一下。

| 字段 | 含义 | 典型内容 | 谁大谁小 |
|------|------|----------|----------|
| **template_image** | **模板图** = **要找的图案**（你要找的「目标物」） | 图标、按钮、从截图上裁下的一小块 | 一般是**小图** |
| **target_image** | **目标图** = **被搜索的那张图**（搜索发生的「场所」） | 整张截图、完整界面 | 一般是**大图** |

**一句话**：在 **target_image** 里找 **template_image** 出现的位置。

### 为什么容易理解反？

- 中文里「目标」常被理解成「要找的东西」→ 会误以为 target_image = 要找的小图。
- 在本接口里 **target** 取的是「**被搜索的对象**」这层意思，即「搜索操作施加于其上的那张图」= 大图。
- **要找的东西**（你的目标物）对应的是 **template_image**（模板图）。

可以这样记：

- **template** = 要找的**图案**（小图）
- **target** = 被搜索的**整张图**（大图）；或记：target = 「在谁身上找」的那张图

命名来源：OpenCV 的 `matchTemplate(image, templ, ...)` 里，`templ` 是模板（要找的图案），`image` 是被搜索的图；这里把被搜索的图命名为 `target_image`。

若 JSON 里只写了 `image_a` / `image_b`，会按**尺寸自动判定**：小图当模板、大图当目标，无需区分命名。

## 文件命名规范

- 使用描述性的名称，如 `icon_search.json`、`button_login.json`
- 避免使用通用名称，如 `test1.json`、`case.json`

## JSON字段详解

### 必需字段

- `type`: 固定为 `"image_match"`
- `template_image`: 模板图片路径（相对于JSON文件）— 要寻找的小图/图案
- `target_image`: 目标图片路径（相对于JSON文件）— 被搜索的大图/整张图  
  也可用 `image_a` / `image_b`，程序会按尺寸自动确定谁当模板、谁当目标
- `expected_passed`: 人工标注的groundtruth（true/false）

### 可选字段

- `similarity_threshold`: 相似度阈值（0-1），默认0.65
- `scale_min` / `scale_max` / `scale_step`: 多尺度搜索范围
- `match_method`: 匹配方法，默认`"TM_CCOEFF_NORMED"`
- `expected_bounds`: 期望的匹配位置 `[left, top, right, bottom]`（如来自 UI layout），用于模糊校验
- **bounds 模糊匹配**（与 `expected_bounds` 搭配，三选一或使用默认）：
  - `bounds_tolerance`: 每边允许的最大像素误差，默认 **30**。四边都满足 `|actual - expected| ≤ bounds_tolerance` 即通过。
  - `bounds_center_tolerance`: 用「中心点距离」判定：实际框中心与期望框中心的距离 ≤ 该值（像素）即通过。适合模板缩放略有差异时。
  - `bounds_iou_min`: 用 IoU 判定：`IoU(actual, expected) ≥ bounds_iou_min`（0~1）即通过。兼顾位置和大小偏差。
- `description`: 测试用例描述

## 路径说明

JSON中的路径是**相对于JSON文件所在目录**的相对路径。

例如：
- JSON文件位置：`testcase/image_match/jsons/Android/app1.json`
- 模板图片路径：`../../templates/Android/icon_search.png`
- 实际解析为：`testcase/image_match/templates/Android/icon_search.png`

## 示例

### 示例1：基本配置

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

### 示例2：自定义缩放范围

```json
{
  "type": "image_match",
  "template_image": "../../templates/Android/button_login.png",
  "target_image": "../../screens/Android/login_page.png",
  "similarity_threshold": 0.85,
  "scale_min": 0.3,
  "scale_max": 5.0,
  "scale_step": 0.05,
  "expected_passed": true,
  "description": "在登录页面中查找登录按钮（跨分辨率）"
}
```

### 示例3：验证位置准确性（模糊匹配）

```json
{
  "type": "image_match",
  "template_image": "../../templates/Android/icon_settings.png",
  "target_image": "../../screens/Android/app_main.png",
  "similarity_threshold": 0.65,
  "expected_passed": true,
  "expected_bounds": [1200, 100, 1300, 200],
  "bounds_tolerance": 40,
  "description": "验证设置图标位置：四边允许 40px 误差"
}
```

按需选择一种 bounds 校验方式（优先级：center > iou > 逐边）：

- **只设 `bounds_tolerance`**（或都不设，默认 30px）：左/上/右/下每条边与 expected 差不超过该值即通过。
- **设 `bounds_center_tolerance`**：不要求四边精确，只要求「检测框中心」与「期望框中心」距离 ≤ 该值（像素）。适合 UI 定位与图像匹配框大小略有差异时。
- **设 `bounds_iou_min`**（如 0.5）：要求两个矩形的 IoU ≥ 该值，同时容忍位置和大小偏差。

### 示例4：Negative Case（元素不存在）

```json
{
  "type": "image_match",
  "template_image": "../../templates/Android/icon_notification.png",
  "target_image": "../../screens/Android/home_page.png",
  "similarity_threshold": 0.8,
  "expected_passed": false,
  "description": "验证首页中不存在通知图标（negative case）"
}
```
