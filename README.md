# UIOracleBench 测试集说明

## 概览
- **任务**：Oracle 准确率验证，当前聚焦按钮颜色变化检测的准确率。
- **规模**：参见 [`meta.json`](meta.json)，包含 $10$ 个应用、xx 个缺陷、共 xx 个样本。
- **用途**：UIOracleBench 面向 UI Oracle 检测可行性研究，bug 数据通过人工注入构建。

## 目录结构
```
testcase/
└── button_color_change/
    ├── jsons/
    │   └── sample.json
    └── screens/
        ├── button_before.png
        └── button_after.png
```
- `jsons/`：每个样本的配置文件，描述截图路径、裁剪区域与期望颜色等。
- `screens/`：同一界面操作前后的截图，供比对 Oracle。

## JSON 字段
参考 [`testcase/button_color_change/jsons/sample.json`](testcase/button_color_change/jsons/sample.json)：
- `type`：Oracle 类型，当前统一为 `button_color_change`。
- `screenshot_a` / `screenshot_b`：操作前后截图的相对路径。
- `bounds`：感兴趣区域的像素坐标 $[x_1, y_1, x_2, y_2]$。
- `expected_color`：期望颜色，可留空表示需由检测器推断。
- `label`：`pass` / `fail`，标注是否检测到期望变化（作为groundtruth）。

## 使用方法
1. 读取 `jsons/` 中配置，定位按钮区域。
2. 对 `screens/` 截图裁剪并比较颜色变化。
3. 将检测输出与 `label` 比较，评估算法准确率。

