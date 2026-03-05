# UIOracleBench Dataset

> UI Oracle 检测基准测试数据集

## 概览

- **任务**：Oracle 准确率验证，当前聚焦按钮颜色变化检测的准确率。
- **规模**：参见 [`meta.json`](meta.json)，包含 10 个应用、12 个缺陷、共 96 个样本。
- **用途**：UIOracleBench 面向 UI Oracle 检测可行性研究，bug 数据通过人工注入构建。

## 仓库说明

这是 **UIOracleBench 数据集仓库**，包含测试用例数据和元信息。

- **数据集仓库**（本仓库）：`AIChecker-Dataset` 或 `UIOracleBench-Dataset`
- **工具仓库**：`AIChecker`（包含评估工具和实验代码）

> 💡 **论文 Artifact 使用**：请参考 [`ARTIFACT.md`](ARTIFACT.md) 了解如何整合两个仓库用于论文提交。

## 目录结构

```
testcase/
└── button_color_change/
    ├── jsons/
    │   ├── baidumap/
    │   │   ├── baidumap_1.json
    │   │   └── baidumap_1_f.json
    │   ├── taobao/
    │   │   ├── taobao_1.json
    │   │   └── taobao_1_f.json
    │   ├── xhs/
    │   │   ├── xhs_1.json
    │   │   └── xhs_1_f.json
    │   └── sample.json
    └── screens/
        ├── baidumap/
        ├── taobao/
        └── xhs/
```

- `jsons/`：每个样本的配置文件，描述截图路径、裁剪区域与期望颜色等。
- `screens/`：同一界面操作前后的截图，供比对 Oracle。

## JSON 字段

参考 [`testcase/button_color_change/jsons/sample.json`](testcase/button_color_change/jsons/sample.json)：

- `type`：Oracle 类型，当前统一为 `button_color_change`。
- `screenshot_a` / `screenshot_b`：操作前后截图的相对路径。
- `bounds`：感兴趣区域的像素坐标 $[x_1, y_1, x_2, y_2]$。
- `expected_color`：期望颜色，可留空表示需由检测器推断。
- `expected_passed`：布尔值，groundtruth，表示期望的检测结果（`true`=应检测到变化，`false`=不应检测到变化）。
- `tolerance`（可选）：颜色变化检测的容差值，默认 20。
- `label`（已弃用，保留以向后兼容）：`pass` / `fail`，等同于 `expected_passed`。

## 使用方法

### 独立使用数据集

1. 克隆本仓库：
   ```bash
   git clone <dataset-repo-url>
   cd AIChecker-Dataset
   ```

2. 读取 `jsons/` 中配置，定位按钮区域。
3. 对 `screens/` 截图裁剪并比较颜色变化。
4. 将检测输出与 `label` 比较，评估算法准确率。

### 配合工具仓库使用

1. 克隆工具仓库：
   ```bash
   git clone <tool-repo-url> AIChecker
   ```

2. 在工具仓库中配置数据集路径（参考工具仓库的 README）。

3. 运行评估：
   ```bash
   cd AIChecker
   python evaluate.py --dataset ../AIChecker-Dataset
   ```

## 版本管理

数据集使用语义化版本号（Semantic Versioning）：
- **主版本号**：数据结构重大变更
- **次版本号**：新增测试用例或应用
- **修订号**：修复标注错误、更新元数据

当前版本：参见 [`meta.json`](meta.json) 中的 `version` 字段。

## 许可证

[待添加许可证信息]

## 引用

如果使用本数据集，请引用：

```bibtex
@dataset{uioraclebench2024,
  title={UIOracleBench: A Benchmark for UI Oracle Detection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-org]/AIChecker-Dataset}
}
```

## 相关仓库

- **工具仓库**：[AIChecker](https://github.com/[your-org]/AIChecker) - 评估工具和实验代码
