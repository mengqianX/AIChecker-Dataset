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

## 运行指南（从环境配置开始）

### 1) 环境要求

- Python `>=3.10`
- macOS / Linux（Windows 请将激活命令改为对应 PowerShell/CMD 版本）

### 2) 克隆仓库

```bash
git clone <dataset-repo-url>
cd AIChecker-Dataset
```

### 3) 创建并激活虚拟环境

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Windows CMD:

```bat
py -3 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -U pip
```

### 4) 安装依赖

仓库内的 Python 包位于 `AIChecker/`，推荐开发安装：

```bash
pip install -e "./AIChecker[dev]"

```

### 5) 配置环境变量（API Key 等）

```bash
cp .env.example .env
```

然后编辑 `.env`，至少配置你要使用后端对应的 key，例如：

- `DASHSCOPE_API_KEY`（Qwen）
- `UI_TARS_BASE_URL` + `UI_TARS_API_KEY`（UI-TARS）
- `MAI_UI_BASE_URL` + `MAI_UI_API_KEY`（MAI-UI）

> 测试会自动尝试加载仓库根目录的 `.env`。

### 6) 运行按钮颜色回归测试

在仓库根目录执行（Windows / macOS / Linux 通用）：

```bash
python -m pytest AIChecker/tests/regression/test_button_cases.py -q
```

可选参数（通过环境变量控制）：

- `BUTTON_COLOR_PROFILE`：`default` / `strict` / `robust`
- `BUTTON_COLOR_MODE`：`hybrid` / `pure_segmentation`

示例（Windows PowerShell 写法）：

```powershell
$env:BUTTON_COLOR_PROFILE="robust"
$env:BUTTON_COLOR_MODE="hybrid"
python -m pytest AIChecker/tests/regression/test_button_cases.py -q
```

示例（macOS / Linux 写法）：

```bash
BUTTON_COLOR_PROFILE=robust BUTTON_COLOR_MODE=hybrid \
python -m pytest AIChecker/tests/regression/test_button_cases.py -q
```

### 7) 一键执行测试并生成报告（推荐）

```bash
python scripts/testing/run_button_pipeline.py \
  --test-target AIChecker/tests/regression/test_button_cases.py \
  --button-profile default \
  --button-mode hybrid
```

运行完成后，可在 `AIChecker/reports/` 和 `AIChecker/reports/history/` 查看测试记录与汇总结果。

## 数据集独立使用

1. 读取 `jsons/` 中配置，定位按钮区域。
2. 对 `screens/` 截图裁剪并比较颜色变化。
3. 将检测输出与 `expected_passed`（或兼容字段 `label`）比较，评估算法准确率。

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
