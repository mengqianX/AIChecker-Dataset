# Count Change 测试用例目录

本目录包含数量变化检测（Count Change）的测试用例。

## 目录结构

```
count_change/
├── jsons/                    # 测试用例JSON配置文件
│   ├── sample.json           # 示例测试用例
│   ├── Android/              # Android平台测试用例
│   │   ├── sample_like.json
│   │   ├── sample_comment.json
│   │   └── sample_cart.json
│   ├── HarmonyOS/            # HarmonyOS平台测试用例
│   │   ├── sample_like.json
│   │   ├── sample_comment.json
│   │   └── sample_follow.json
│   └── README.md             # JSON文件说明
└── screens/                  # 截图目录
    ├── sample/               # 示例截图
    ├── Android/sample/       # Android平台截图
    ├── HarmonyOS/sample/     # HarmonyOS平台截图
    └── README.md             # 截图目录说明
```

## 快速开始

### 1. 准备测试数据

1. **替换占位符数据**：
   - 编辑 `jsons/` 目录下的JSON文件
   - 将 `bounds` 从 `[0, 0, 0, 0]` 替换为实际的目标控件坐标
   - 将截图路径替换为实际的截图文件路径

2. **添加截图文件**：
   - 将操作前后的截图放到 `screens/` 对应的子目录下
   - 确保截图文件名与JSON中的路径一致

### 2. 运行测试

```bash
# 设置API密钥（使用 qwen 时）
export DASHSCOPE_API_KEY="your-api-key-here"

# 运行所有测试用例
cd AIChecker
pytest tests/testcase_count_change_test.py -v

# 运行单个测试用例
pytest tests/testcase_count_change_test.py::test_sample_count_change -v
```

### 3. 调试：切换后端（qwen / ui-tars）

无需修改 JSON 文件，通过环境变量即可切换后端，方便对比 qwen 与 UI-TARS 效果。

**使用 qwen（默认）**

```bash
cd AIChecker
export DASHSCOPE_API_KEY="your-api-key"
pytest -s ./tests/testcase_count_change_test.py -v
```

**使用 UI-TARS（本地 vLLM）**

```bash
cd AIChecker
export COUNT_CHANGE_BACKEND="ui-tars"
export UI_TARS_BASE_URL="http://localhost:8000/v1"   # 本地 vLLM 地址
export UI_TARS_MODEL="ui-tars"                        # 或实际模型名
# 本地 vLLM 通常不需要 api_key

pytest -s ./tests/testcase_count_change_test.py -v
```

**使用 UI-TARS（HuggingFace Inference Endpoints）**

```bash
cd AIChecker
export COUNT_CHANGE_BACKEND="ui-tars"
export UI_TARS_BASE_URL="https://xxx.inference.endpoints.huggingface.cloud"
export UI_TARS_API_KEY="hf_xxx"
export UI_TARS_MODEL="ByteDance-Seed/UI-TARS-7B-DPO"

pytest -s ./tests/testcase_count_change_test.py -v
```

**环境变量说明**

| 变量 | 说明 |
|------|------|
| `COUNT_CHANGE_BACKEND` | 覆盖后端，`qwen` 或 `ui-tars`；不设置时使用 JSON 中的 `backend`（默认 qwen） |
| `UI_TARS_BASE_URL` | UI-TARS API 地址，如 `http://localhost:8000/v1` |
| `UI_TARS_MODEL` | UI-TARS 模型名，如 `ui-tars` 或 `ByteDance-Seed/UI-TARS-7B-DPO` |
| `UI_TARS_API_KEY` | UI-TARS API 密钥（本地 vLLM 可用 `dummy`） |
| `HF_TOKEN` | HuggingFace Token，可作为 `UI_TARS_API_KEY` 的替代 |

### 4. 使用Python API

```python
from aichecker import check_count_change
import json
from pathlib import Path

# 加载测试用例
json_path = Path("testcase/count_change/jsons/sample.json")
with open(json_path, "r") as f:
    payload = json.load(f)

# 解析路径（相对于JSON文件所在目录）
def resolve_path(json_path, rel_path):
    return str((json_path.parent / rel_path).resolve())

payload["screenshot_a"] = resolve_path(json_path, payload["screenshot_a"])
payload["screenshot_b"] = resolve_path(json_path, payload["screenshot_b"])

# 执行检测
result = check_count_change(payload)

# 查看结果
print(f"检测结果: {'✅ 通过' if result.passed else '❌ 未通过'}")
print(f"判定依据: {result.basis}")
print(f"控件信息: {result.control_info}")
```

## JSON字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 测试类型，固定为 `"count_change"` |
| `screenshot_a` | string | 是 | 操作前的截图路径（相对于JSON文件） |
| `screenshot_b` | string | 是 | 操作后的截图路径（相对于JSON文件） |
| `bounds` | array | 是 | 目标控件的bounds `[left, top, right, bottom]` |
| `expected_passed` | boolean | 是 | **人工标注的groundtruth**（true=应检测到变化，false=不应检测到变化） |
| `label` | string | 否 | 标签，`"pass"` 或 `"fail"`（向后兼容） |
| `description` | string | 否 | 测试用例描述 |
| `api_key` | string | 否 | API密钥（可选，优先使用环境变量） |
| `model` | string | 否 | 模型名称（默认：`qwen-vl-max`） |
| `base_url` | string | 否 | API端点URL（可选） |

## 测试场景

### 场景1：点赞数变化
- **描述**：检测点击点赞按钮后，点赞数是否增加
- **文件**：`sample_like.json`
- **预期**：`expected_passed: true`

### 场景2：评论数变化
- **描述**：检测点击评论按钮后，评论数是否变化
- **文件**：`sample_comment.json`
- **预期**：`expected_passed: true`

### 场景3：购物车数量变化
- **描述**：检测添加商品到购物车后，购物车商品数量是否增加
- **文件**：`sample_cart.json`（仅Android）
- **预期**：`expected_passed: true`

### 场景4：关注数变化
- **描述**：检测点击关注按钮后，关注数是否变化
- **文件**：`sample_follow.json`（仅HarmonyOS）
- **预期**：`expected_passed: true`

## 注意事项

1. **Groundtruth标注（重要）**：
   - `expected_passed` **必须由人工标注**，不能依赖模型判断
   - 标注人员需要查看操作前后的截图，人工判断数量是否真的发生了变化
   - 这是评估模型准确性的标准答案（groundtruth）
   - 测试脚本会比较模型的检测结果与groundtruth，验证模型准确性

2. **占位符数据**：当前测试用例中的 `bounds` 和截图路径都是占位符，需要替换为实际数据

3. **API密钥**：需要配置通义千问API密钥（`DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`）

4. **截图要求**：
   - 格式：PNG、JPG等常见格式
   - 命名：`{场景}_{before/after}.{扩展名}`
   - 内容：操作前后的页面状态

5. **测试流程**：
   - 步骤1：人工查看截图，标注 `expected_passed`（groundtruth）
   - 步骤2：模型检测数量变化，返回 `passed`（模型判断）
   - 步骤3：测试脚本比较两者，如果不一致则测试失败

6. **测试跳过**：
   - 如果 `bounds` 为 `[0, 0, 0, 0]`，测试会自动跳过（占位符）
   - 如果截图文件不存在，测试会自动跳过
   - 如果API密钥未配置，测试会自动跳过
   - 如果缺少 `expected_passed` 字段，测试会失败（必须有groundtruth）

## 相关文件

- `AIChecker/aichecker/checkers/count_change_checker.py`：检测器实现
- `AIChecker/tests/testcase_count_change_test.py`：测试脚本
- `AIChecker/docs/COUNT_CHANGE_CHECKER_README.md`：详细文档
