# AIChecker

本仓库是一套**移动应用功能正确性检测框架**：用符号化规则（CV / 模板匹配）和多模态大模型（VLM）判定 UI 操作后的结果是否符合预期，并附带对应测试用例。

框架代码在 `AIChecker/`，用例在 `testcase/`。

## 目录结构

```
.
├── AIChecker/                 # 检测框架
│   ├── aichecker/
│   │   ├── checkers/          # 符号化检测：按钮颜色、开关、数量、进度条、图像匹配
│   │   └── vision/            # 多模态检测：Toast、加载、黑白屏、列表刷新、视频播放等
│   └── tests/                 # pytest（unit / regression）
├── testcase/                  # 按任务类型划分的用例（JSON + 截图 / 视频）
├── scripts/testing/           # 回归测试与报告脚本
├── reports/                   # 测试运行记录与 HTML 报告（本地生成）
└── .env.example               # VLM 等配置模板
```

`testcase/` 下常见类别：


| 目录                                                                                                                                           | 说明         |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `button_color_change` / `toggle_state` / `count_change` / `progress_bar_change` / `image_match`                                              | 符号化检测      |
| `toast` / `long_loading` / `page_load_failure` / `black_white_screen` / `no_response` / `content_list_refresh` / `video_play` / `video_peek` | 多模态 / 视频检测 |


每类用例一般包含 JSON 配置（`jsons/` 或 `json/`）以及对应截图或视频。JSON 里的 `expected_passed` 是 groundtruth。

## 如何使用

环境：Python `>= 3.10`。依赖装在 `AIChecker/.venv`。

```bash
python3 -m venv AIChecker/.venv
AIChecker/.venv/bin/python -m pip install -U pip
AIChecker/.venv/bin/python -m pip install -e "./AIChecker[dev]"
cp .env.example .env
```

编辑 `.env`，配置 OpenAI 兼容的 VLM（多模态检测需要）。最少三项：

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1
```

也可用 `AICHECKER_VLM_API_KEY` / `AICHECKER_VLM_MODEL` / `AICHECKER_VLM_BASE_URL` 覆盖。Qwen、UI-TARS、MAI-UI 的写法见 `.env.example`。

**符号化检测（代码调用）**

```python
from aichecker.checkers import check_button_color, check_toggle, check_image_match
```

**多模态检测（CLI）**

```bash
AIChecker/.venv/bin/python -m aichecker.vision.cli \
  --input-file path/to/case.json \
  --output-dir outputs
```

JSON 中指定 `task_type`、视频或前后截图等；pytest 会自动加载仓库根目录的 `.env`。

## 测试

一律从仓库根目录执行，解释器固定为 `AIChecker/.venv/bin/python`。

```bash
# 符号化回归
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_button_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_toggle_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_count_change_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_progress_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_image_match_cases.py

# 多模态回归（需要 .env 里的 VLM）
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_toast_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_long_loading_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_page_load_failure_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_black_white_screen_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_no_response_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_list_refresh_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_video_play_cases.py
AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_seek_playback_cases.py
```

也可以用 pipeline 跑完测试并生成报告（输出在 `reports/`）：

```bash
AIChecker/.venv/bin/python scripts/testing/run_button_pipeline.py
AIChecker/.venv/bin/python scripts/testing/run_image_match_pipeline.py
AIChecker/.venv/bin/python scripts/testing/run_toast_pipeline.py
```



### 测试相关配置


| 变量                                                    | 作用                                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------------ |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | VLM 连接（也可用 `AICHECKER_VLM_*`）                                      |
| `TESTAGENT_ROOT`                                      | 用例根目录，默认本仓库；vision 回归读 `{TESTAGENT_ROOT}/testcase/<category>/json` |
| `BUTTON_COLOR_PROFILE`                                | `default` / `strict` / `robust`                                    |
| `BUTTON_COLOR_MODE`                                   | `hybrid` / `pure_segmentation`                                     |
| `AICHECKER_IMAGE_MATCH_TIMING_CSV`                    | 可选，image match 耗时 CSV 路径                                           |
| `AICHECKER_IMAGE_MATCH_TIMING_VERBOSE`                | `true` 时在控制台打印耗时                                                   |


按钮颜色示例：

```bash
BUTTON_COLOR_PROFILE=robust BUTTON_COLOR_MODE=hybrid \
  AIChecker/.venv/bin/python -m pytest -q AIChecker/tests/regression/test_button_cases.py
```

