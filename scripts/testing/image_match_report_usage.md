# Image Match 使用说明（简版）

更完整的测试报告文档见：

- [test_report_guide.md](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/test_report_guide.md)

这套流程做三件事：

1. 跑 `pytest`
2. 把结果写到历史 CSV
3. 生成 HTML 报告

## 最常用（一条命令）

在仓库根目录执行：

```bash
python3 scripts/testing/run_image_match_pipeline.py --always-generate-report
```

执行后会更新：

- `reports/LATEST_SNAPSHOT.html`（最新结果）
- `reports/HISTORY_TIMELINE.html`（历史趋势）
- `reports/FAILURE_VIEW.html`（只看不一致/异常）
- `reports/history/test_runs.csv`
- `reports/history/test_case_results.csv`

## 常用场景

### 1) 只跑一个 case

```bash
python3 scripts/testing/run_image_match_pipeline.py \
  --test-target "AIChecker/tests/regression/test_image_match_cases.py" \
  --pytest-args='-q -k antennapod_1' \
  --note "single case run"
```

### 2) 不跑测试，只重生成 HTML 报告

```bash
python3 scripts/testing/run_image_match_pipeline.py --skip-pytest --note "regenerate html"
```

### 3) pytest 有失败测试用例也要生成报告
如果不加这个参数的话，如果有测试用例没有通过，就会直接退出不生成测试报告。

```bash
python3 scripts/testing/run_image_match_pipeline.py --always-generate-report
```

### 4) 指定虚拟环境

```bash
python3 scripts/testing/run_image_match_pipeline.py --venv-path AIChecker/.venv
```

## 报告怎么看

- `LATEST_SNAPSHOT.html`：看本次整体结果和每个 case 详情（含图片）
- `HISTORY_TIMELINE.html`：看每个 case 从失败到修复的时间线
- `FAILURE_VIEW.html`：只看 `不一致(MISMATCH)` 和 `异常(ERROR)`，排查最快

新版报告补了几类更适合排查的信息：

- 快照页会优先展示 `新回归`、`持续失败`、`已修复`，不用先通读整张大表
- 快照页和失败页都支持按 `App / 状态 / 历史标签 / 关键字` 过滤
- 时间线页会额外标出 `持续失败用例` 和 `最近波动用例`
- 历史 CSV 里如果残留旧机器上的绝对路径，报告会尽量自动回填当前仓库里的本地截图

## 状态含义

- `一致(MATCH)`：实际结果和预期一致
- `不一致(MISMATCH)`：实际结果和预期不一致
- `异常(ERROR)`：执行过程报错
- `未知(UNKNOWN)`：没有拿到有效实际结果（例如依赖缺失）

## 常见问题

### 为什么是 `跳过(SKIPPED)` 或 `未知(UNKNOWN)`？

通常是环境缺依赖（例如 `cv2`）或用例被 skip。

### 为什么报告里没图？

检查目录 `AIChecker/tests/image_match_output` 下是否有：

- `template.png`
- `target.png`
- `match_result.png`

### 为什么只看到少量 case？

可能只跑了单个测试节点（`--test-target` 指定了某个 case）。重新跑全量即可。
