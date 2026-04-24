# 测试报告使用指南

这份文档面向仓库中的自动化测试报告体系，覆盖以下内容：

1. 如何生成报告
2. 报告会产出哪些文件
3. 每个报告页面包含什么内容
4. 排查问题时应该怎么看报告
5. 常见问题与排错建议

本文主要介绍以下四类主报告：

- `image_match`
- `button_color`
- `count_change`
- `progress_change`

另外，文档最后也会补充 `image_match` 的“模板法 vs 特征法”对比报告，它属于独立报告链路，不和上面四类主报告混用。

## 1. 报告体系概览

当前测试报告体系由三部分组成：

1. `pytest` 执行测试用例
2. `AIChecker/tests/conftest.py` 在测试执行过程中采集结果并写入历史 CSV
3. `scripts/testing/generate_test_report.py` 读取历史结果并生成 HTML 报告

也就是说，HTML 报告不是单纯从控制台输出拼出来的，而是基于“测试运行历史”二次加工后的结果。

这一套设计有几个直接好处：

- 可以重复生成 HTML，而不必每次都重跑测试
- 可以把多次运行结果串起来，形成时间线和修复轨迹
- 可以在报告里补充截图、错误摘要、历史标签等更适合排查的信息

## 2. 报告相关脚本

主报告的入口脚本位于 [scripts/testing](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing)：

- [run_image_match_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_image_match_pipeline.py)
- [run_button_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_button_pipeline.py)
- [run_count_change_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_count_change_pipeline.py)
- [run_progress_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_progress_pipeline.py)
- [generate_test_report.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/generate_test_report.py)

采集 pytest 结果的逻辑在：

- [conftest.py](/Users/drifter327/Code/AIChecker-Dataset/AIChecker/tests/conftest.py)

独立的模板法 vs 特征法对比报告入口在：

- [run_image_match_feature_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_image_match_feature_pipeline.py)

## 3. 报告会生成到哪里

### 3.1 主报告输出目录

每个 checker 都会生成自己独立的报告目录：

- `reports/image_match/`
- `reports/button_color/`
- `reports/count_change/`
- `reports/progress_change/`

每个目录下默认包含 3 份 HTML：

- `LATEST_SNAPSHOT.html`
  说明：本次最新运行的快照报告，最常看
- `HISTORY_TIMELINE.html`
  说明：历史时间线报告，适合看长期趋势、持续失败、已修复、波动用例
- `FAILURE_VIEW.html`
  说明：只看本次运行里有问题的 case，适合第一时间排查

### 3.2 历史数据目录

每个 checker 的历史数据保存在：

- `reports/history/<checker>/test_runs.csv`
- `reports/history/<checker>/test_case_results.csv`

其中：

- `test_runs.csv` 记录每次运行的元信息，例如 `run_id`、时间、分支、提交、备注、case 数量
- `test_case_results.csv` 记录每个 case 在每次运行中的结果、状态、错误信息、截图路径等

### 3.3 模板法 vs 特征法对比报告

这个报告和主报告是分开的，默认输出到：

- `reports/image_match_compare/TEMPLATE_VS_FEATURE.html`
- `reports/image_match_compare/template_vs_feature_results.csv`

## 4. 最推荐的生成方式

最推荐使用各 checker 对应的 pipeline 脚本。它们会帮你：

- 自动寻找虚拟环境
- 跑对应的 pytest 用例
- 调用报告生成器
- 在需要时带上备注信息

### 4.1 image_match

```bash
python3 scripts/testing/run_image_match_pipeline.py --always-generate-report
```

### 4.2 button_color

```bash
python3 scripts/testing/run_button_pipeline.py --always-generate-report
```

### 4.3 count_change

```bash
python3 scripts/testing/run_count_change_pipeline.py --always-generate-report
```

### 4.4 progress_change

```bash
python3 scripts/testing/run_progress_pipeline.py --always-generate-report
```

## 5. 常见生成场景

### 5.1 只跑一个文件或一个 case

以 `button_color` 为例：

```bash
python3 scripts/testing/run_button_pipeline.py \
  --test-target "AIChecker/tests/regression/test_button_cases.py" \
  --pytest-args='-q -k aiqiyi_3' \
  --note "single case verify"
```

常用场景：

- 只验证某个回归 case 是否修复
- 只想更新某一类问题的报告
- 想缩短本地排查时间

### 5.2 不跑 pytest，只重生成 HTML

适用于：

- 历史 CSV 已经有数据，只想重新生成页面
- 报告样式或展示逻辑刚改过，需要重新出 HTML

示例：

```bash
python3 scripts/testing/run_button_pipeline.py --skip-pytest --note "regenerate html only"
```

其它 checker 也同理，把脚本名替换掉即可。

### 5.3 pytest 失败了也要继续出报告

默认情况下，如果 pytest 返回非 0，pipeline 会直接停止，不生成报告。

如果你希望“就算失败也要出报告”，请加：

```bash
--always-generate-report
```

这在排查失败 case 时非常有用，因为你通常更想看到报告里的错误摘要和截图，而不是只拿到一个 pytest 退出码。

### 5.4 指定虚拟环境

如果自动发现虚拟环境失败，可以显式指定：

```bash
python3 scripts/testing/run_button_pipeline.py --venv-path AIChecker/.venv
```

所有主 pipeline 都支持这个参数。

## 6. 直接调用报告生成器

通常推荐走 pipeline，但在某些场景下，你可以直接调用报告生成器：

```bash
python3 scripts/testing/generate_test_report.py \
  --source pytest \
  --checker button_color \
  --note "manual regenerate"
```

### 6.1 参数说明

- `--checker`
  说明：指定要生成哪一类报告
  可选值：`image_match`、`button_color`、`count_change`、`progress_change`

- `--source`
  说明：指定报告的数据来源
  可选值：`pytest`、`checker`

- `--note`
  说明：写入本次生成备注，方便后续从历史中识别这次运行

- `--image-output-root`
  说明：仅 `image_match` 使用，用于指定调试图目录

### 6.2 `--source=pytest` 和 `--source=checker` 的区别

`--source=pytest`

- 从历史 CSV 里读取最近一次 pytest 结果
- 适合四类主报告
- 是最常用的模式

`--source=checker`

- 当前只支持 `image_match`
- 会直接读取 `testcase/image_match/jsons` 下的 case，并重新调用 checker
- 更适合离线验证 checker 本身，不依赖最近一次 pytest 会话

如果你不确定用哪个，优先用：

```bash
--source pytest
```

## 7. 报告里记录了哪些内容

报告不是只有“通过/失败”两个结论。当前主报告会尽量记录下列信息：

- 运行 ID
- 运行时间
- Git branch
- Git commit
- 备注信息
- App 名称
- Case ID
- case 文件路径
- 预期结果
- 实际结果
- 状态
- 相似度或变化比
- 阈值
- 预期框
- 实际框
- 错误信息
- 预览图路径

对于不同 checker，记录的图片略有不同：

### 7.1 image_match

通常包含：

- template 图
- target 图
- match result 图

### 7.2 button_color

通常包含：

- 原图 Before
- 原图 After
- 按钮裁剪图 Before
- 按钮裁剪图 After

### 7.3 count_change / progress_change

通常包含：

- 原图 Before
- 原图 After

## 8. 三类主 HTML 报告怎么理解

## 8.1 `LATEST_SNAPSHOT.html`

这是日常最常看的页面，用于回答：

- 这次整体结果怎么样
- 有没有新回归
- 哪些应用问题最多
- 具体哪个 case 出问题

当前页面通常包含以下板块：

### 1) 运行元信息

会展示：

- 运行 ID
- 运行时间
- 分支
- 提交
- 备注

适合在多人协作或连续多次回归时确认“这份报告到底是哪次跑出来的”。

### 2) 结果概览

会展示类似这些指标：

- 总用例数
- 一致率
- 问题数
- 新回归数量
- 已修复数量
- 最近波动数量

其中最值得优先关注的是：

- `问题数`
- `新回归`
- `最近波动`

### 3) 优先关注问题

这个板块会把本次有问题的 case 放到前面，并附带：

- App
- Case ID
- 历史标签
- 当前状态
- 最近一次通过时间
- 错误摘要
- 截图

它相当于“本次回归的重点问题列表”。

### 4) 按应用概览

这个板块会按 App 聚合，显示：

- 每个 App 的用例数
- 问题数
- 一致数
- 跳过数
- 问题率

适合快速判断：

- 问题是集中在单个 App 还是散落在多个 App
- 某个 App 是否整体不稳定

### 5) 快速筛选

快照页和失败页都支持筛选，通常可以按：

- App
- 状态
- 历史标签
- 关键字

建议当问题量比较大时，不要手动滚表，先用筛选把范围收窄。

### 6) 全量明细

这是所有 case 的完整表格。问题 case 会排在每个 App 的前面，通常用于：

- 二次确认某个问题的上下文
- 查看有问题 case 的预期/实际/阈值/坐标框
- 点开图片做人工比对

## 8.2 `FAILURE_VIEW.html`

这个页面只保留本次运行中的问题项，也就是：

- `不一致(MISMATCH)`
- `异常(ERROR)`

它最适合在以下场景下使用：

- 刚跑完回归，第一时间看坏掉的 case
- 和同事同步时，只想给对方看问题列表
- 需要快速定位哪些 case 需要重新验证

页面通常包含：

- 本次问题数
- 新回归数量
- 持续失败数量
- 问题明细表

问题明细表会展示：

- 预期 / 实际
- 历史标签
- 最近通过时间
- 结构化错误摘要
- 对应截图

如果你只关心“现在坏了什么”，优先看这个页面。

## 8.3 `HISTORY_TIMELINE.html`

这个页面用于回答：

- 哪些问题是长期存在的
- 哪些问题最近修好了
- 哪些 case 很不稳定

它通常包含几个重点板块：

### 1) 历史运行概览

例如：

- 历史运行次数
- 累计用例数
- 最新运行的问题数
- 持续失败用例数
- 波动用例数
- 已修复用例数

### 2) 持续失败用例

这类 case 通常优先级最高，因为它们不是偶发问题，而是已经连续多次失败。

### 3) 最近波动用例

如果一个 case 最近几次在“通过”和“失败”之间来回切换，一般说明：

- 阈值不稳定
- 样本本身质量不够稳定
- 外部依赖或环境有干扰

### 4) 用例修复追踪

这里通常能看到：

- 首次失败时间
- 首次修复通过时间
- 总运行次数
- 当前最新状态

适合在修复后回顾“问题出现到修好”的完整过程。

### 5) 最近运行明细

这是按时间倒序展示的轻量历史明细，适合回看最近几次跑测结果。

## 9. 报告中的状态和历史标签

### 9.1 状态含义

- `一致(MATCH)`
  说明：实际结果和预期一致

- `不一致(MISMATCH)`
  说明：实际结果和预期不一致

- `异常(ERROR)`
  说明：执行过程中抛异常，或者测试失败但无法归类为普通通过/失败

- `跳过(SKIPPED)`
  说明：pytest 将该 case 跳过了

- `未知(UNKNOWN)`
  说明：没有得到有效结果，通常与 skip 或缺依赖有关

- `预期未知(UNKNOWN_EXPECTED)`
  说明：case 没有给出明确 groundtruth，无法比较实际和预期

### 9.2 历史标签含义

当前报告里常见的历史标签包括：

- `新回归`
  说明：上次是好的，这次变坏了

- `持续失败`
  说明：上次就坏，这次还坏

- `首次失败`
  说明：历史上还没记录到前一次坏结果，这次第一次发现失败

- `首次异常`
  说明：这次第一次出现异常类型的问题

- `已修复`
  说明：上次坏了，这次恢复正常

- `波动中`
  说明：最近几次状态有切换，不够稳定

- `稳定通过`
  说明：最近都保持通过

- `首次通过`
  说明：第一次记录到通过结果

- `待确认`
  说明：无法从历史中稳定推断标签

## 10. 推荐的看报告顺序

如果你刚跑完一轮回归，建议按下面顺序看：

### 场景一：想快速知道“这次有没有新问题”

1. 打开 `FAILURE_VIEW.html`
2. 先看问题总数
3. 再看 `新回归`
4. 然后按 App / 历史标签 / 关键字筛选

### 场景二：想全面了解这次结果

1. 打开 `LATEST_SNAPSHOT.html`
2. 先看总体指标
3. 再看 `优先关注问题`
4. 再看 `按应用概览`
5. 最后进入全量明细确认截图和阈值

### 场景三：想追长期问题或验证修复效果

1. 打开 `HISTORY_TIMELINE.html`
2. 看 `持续失败用例`
3. 看 `最近波动用例`
4. 看 `用例修复追踪`

## 11. 常见问题

### 11.1 为什么没有生成报告

常见原因：

- pytest 失败且没有加 `--always-generate-report`
- 虚拟环境没找到
- 直接运行 `generate_test_report.py --source pytest` 时，历史 CSV 里还没有数据

建议排查：

1. 先看 pipeline 输出是否提示 `Skip report generation`
2. 若是 pytest 失败但仍需出报告，加 `--always-generate-report`
3. 若是环境问题，显式传 `--venv-path`

### 11.2 为什么报告里没有图

常见原因：

- 调试图目录不存在
- 历史 CSV 里记录的是旧机器上的绝对路径
- 某个 checker 本身就没有生成对应调试图

当前报告生成器会尽量自动修复旧路径，并回填当前仓库可定位到的本地图片，但前提是这些图片真实存在于仓库或 debug 目录中。

### 11.3 为什么只看到少量 case

常见原因：

- 本次只跑了单个文件或单个 `-k` 过滤条件
- 当前报告读取的是最近一次运行，而最近一次本身就是局部运行

如果要恢复成全量视角，请重新跑全量 pipeline。

### 11.4 为什么看到 `SKIPPED` 或 `UNKNOWN`

这通常意味着：

- case 缺少输入文件
- 依赖缺失
- 外部接口或鉴权失败
- 测试逻辑主动 `pytest.skip`

这类 case 需要先解决环境或输入问题，再看业务结果。

### 11.5 为什么 `count_change` 有时问题多，但错误信息不稳定

`count_change` 往往依赖外部模型或接口，因此可能受到以下因素影响：

- API 鉴权
- 模型返回不稳定
- 网络环境
- 输入截图质量

如果问题表现为“最近波动用例”，建议优先从这几个方向排查。

## 12. 模板法 vs 特征法对比报告

如果你要比较 `image_match` 的模板法和特征法效果，请使用：

```bash
python3 scripts/testing/run_image_match_feature_pipeline.py
```

它生成的是独立报告，不是主测试报告的一部分。

默认输出：

- `reports/image_match_compare/TEMPLATE_VS_FEATURE.html`
- `reports/image_match_compare/template_vs_feature_results.csv`

这个报告主要用于比较两种算法方法的正确性和差异，不用于替代 `LATEST_SNAPSHOT / FAILURE_VIEW / HISTORY_TIMELINE` 这三类主报告。

## 13. 推荐维护方式

如果你后续还会继续维护这套报告，建议遵循以下习惯：

- 日常回归优先走 pipeline，而不是手工拼命令
- 修改展示逻辑后，使用 `--skip-pytest` 重生成 HTML 验证页面
- 新增 checker 或字段时，同步更新 `conftest.py` 的写入逻辑和本指南
- 当报告目录结构调整时，优先更新这份文档，避免命令和路径过期

## 14. 相关文件索引

- [generate_test_report.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/generate_test_report.py)
- [run_image_match_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_image_match_pipeline.py)
- [run_button_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_button_pipeline.py)
- [run_count_change_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_count_change_pipeline.py)
- [run_progress_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_progress_pipeline.py)
- [run_image_match_feature_pipeline.py](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/run_image_match_feature_pipeline.py)
- [conftest.py](/Users/drifter327/Code/AIChecker-Dataset/AIChecker/tests/conftest.py)
- [image_match_report_usage.md](/Users/drifter327/Code/AIChecker-Dataset/scripts/testing/image_match_report_usage.md)
