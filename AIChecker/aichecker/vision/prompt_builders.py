"""Prompt 构造器与版本化提示词加载。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class PromptPack:
    """一次模型调用需要的 system/user prompt。"""

    selected_prompt_type: str
    task_intent: str
    system_prompt: str
    user_prompt: str


class _SafeFormatDict(dict[str, Any]):
    """格式化时保留未知占位符，避免 KeyError。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _build_general_prompt(context: dict[str, Any]) -> PromptPack:
    """通用检测提示词。"""
    task_intent = "检测当前片段是否存在视觉或交互异常。"
    system_prompt = (
        "你是资深 GUI 自动化测试专家。"
        "你将收到交互前后的两张截图和测试意图。"
        "请严格输出 JSON，且只输出 JSON，不要包含任何额外文本。"
        "JSON 必须包含字段：bug_detected(bool)、reason(str)。"
    )
    user_prompt = (
        f"任务类型：{context['task_type']}\n"
        f"测试意图：{task_intent}\n"
        f"片段区间：{context['before_timestamp_sec']:.2f}s -> {context['after_timestamp_sec']:.2f}s\n"
        "请对比两张图判断是否存在视觉或交互结果相关的 Bug。"
        "若存在明显异常（例如未跳转、错误弹窗、布局错乱、白屏、关键控件消失），"
        "bug_detected=true；否则为 false。"
    )
    return PromptPack(
        selected_prompt_type="general",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_loading_prompt(context: dict[str, Any]) -> PromptPack:
    """长时间加载检测提示词（与 no_response 分离）。"""
    task_intent = "检测页面是否卡在异常长时间加载态（转圈/骨架屏/进度条长期不结束）。"
    system_prompt = (
        "你是资深移动端 GUI 质量专家，专门识别异常长时间加载。"
        "你将看到片段起始帧与结束帧。"
        "请严格输出 JSON，且只输出 JSON，不要包含任何额外文本。"
        "JSON 必须包含字段："
        "bug_detected(bool)、reason(str)、decision_basis(str)、"
        "anomaly_type(str: long_loading|black_screen|white_screen|garbled_screen|none|unknown)。"
        "本任务不判断 no_response；无响应由独立检测项处理。"
    )
    user_prompt = (
        "任务类型：long_loading（长时间加载检测）\n"
        f"测试意图：{task_intent}\n"
        f"片段区间：{context['before_timestamp_sec']:.2f}s -> {context['after_timestamp_sec']:.2f}s\n"
        "图1=片段起始，图2=片段结束。\n"
        "\n"
        "判定规则：\n"
        "1) long_loading（bug_detected=true）：图2仍能明确看到转圈、骨架屏、进度条、加载中文案等加载态，且未见加载结果。\n"
        "2) black_screen / white_screen / garbled_screen：按字面明显异常选择。\n"
        "3) none（bug_detected=false）：图2已离开加载态，或从未出现加载态。\n"
        "4) 证据不足：anomaly_type=unknown，bug_detected=false。\n"
        "\n"
        "不要输出 no_response / load_failed。"
        "请在 decision_basis 写明图2是否仍处于加载动效/加载文案。"
    )
    return PromptPack(
        selected_prompt_type="loading",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_no_response_prompt(context: dict[str, Any]) -> PromptPack:
    """无响应检测提示词（与 long_loading 分离）。"""
    task_intent = "检测操作后目标区域/页面是否出现有效响应；若无稳定新状态则判无响应。"
    system_prompt = (
        "你是资深移动端 GUI 质量专家，专门识别点击/操作后无响应。"
        "你将看到片段起始帧与结束帧（可能是全页或裁剪控件区）。"
        "请严格输出 JSON，且只输出 JSON，不要包含任何额外文本。"
        "JSON 必须包含字段："
        "bug_detected(bool)、has_effective_response(bool)、reason(str)、decision_basis(str)、"
        "anomaly_type(str: no_response|none|unknown)。"
        "本任务不判断 long_loading；长时间加载由独立检测项处理。"
    )
    user_prompt = (
        "任务类型：no_response（无响应检测）\n"
        f"测试意图：{task_intent}\n"
        f"片段区间：{context['before_timestamp_sec']:.2f}s -> {context['after_timestamp_sec']:.2f}s\n"
        "图1=操作前（或片段起始），图2=片段结束。\n"
        "\n"
        "只回答一件事：图2相对图1是否形成了操作带来的稳定新状态。\n"
        "1) 有有效响应：has_effective_response=true, bug_detected=false, anomaly_type=none\n"
        "   - 页面跳转、列表/正文更新、按钮/开关切换后的新外观并保持、弹层出现/关闭等。\n"
        "   - 操作成功后的结果页即使静止，也算已响应。\n"
        "2) 无响应：has_effective_response=false, bug_detected=true, anomaly_type=no_response\n"
        "   - 图2相对图1几乎同一状态，看不出稳定新结果；\n"
        "   - 仅有按下高亮/涟漪/闪一下又回到原样，不算有效响应。\n"
        "3) 证据不足：has_effective_response 可省略语义，anomaly_type=unknown, bug_detected=false。\n"
        "\n"
        "禁止输出 long_loading。即使看见加载动效，本任务也只根据“有无稳定业务结果”判断；"
        "加载动效本身既不能单独证明已响应，也不要改报成 long_loading。\n"
        "请在 decision_basis 写明图2相对图1是否形成稳定新状态。"
    )
    return PromptPack(
        selected_prompt_type="no_response",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def build_loading_failure_probe_prompt(context: dict[str, Any]) -> PromptPack:
    """loading 失败文案探测提示词（用于 detector 内部子步骤）。"""
    task_intent = "观察候选帧是否出现加载/请求失败提示，以及后帧是否已回到可用业务界面。"
    system_prompt = (
        "你是移动端 GUI 观察员，只输出结构化观察，不做最终 bug 判定。"
        "请严格输出 JSON，且只输出 JSON，不要输出空 JSON。"
        "JSON 必须包含字段："
        "failure_text_visible(bool), evidence_text(str), page_recovered(bool), visual_state(str), "
        "confidence(number,0~1), reason(str)。"
        "visual_state 只能取 normal|blank|skeleton|black|loading|unknown。"
        "不要输出 load_failed、failure_type、anomaly_type 或 bug_detected。"
    )
    user_prompt = (
        "任务类型：loading_failure_probe\n"
        f"任务意图：{task_intent}\n"
        f"片段区间：{context['before_timestamp_sec']:.2f}s -> {context['after_timestamp_sec']:.2f}s\n"
        "图1=候选前帧，图2=候选帧，图3=候选后帧。\n"
        "\n"
        "按主文案语义判断（有弹窗/有重试按钮本身不构成失败）：\n"
        "- failure_text_visible=true：主语义是失败了"
        "（网络/服务/请求/加载失败、超时、出错、无法连接、出了点问题等）；"
        "evidence_text 填失败原文。\n"
        "- failure_text_visible=false：主语义不是失败"
        "（没搜到/暂无数据、成就运营、聊天、普通确认等）；evidence_text=\"\"。\n"
        "\n"
        "对照：无法连接/服务异常/出了点问题 → true；"
        "没有符合条件/还没有内容（即使有重试）→ false。\n"
        "\n"
        "page_recovered：仅当图3主内容已是可用业务界面，且失败主文案不再占据主内容时为 true。"
        "若图2或图3仍显示失败主文案，page_recovered 必须为 false"
        "（标题栏/Tab/底部导航不算已恢复）。\n"
        "正常空结果页（没有符合条件/还没有内容）算已恢复，page_recovered=true。\n"
        "visual_state 描述外观；reason 一句话说明主语义。"
    )
    return PromptPack(
        selected_prompt_type="loading_failure_probe",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_toast_prompt(context: dict[str, Any]) -> PromptPack:
    """Toast 提示词（代码内置，不依赖外部 JSON 文件）。"""
    del context
    task_intent = "检测动作触发后的 toast 文案是否与预期语义一致。"
    system_prompt = (
        "你是移动端UI测试助手。任务是判断候选时刻是否出现“瞬时toast/snackbar”，并校验其文案语义是否符合动作。"
        "请严格按顺序执行并遵守时序约束：\n"
        "1) 先判断 toast_visible；\n"
        "2) 仅当 toast_visible=true 时再提取 toast_text；\n"
        "3) action_semantic 只能依据图1(动作前)与图2(候选)推断，禁止根据图3或toast_text反推动作；\n"
        "4) inferred_expected_toast_text 只能依据 action_semantic 生成，不能直接照搬 toast_text；\n"
        "5) 若图1与图2仍是同一操作上下文(如同一弹窗/同一表单)，优先判定为该上下文对应动作，不得被图3中的toast语义牵引改判；\n"
        "6) 若动作属于低可观测手势（如左右滑动条目）且仅靠图1+图2无法稳定判断动作语义，必须输出 action_semantic=\"unknown\"，并将 expectation_met=null、reverse_inference_risk=\"high\"。\n"
        "固定页面文案、列表内容、标题栏、底部统计不算toast。若 toast_visible=false，必须输出 toast_text=\"\"、inferred_expected_toast_text=\"\"、expectation_met=false。"
        "请只输出 JSON，不要额外文本。"
        "JSON 必须包含字段：toast_visible(bool), toast_text(str), action_semantic(str), inferred_expected_toast_text(str), expectation_met(bool|null), reverse_inference_risk(str), action_evidence_from_frame12(str), toast_evidence_from_frame23(str), confidence(number, 0~1), reason(str)。"
        "其中 reverse_inference_risk 只能是 low 或 high。action_evidence_from_frame12 必须只引用图1和图2可见证据；toast_evidence_from_frame23 只引用图2和图3可见证据。reason 限制为一句话且不超过40个汉字。"
    )
    user_prompt = (
        "任务意图：{task_intent}\n"
        "候选帧时间：{candidate_timestamp_sec:.2f}s\n"
        "测试关键词提示（可选）：{keywords_text}\n"
        "前处理摘要：{preprocess_summary}\n"
        "前处理证据(JSON, 精简)：{preprocess_structured_json}\n"
        "请基于三张完整帧判断是否出现toast：图1=动作前完整帧，图2=候选完整帧，图3=候选后完整帧。\n"
        "关键规则：先由图1+图2确定 action_semantic，再判断图2/图3中的toast是否与该动作匹配；禁止使用图3的toast文本反推动作语义。若动作可观测性不足（典型是滑动手势），请输出 action_semantic=unknown 且 expectation_met=null。"
    )
    return PromptPack(
        selected_prompt_type="toast",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def build_toast_prompt(
    prompt_version: str = "current",
    prompts_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> PromptPack:
    """构建 toast 提示词（兼容接口，内部使用代码内置模板）。"""
    del prompt_version, prompts_dir
    if isinstance(logger, logging.Logger):
        logger.info("toast prompt loaded: source=inline")
    return build_prompt_for_type(
        task_type="toast",
        context={},
    )


def render_toast_user_prompt(prompt_pack: PromptPack, context: dict[str, Any]) -> str:
    """将上下文填充到 toast user prompt 模板。"""
    return prompt_pack.user_prompt.format_map(_SafeFormatDict(context))


def _build_count_change_prompt(context: dict[str, Any]) -> PromptPack:
    """数量变化检测提示词。"""
    task_intent = "检测目标控件触发后，其语义关联的数量指标是否按预期变化。"
    metric_hints = context.get("metric_hints")
    hints_text = "、".join(metric_hints) if metric_hints else "无"
    control_name = context.get("control_name_hint") or "未提供"
    preprocess_structured = context.get("preprocess_structured")
    preprocess_structured_text = (
        json.dumps(preprocess_structured, ensure_ascii=False) if preprocess_structured else "{}"
    )
    system_prompt = (
        "你是资深 GUI 自动化测试专家，擅长控件语义识别与跨区域数字关联。"
        "你将看到前后页面截图以及控件 bounds 信息。"
        "请识别控件语义，并找到与其逻辑关联的数字指标（可不在控件附近）。"
        "请严格输出 JSON，且只输出 JSON。"
        "JSON 必须包含字段："
        "semantic_target(str), linked_metric(str), before_value(int|null), after_value(int|null), "
        "value_changed(bool), change_direction(str), expectation_met(bool), confidence(number), reason(str)。"
    )
    user_prompt = (
        f"任务意图：{task_intent}\n"
        f"期望变化规则：{context['expected_change']}（increase/decrease/any_change/no_change/any）\n"
        f"控件名称提示：{control_name}\n"
        f"指标关键词提示：{hints_text}\n"
        f"控件 bounds：x={context['control_bounds_x']}, y={context['control_bounds_y']}, "
        f"width={context['control_bounds_width']}, height={context['control_bounds_height']}\n"
        f"前处理摘要：{context['preprocess_summary']}\n"
        f"前处理结构化证据(JSON)：{preprocess_structured_text}\n"
        "请先识别该控件语义，再关联最相关数字指标，判断前后是否变化及方向。"
        "请优先使用上述证据来定位需要关注的区域，但最终结论以图像事实为准。"
    )
    return PromptPack(
        selected_prompt_type="count_change",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_list_refresh_prompt(context: dict[str, Any]) -> PromptPack:
    """列表刷新检测提示词。"""
    task_intent = "检测控件触发后，目标内容列表区域是否发生刷新变化。"
    preprocess_structured = context.get("preprocess_structured")
    preprocess_structured_text = (
        json.dumps(preprocess_structured, ensure_ascii=False) if preprocess_structured else "{}"
    )
    system_prompt = (
        "你是资深 GUI 自动化测试专家，擅长判断内容列表是否在交互后完成刷新。"
        "你将看到前后页面截图、可选目标区域 bounds 以及列表区域变化证据。"
        "请严格输出 JSON，且只输出 JSON。"
        "JSON 必须包含字段："
        "list_refreshed(bool), still_loading(bool), target_region(str), "
        "expectation_met(bool), confidence(number), reason(str)。"
    )
    user_prompt = (
        f"任务意图：{task_intent}\n"
        f"期望结果：expected_list_refresh={context['expected_list_refresh']}\n"
        f"自然语言预期：{context.get('expected_result_text') or '未提供'}\n"
        f"控件名称提示：{context.get('control_name_hint') or '未提供'}\n"
        f"目标区域 bounds：x={context['target_bounds_x']}, y={context['target_bounds_y']}, "
        f"width={context['target_bounds_width']}, height={context['target_bounds_height']}\n"
        f"前处理摘要：{context.get('preprocess_summary') or '无'}\n"
        f"前处理结构化证据(JSON)：{preprocess_structured_text}\n"
        "请重点判断控件响应后，列表内容是否出现可见更新（如条目顺序、文本、封面、时间戳、计数等变化）。"
        "同时检查后页是否仍处于 loading/buffering/骨架屏/加载中状态；若后页仍在加载中，still_loading=true，"
        "通常不应判定为刷新完成。"
        "若仅出现轻微动画或非列表区域变化，不应判定为列表刷新。"
    )
    return PromptPack(
        selected_prompt_type="list_refresh",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


PROMPT_BUILDERS: dict[str, Callable[[dict[str, Any]], PromptPack]] = {
    "general": _build_general_prompt,
    "loading": _build_loading_prompt,
    "no_response": _build_no_response_prompt,
    "toast": _build_toast_prompt,
    "count_change": _build_count_change_prompt,
    "list_refresh": _build_list_refresh_prompt,
}


def build_prompt_for_type(task_type: str, context: dict[str, Any]) -> PromptPack:
    """
    按 task_type 构造提示词，未知类型自动回退到 general。

    Args:
        task_type: 任务类型，例如 loading。
        context: 上下文变量。

    Returns:
        PromptPack: 构造完成的提示词对象。
    """
    normalized = (task_type or "").strip().lower()
    builder = PROMPT_BUILDERS.get(normalized, PROMPT_BUILDERS["general"])
    return builder(context)


def build_count_change_prompt(context: dict[str, Any]) -> PromptPack:
    """构建 count_change 提示词（兼容接口，内部走统一分发）。"""
    return build_prompt_for_type(task_type="count_change", context=context)
