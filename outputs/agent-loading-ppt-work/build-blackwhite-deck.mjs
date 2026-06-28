import fs from "node:fs/promises";
import path from "node:path";
import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  padSlideNumber,
  saveBlobToFile,
} from "/Users/xmq/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const root = "/Users/xmq/GitHubRepo/AIChecker-Dataset";
const workspace = path.join(root, "outputs/agent-loading-ppt-work");
const previewDir = path.join(workspace, "preview");
const outputPptx = path.join(root, "outputs/agent-loading-detection-report.pptx");
const W = 1280;
const H = 720;

const COLORS = {
  black: "#000000",
  white: "#FFFFFF",
  gray: "#666666",
  light: "#F4F4F4",
  mid: "#D8D8D8",
};

function addTitle(ctx, slide, title, subtitle = "") {
  ctx.addText(slide, {
    text: title,
    x: 64,
    y: 42,
    w: 920,
    h: 48,
    fontSize: 31,
    bold: true,
    color: COLORS.black,
    typeface: "Arial",
  });
  if (subtitle) {
    ctx.addText(slide, {
      text: subtitle,
      x: 66,
      y: 91,
      w: 1000,
      h: 30,
      fontSize: 15,
      color: COLORS.gray,
      typeface: "Arial",
    });
  }
  ctx.addShape(slide, {
    x: 64,
    y: 124,
    w: 1152,
    h: 1.4,
    fill: COLORS.black,
    line: ctx.line(COLORS.black, 0),
  });
}

function addFooter(ctx, slide, n) {
  ctx.addText(slide, {
    text: String(n).padStart(2, "0"),
    x: 1168,
    y: 668,
    w: 48,
    h: 22,
    fontSize: 13,
    color: COLORS.gray,
    align: "right",
    typeface: "Arial",
  });
}

function addBullets(ctx, slide, bullets, x, y, w, h, opts = {}) {
  const fontSize = opts.fontSize ?? 22;
  const lineH = opts.lineH ?? 41;
  bullets.forEach((text, i) => {
    const yy = y + i * lineH;
    ctx.addText(slide, {
      text: "•",
      x,
      y: yy + 1,
      w: 24,
      h: 26,
      fontSize,
      bold: true,
      color: COLORS.black,
      typeface: "Arial",
    });
    ctx.addText(slide, {
      text,
      x: x + 34,
      y: yy,
      w,
      h: Math.max(30, h / Math.max(1, bullets.length)),
      fontSize,
      color: COLORS.black,
      typeface: "Arial",
      insets: { left: 0, right: 8, top: 0, bottom: 0 },
    });
  });
}

function addBox(ctx, slide, { text, x, y, w, h, fontSize = 18, bold = false, fill = COLORS.white }) {
  const box = ctx.addShape(slide, {
    x,
    y,
    w,
    h,
    fill,
    line: ctx.line(COLORS.black, 1.3),
  });
  box.text = text;
  box.text.fontSize = fontSize;
  box.text.color = COLORS.black;
  box.text.bold = bold;
  box.text.typeface = "Arial";
  box.text.alignment = "center";
  box.text.verticalAlignment = "middle";
  box.text.insets = { left: 12, right: 12, top: 8, bottom: 8 };
  return box;
}

function addLabel(ctx, slide, text, x, y, w, h, fontSize = 18, bold = false) {
  return ctx.addText(slide, {
    text,
    x,
    y,
    w,
    h,
    fontSize,
    bold,
    color: COLORS.black,
    typeface: "Arial",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

function addTable(ctx, slide, headers, rows, x, y, colW, rowH = 52, fontSize = 17) {
  let yy = y;
  headers.forEach((h, i) => {
    const cell = ctx.addShape(slide, {
      x: x + colW.slice(0, i).reduce((a, b) => a + b, 0),
      y: yy,
      w: colW[i],
      h: rowH,
      fill: COLORS.light,
      line: ctx.line(COLORS.black, 1),
    });
    cell.text = h;
    cell.text.fontSize = fontSize;
    cell.text.bold = true;
    cell.text.typeface = "Arial";
    cell.text.alignment = "center";
    cell.text.verticalAlignment = "middle";
    cell.text.insets = { left: 8, right: 8, top: 6, bottom: 6 };
  });
  yy += rowH;
  rows.forEach((row) => {
    row.forEach((value, i) => {
      const cell = ctx.addShape(slide, {
        x: x + colW.slice(0, i).reduce((a, b) => a + b, 0),
        y: yy,
        w: colW[i],
        h: rowH,
        fill: COLORS.white,
        line: ctx.line(COLORS.black, 1),
      });
      cell.text = value;
      cell.text.fontSize = fontSize - 1;
      cell.text.typeface = "Arial";
      cell.text.alignment = i === 0 ? "left" : "left";
      cell.text.verticalAlignment = "middle";
      cell.text.insets = { left: 12, right: 10, top: 6, bottom: 6 };
    });
    yy += rowH;
  });
}

function addDownArrow(ctx, slide, x, y) {
  ctx.addText(slide, {
    text: "↓",
    x,
    y,
    w: 40,
    h: 36,
    fontSize: 26,
    bold: true,
    align: "center",
    color: COLORS.black,
    typeface: "Arial",
  });
}

function slide1(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addText(slide, {
    text: "Agent 时序类异常检测能力进展",
    x: 80,
    y: 180,
    w: 980,
    h: 76,
    fontSize: 38,
    bold: true,
    color: COLORS.black,
    typeface: "Arial",
  });
  ctx.addText(slide, {
    text: "面向无响应、黑白屏、加载失败与长时间加载场景的检测方案",
    x: 82,
    y: 270,
    w: 1000,
    h: 42,
    fontSize: 22,
    color: COLORS.black,
    typeface: "Arial",
  });
  ctx.addShape(slide, { x: 82, y: 340, w: 760, h: 2, fill: COLORS.black, line: ctx.line(COLORS.black, 0) });
  ctx.addText(slide, {
    text: "汇报时间：2026.06\n汇报主题：视频输入下单操作场景的 Agent 异常检测能力",
    x: 82,
    y: 382,
    w: 760,
    h: 76,
    fontSize: 18,
    color: COLORS.gray,
    typeface: "Arial",
  });
  addFooter(ctx, slide, 1);
  return slide;
}

function slide2(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "本阶段工作目标", "围绕视频输入下的单操作场景，设计并实现 Agent 时序类异常检测能力");
  addTable(
    ctx,
    slide,
    ["检测项", "检测目标"],
    [
      ["无响应判定", "控件或页面操作后是否发生无响应"],
      ["点击后出现黑白屏", "控件响应后是否进入黑屏或白屏状态"],
      ["页面加载失败", "控件响应后是否出现加载失败相关提示或弹窗"],
      ["页面长时间加载中", "控件响应后是否长期停留在加载状态"],
    ],
    120,
    180,
    [280, 760],
    68,
    20,
  );
  addFooter(ctx, slide, 2);
  return slide;
}

function slide3(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "输入与基本假设", "当前输入形式是一段操作视频，系统自主分析页面状态变化");
  addBullets(
    ctx,
    slide,
    [
      "一个视频主要对应一个关键操作场景",
      "视频中包含操作前、操作响应过程和最终状态",
      "系统不依赖人工标注具体点击发生在哪一帧",
      "通过抽帧序列自主分析页面状态变化",
    ],
    100,
    185,
    920,
    280,
    { fontSize: 23, lineH: 58 },
  );
  addBox(ctx, slide, {
    text: "后续如果输入协议明确提供操作时间点或关键帧，可以进一步提升检测精度。",
    x: 100,
    y: 510,
    w: 980,
    h: 76,
    fontSize: 20,
    fill: COLORS.light,
  });
  addFooter(ctx, slide, 3);
  return slide;
}

function slide4(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "整体技术路线", "CV 负责低成本证据提取，VLM 负责语义型失败提示识别");
  const x = 90;
  addBox(ctx, slide, { text: "输入视频", x: 500, y: 155, w: 220, h: 48, fontSize: 21, bold: true });
  addDownArrow(ctx, slide, 590, 205);
  addBox(ctx, slide, { text: "感知层\n视频抽帧 / 帧间变化 / 亮度统计 / 候选帧筛选", x: 260, y: 250, w: 700, h: 86, fontSize: 19 });
  addDownArrow(ctx, slide, 590, 338);
  addBox(ctx, slide, { text: "检测层", x: 500, y: 382, w: 220, h: 46, fontSize: 20, bold: true, fill: COLORS.light });
  addBox(ctx, slide, { text: "Loading 状态检测\n无响应 / 黑白屏 / 长时间加载", x, y: 455, w: 500, h: 96, fontSize: 19 });
  addBox(ctx, slide, { text: "失败提示检测\n加载失败 / 网络异常 / 请求失败 / 重试提示", x: 690, y: 455, w: 500, h: 96, fontSize: 19 });
  addDownArrow(ctx, slide, 590, 552);
  addBox(ctx, slide, { text: "决策层\n输出异常类型、判定依据和关键证据", x: 330, y: 600, w: 560, h: 64, fontSize: 19 });
  addFooter(ctx, slide, 4);
  return slide;
}

function slide5(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "检测项一：无响应判定", "判断控件或页面操作后是否没有产生有效反馈");
  addBullets(ctx, slide, [
    "对视频按固定间隔抽帧",
    "计算相邻帧之间的像素变化比例",
    "统计变化均值、最大值和标准差",
    "如果页面长时间几乎没有变化，则判定为疑似无响应",
  ], 90, 170, 550, 300, { fontSize: 21, lineH: 50 });
  addBox(ctx, slide, { text: "输出证据", x: 760, y: 170, w: 260, h: 48, fontSize: 20, bold: true, fill: COLORS.light });
  addBullets(ctx, slide, [
    "帧间变化比例序列",
    "平均变化率",
    "最大变化率",
    "静止判定阈值",
    "判定原因",
  ], 745, 250, 390, 300, { fontSize: 20, lineH: 45 });
  addFooter(ctx, slide, 5);
  return slide;
}

function slide6(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "无响应判定示意", "该检测项主要依赖 CV 信号，不需要默认调用 VLM");
  const xs = [120, 360, 600, 840];
  xs.forEach((xx, i) => addBox(ctx, slide, { text: i < 3 ? `frame_${i}` : "frame_n", x: xx, y: 200, w: 150, h: 70, fontSize: 20, bold: true }));
  addLabel(ctx, slide, "→", 296, 216, 38, 40, 26, true);
  addLabel(ctx, slide, "→", 536, 216, 38, 40, 26, true);
  addLabel(ctx, slide, "→  ...  →", 770, 216, 86, 40, 23, true);
  addBox(ctx, slide, { text: "diff_0", x: 240, y: 315, w: 100, h: 42, fontSize: 18, fill: COLORS.light });
  addBox(ctx, slide, { text: "diff_1", x: 480, y: 315, w: 100, h: 42, fontSize: 18, fill: COLORS.light });
  addBox(ctx, slide, { text: "diff_n", x: 760, y: 315, w: 100, h: 42, fontSize: 18, fill: COLORS.light });
  addBox(ctx, slide, { text: "如果 mean(diff) 很低，且 max(diff) 很低", x: 190, y: 430, w: 410, h: 74, fontSize: 21 });
  addLabel(ctx, slide, "→", 624, 446, 40, 40, 28, true);
  addBox(ctx, slide, { text: "操作后页面长期没有明显视觉反馈\n判定为疑似无响应", x: 690, y: 420, w: 420, h: 92, fontSize: 21, bold: true, fill: COLORS.light });
  addFooter(ctx, slide, 6);
  return slide;
}

function slide7(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "检测项二：点击后黑白屏", "判断控件响应后，页面是否进入异常黑屏或白屏");
  addBullets(ctx, slide, [
    "重点检查视频末尾窗口，而不是只看最后一帧",
    "对末尾多帧计算黑色像素比例、白色像素比例和亮度标准差",
    "要求连续多帧接近纯黑或纯白，才判定为黑屏或白屏",
    "避免将单帧转场、视频切换瞬间或截图时机问题误判为黑白屏",
  ], 92, 175, 1000, 330, { fontSize: 22, lineH: 55 });
  addFooter(ctx, slide, 7);
  return slide;
}

function slide8(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "黑白屏检测证据", "基于末尾窗口的亮度分布和连续命中情况进行判定");
  addTable(ctx, slide, ["指标", "含义"], [
    ["black_ratio", "近黑像素占比"],
    ["white_ratio", "近白像素占比"],
    ["std_luma", "亮度标准差"],
    ["terminal_run", "末尾连续命中帧数"],
    ["window_ratio", "末尾窗口内命中比例"],
  ], 90, 160, [330, 460], 55, 18);
  addBox(ctx, slide, {
    text: "判定逻辑\n\n末尾窗口中连续多帧满足：\nblack_ratio 高且 std_luma 低\n或 white_ratio 高且 std_luma 低\n\n→ 判定为黑屏 / 白屏异常",
    x: 860,
    y: 180,
    w: 300,
    h: 330,
    fontSize: 18,
    fill: COLORS.light,
  });
  addFooter(ctx, slide, 8);
  return slide;
}

function slide9(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "检测项三：页面加载失败", "判断控件响应后是否出现加载失败相关提示或弹窗信息");
  addBox(ctx, slide, {
    text: "这类问题本质上是语义检测，不适合只依赖像素变化。",
    x: 105,
    y: 165,
    w: 900,
    h: 58,
    fontSize: 22,
    fill: COLORS.light,
  });
  addBullets(ctx, slide, [
    "从抽帧序列中选择候选帧",
    "使用 CV 变化分数筛选更可能出现提示或弹窗的帧",
    "对候选帧及其前后帧调用 VLM",
    "判断是否存在加载失败、网络异常、请求失败、超时、重试等语义",
  ], 110, 280, 950, 300, { fontSize: 22, lineH: 55 });
  addFooter(ctx, slide, 9);
  return slide;
}

function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "页面加载失败检测流程", "通过候选帧预筛减少 VLM 调用次数，控制 token 成本");
  const boxes = [
    ["抽帧序列", "输入视频按固定间隔抽帧"],
    ["候选帧选择", "尾部窗口 / 首帧 / 中间帧 / 高变化帧"],
    ["CV 预筛", "前后帧变化 / 相对初始帧变化"],
    ["VLM 语义判断", "候选前帧 / 候选帧 / 候选后帧"],
    ["结构化输出", "是否失败 / 失败类型 / 证据文案 / 置信度"],
  ];
  boxes.forEach(([t, d], i) => {
    addBox(ctx, slide, { text: `${t}\n${d}`, x: 210, y: 150 + i * 96, w: 760, h: 66, fontSize: 18, bold: i === 0 });
    if (i < boxes.length - 1) addDownArrow(ctx, slide, 575, 214 + i * 96);
  });
  addFooter(ctx, slide, 10);
  return slide;
}

function slide11(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "检测项四：页面长时间加载中", "判断页面是否长期停留在 loading 状态");
  addBullets(ctx, slide, [
    "计算帧间变化比例",
    "判断变化是否长期保持小幅、稳定、规律",
    "结合视频持续时间，避免短视频中的正常 loading 被误报",
    "若页面长期只有加载指示器或骨架屏变化，而没有进入完成态，则判定为疑似长时间加载",
  ], 90, 165, 1020, 280, { fontSize: 21, lineH: 52 });
  addBox(ctx, slide, { text: "典型表现", x: 90, y: 500, w: 210, h: 48, fontSize: 20, bold: true, fill: COLORS.light });
  addLabel(ctx, slide, "转圈动画持续存在 / 进度反馈长期没有完成 / 页面内容长期不进入结果态 / 骨架屏持续闪烁", 330, 510, 820, 40, 19);
  addFooter(ctx, slide, 11);
  return slide;
}

function slide12(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "长时间加载检测证据", "结合变化稳定性和检测窗口时长进行判断");
  addTable(ctx, slide, ["指标", "含义"], [
    ["mean_change_ratio", "平均帧间变化率"],
    ["std_change_ratio", "变化率标准差"],
    ["max_change_ratio", "最大帧间变化率"],
    ["duration_sec", "视频检测窗口时长"],
    ["min_duration_sec", "判定长时间加载的最小时长"],
  ], 90, 160, [340, 500], 55, 18);
  addBox(ctx, slide, {
    text: "判定逻辑\n\n帧间变化长期较小\n变化波动稳定\n视频时长超过最小时长阈值\n\n→ 判定为疑似长时间加载",
    x: 890,
    y: 190,
    w: 280,
    h: 290,
    fontSize: 19,
    fill: COLORS.light,
  });
  addFooter(ctx, slide, 12);
  return slide;
}

function slide13(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "当前实现状态", "已形成 loading 类时序异常的基础检测链路");
  addBullets(ctx, slide, [
    "视频抽帧与帧间变化计算",
    "无响应检测逻辑",
    "末尾窗口黑白屏检测逻辑",
    "长时间加载检测逻辑",
    "页面加载失败独立检测器",
    "CV 候选帧预筛 + VLM 失败语义检测",
    "结构化输出异常类型、判定依据和关键证据",
    "基础单元测试验证",
  ], 90, 160, 610, 420, { fontSize: 19, lineH: 42 });
  addBox(ctx, slide, { text: "当前输出可区分", x: 790, y: 170, w: 290, h: 48, fontSize: 20, bold: true, fill: COLORS.light });
  addBullets(ctx, slide, ["no_response", "black_screen", "white_screen", "long_loading", "load_failed"], 790, 250, 300, 260, { fontSize: 21, lineH: 45 });
  addFooter(ctx, slide, 13);
  return slide;
}

function slide14(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "当前方案特点", "不是纯 VLM 端到端判断，而是结合 CV 和 VLM 的分层检测");
  addBullets(ctx, slide, [
    "视觉状态类问题通过 CV 判断，成本低、速度快",
    "语义提示类问题通过 VLM 判断，适合识别失败文案和弹窗",
    "每类检测项有独立证据，便于解释和调试",
    "可以输出结构化报告，支持后续统计准确率、召回率和 token 成本",
    "适配当前单视频单操作场景",
  ], 100, 170, 990, 360, { fontSize: 22, lineH: 57 });
  addFooter(ctx, slide, 14);
  return slide;
}

function slide15(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "与 Agent 框架的关系", "本阶段主要补齐 loading 类时序异常的检测能力");
  const lanes = [
    ["感知层", "视频抽帧\nROI / 关键区域提取\nCV 证据计算"],
    ["推理层", "VLM 状态抽象\n失败提示语义识别"],
    ["决策层", "规则判定\n异常类型融合\n测试报告输出"],
  ];
  lanes.forEach(([title, body], i) => {
    addBox(ctx, slide, { text: title, x: 90 + i * 390, y: 180, w: 300, h: 54, fontSize: 22, bold: true, fill: COLORS.light });
    addBox(ctx, slide, { text: body, x: 90 + i * 390, y: 255, w: 300, h: 170, fontSize: 20 });
    if (i < 2) addLabel(ctx, slide, "→", 418 + i * 390, 314, 46, 46, 30, true);
  });
  addBox(ctx, slide, { text: "Agent 框架中的时序异常检测模块", x: 340, y: 500, w: 560, h: 70, fontSize: 22, bold: true, fill: COLORS.light });
  addFooter(ctx, slide, 15);
  return slide;
}

function slide16(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "需要甲方确认的问题", "这些信息会直接影响检测阈值、候选帧数量和 VLM 调用策略");
  addBullets(ctx, slide, [
    "输入视频是否可以默认认为只包含一个关键操作场景？",
    "长时间加载的时间阈值应设置为多少秒？",
    "页面加载失败的业务文案范围是否有内部词表？",
    "是否可以提供真实正负样本视频？",
    "单次检测的耗时上限是多少？",
    "单次检测的 token 上限是多少？",
  ], 100, 165, 1010, 390, { fontSize: 22, lineH: 56 });
  addFooter(ctx, slide, 16);
  return slide;
}

function slide17(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "下一步计划与阶段性总结", "下一阶段重点进入真实样本评测和参数校准");
  addBullets(ctx, slide, [
    "构建 loading 类真实视频测试集，覆盖四类异常场景",
    "统计每类检测项的通过率、Precision、Recall",
    "记录单视频耗时、VLM 调用次数和 token 消耗",
    "优化候选帧筛选策略，并根据真实数据调整检测阈值",
    "对比纯 VLM 判断与 CV-first + VLM 判断的准确率和成本",
  ], 90, 155, 1030, 290, { fontSize: 21, lineH: 50 });
  addBox(ctx, slide, {
    text: "阶段性总结：当前方案将视觉状态类异常与语义提示类异常拆分检测，通过 CV 优先、VLM 补充的方式，在保证可解释性的同时控制推理成本。",
    x: 95,
    y: 510,
    w: 1030,
    h: 86,
    fontSize: 21,
    fill: COLORS.light,
  });
  addFooter(ctx, slide, 17);
  return slide;
}

const slides = [
  slide1,
  slide2,
  slide3,
  slide4,
  slide5,
  slide6,
  slide7,
  slide8,
  slide9,
  slide10,
  slide11,
  slide12,
  slide13,
  slide14,
  slide15,
  slide16,
  slide17,
];

await ensureArtifactToolWorkspace(workspace);
const artifact = await importArtifactTool(workspace);
const { Presentation, PresentationFile } = artifact;
const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const ctx = createSlideContext(artifact, {
  slideSize: { width: W, height: H },
  workspaceDir: workspace,
  outputDir: path.dirname(outputPptx),
  assetDir: path.join(workspace, "assets"),
  titleFont: "Arial",
  bodyFont: "Arial",
  monoFont: "Courier New",
});

for (const [index, fn] of slides.entries()) {
  const before = presentation.slides.count;
  const slide = await fn(presentation, ctx);
  if (presentation.slides.count !== before + 1) {
    throw new Error(`Slide ${index + 1} did not add exactly one slide.`);
  }
  if (!slide) {
    throw new Error(`Slide ${index + 1} did not return a slide.`);
  }
}

await fs.mkdir(path.dirname(outputPptx), { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);

await fs.mkdir(previewDir, { recursive: true });
const previewPaths = [];
for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  const blob = await presentation.export({ slide, format: "png", scale: 1 });
  const out = path.join(previewDir, `slide-${padSlideNumber(i + 1)}.png`);
  await saveBlobToFile(blob, out);
  previewPaths.push(out);
}

console.log(JSON.stringify({ outputPptx, slideCount: presentation.slides.count, previewDir, previewPaths }, null, 2));
