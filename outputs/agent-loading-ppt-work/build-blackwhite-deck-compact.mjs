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
const previewDir = path.join(workspace, "preview-compact");
const outputPptx = path.join(root, "outputs/agent-loading-detection-report-compact.pptx");
const W = 1280;
const H = 720;
const C = { black: "#000000", white: "#FFFFFF", gray: "#666666", light: "#F4F4F4" };

function addTitle(ctx, slide, title, subtitle = "") {
  ctx.addText(slide, { text: title, x: 64, y: 42, w: 980, h: 48, fontSize: 31, bold: true, color: C.black, typeface: "Arial" });
  if (subtitle) {
    ctx.addText(slide, { text: subtitle, x: 66, y: 91, w: 1030, h: 30, fontSize: 15, color: C.gray, typeface: "Arial" });
  }
  ctx.addShape(slide, { x: 64, y: 124, w: 1152, h: 1.4, fill: C.black, line: ctx.line(C.black, 0) });
}

function addFooter(ctx, slide, n) {
  ctx.addText(slide, { text: String(n).padStart(2, "0"), x: 1168, y: 668, w: 48, h: 22, fontSize: 13, color: C.gray, align: "right", typeface: "Arial" });
}

function addBullets(ctx, slide, bullets, x, y, w, opts = {}) {
  const fontSize = opts.fontSize ?? 23;
  const lineH = opts.lineH ?? 55;
  bullets.forEach((text, i) => {
    const yy = y + i * lineH;
    ctx.addText(slide, { text: "•", x, y: yy + 1, w: 24, h: 28, fontSize, bold: true, color: C.black, typeface: "Arial" });
    ctx.addText(slide, {
      text,
      x: x + 34,
      y: yy,
      w,
      h: lineH - 4,
      fontSize,
      color: C.black,
      typeface: "Arial",
      insets: { left: 0, right: 8, top: 0, bottom: 0 },
    });
  });
}

function addBox(ctx, slide, { text, x, y, w, h, fontSize = 19, bold = false, fill = C.white, align = "center" }) {
  const box = ctx.addShape(slide, { x, y, w, h, fill, line: ctx.line(C.black, 1.2) });
  box.text = text;
  box.text.fontSize = fontSize;
  box.text.color = C.black;
  box.text.bold = bold;
  box.text.typeface = "Arial";
  box.text.alignment = align;
  box.text.verticalAlignment = "middle";
  box.text.insets = { left: 14, right: 14, top: 8, bottom: 8 };
  return box;
}

function addText(ctx, slide, text, x, y, w, h, fontSize = 20, bold = false) {
  return ctx.addText(slide, { text, x, y, w, h, fontSize, bold, color: C.black, typeface: "Arial" });
}

function addArrow(ctx, slide, x, y) {
  ctx.addText(slide, { text: "↓", x, y, w: 42, h: 34, fontSize: 26, bold: true, align: "center", color: C.black, typeface: "Arial" });
}

function addTable(ctx, slide, headers, rows, x, y, widths, rowH = 58) {
  const drawCell = (text, col, rowY, fill, bold = false) => {
    const xx = x + widths.slice(0, col).reduce((a, b) => a + b, 0);
    const cell = ctx.addShape(slide, { x: xx, y: rowY, w: widths[col], h: rowH, fill, line: ctx.line(C.black, 1) });
    cell.text = text;
    cell.text.fontSize = 18;
    cell.text.bold = bold;
    cell.text.typeface = "Arial";
    cell.text.alignment = "left";
    cell.text.verticalAlignment = "middle";
    cell.text.insets = { left: 12, right: 10, top: 6, bottom: 6 };
  };
  headers.forEach((h, i) => drawCell(h, i, y, C.light, true));
  rows.forEach((row, r) => row.forEach((value, i) => drawCell(value, i, y + rowH * (r + 1), C.white)));
}

function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addText(slide, { text: "Agent 时序类异常检测能力进展", x: 80, y: 190, w: 980, h: 76, fontSize: 38, bold: true, color: C.black, typeface: "Arial" });
  ctx.addText(slide, { text: "无响应、黑白屏、页面加载失败与长时间加载检测方案", x: 82, y: 280, w: 980, h: 42, fontSize: 23, color: C.black, typeface: "Arial" });
  ctx.addShape(slide, { x: 82, y: 350, w: 760, h: 2, fill: C.black, line: ctx.line(C.black, 0) });
  ctx.addText(slide, { text: "汇报时间：2026.06\n输入场景：视频输入下的单操作 GUI 异常检测", x: 82, y: 392, w: 760, h: 76, fontSize: 18, color: C.gray, typeface: "Arial" });
  addFooter(ctx, slide, 1);
  return slide;
}

function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "本阶段目标", "围绕单操作视频，补齐 Agent 对 loading 类时序异常的检测能力");
  addTable(ctx, slide, ["检测项", "检测目标"], [
    ["无响应判定", "操作后页面或控件没有有效反馈"],
    ["点击后黑白屏", "响应后进入持续黑屏或白屏状态"],
    ["页面加载失败", "出现失败提示、网络异常、请求失败或重试弹窗"],
    ["页面长时间加载中", "长期停留在 loading / spinner / 骨架屏状态"],
  ], 120, 180, [300, 720], 70);
  addFooter(ctx, slide, 2);
  return slide;
}

function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "整体方案", "视觉状态类问题优先用 CV，语义提示类问题交给 VLM");
  addBox(ctx, slide, { text: "输入视频", x: 500, y: 155, w: 220, h: 50, fontSize: 22, bold: true });
  addArrow(ctx, slide, 590, 208);
  addBox(ctx, slide, { text: "感知层\n抽帧、帧间变化、亮度统计、候选帧筛选", x: 285, y: 255, w: 650, h: 82, fontSize: 20 });
  addArrow(ctx, slide, 590, 340);
  addBox(ctx, slide, { text: "Loading 状态检测\n无响应 / 黑白屏 / 长时间加载", x: 120, y: 405, w: 470, h: 98, fontSize: 20 });
  addBox(ctx, slide, { text: "失败提示检测\n加载失败 / 网络异常 / 请求失败 / 重试提示", x: 690, y: 405, w: 470, h: 98, fontSize: 20 });
  addArrow(ctx, slide, 590, 510);
  addBox(ctx, slide, { text: "结构化输出：异常类型、判定依据、关键证据", x: 330, y: 580, w: 560, h: 58, fontSize: 20, bold: true, fill: C.light });
  addFooter(ctx, slide, 3);
  return slide;
}

function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "无响应与长时间加载：基于帧间变化判断", "这两类问题主要体现为页面变化不足或长期只有小幅规律变化");
  addBox(ctx, slide, { text: "无响应判定", x: 110, y: 170, w: 300, h: 58, fontSize: 23, bold: true, fill: C.light });
  addBullets(ctx, slide, ["计算相邻帧变化比例", "若整体变化长期很低，则认为操作后缺少有效反馈", "输出变化均值、最大值和判定依据"], 110, 260, 470, { fontSize: 21, lineH: 56 });
  addBox(ctx, slide, { text: "长时间加载", x: 710, y: 170, w: 300, h: 58, fontSize: 23, bold: true, fill: C.light });
  addBullets(ctx, slide, ["识别长期小幅、稳定、规律的变化", "结合视频持续时间，避免短时正常 loading 误报", "适用于 spinner、骨架屏、进度动画等场景"], 710, 260, 440, { fontSize: 21, lineH: 56 });
  addFooter(ctx, slide, 4);
  return slide;
}

function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "点击后黑白屏：基于末尾窗口判断", "避免把单帧转场、切换瞬间或截图时机问题误判为异常");
  addBullets(ctx, slide, [
    "重点检查视频末尾窗口，而不是只看最后一帧",
    "统计末尾多帧的黑色/白色像素比例和亮度稳定性",
    "只有连续多帧接近纯黑或纯白，才判定为黑屏/白屏",
  ], 100, 180, 980, { fontSize: 24, lineH: 68 });
  addBox(ctx, slide, { text: "判定核心：最终状态持续异常，而不是某一帧偶然异常。", x: 190, y: 500, w: 840, h: 70, fontSize: 24, bold: true, fill: C.light });
  addFooter(ctx, slide, 5);
  return slide;
}

function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "页面加载失败：独立语义检测", "失败提示和弹窗属于语义问题，单纯依赖像素变化不稳定");
  const steps = [
    ["1", "候选帧选择", "从首帧、中间帧、尾部窗口和高变化帧中选候选"],
    ["2", "CV 预筛", "优先保留更可能出现提示或弹窗的帧"],
    ["3", "VLM 判断", "结合候选帧及前后帧识别失败语义"],
    ["4", "结构化输出", "输出失败类型、证据文案和置信度"],
  ];
  steps.forEach(([n, t, d], i) => {
    addBox(ctx, slide, { text: n, x: 90, y: 165 + i * 105, w: 52, h: 52, fontSize: 24, bold: true, fill: C.light });
    addText(ctx, slide, t, 170, 162 + i * 105, 260, 32, 23, true);
    addText(ctx, slide, d, 170, 198 + i * 105, 880, 38, 20);
  });
  addFooter(ctx, slide, 6);
  return slide;
}

function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "当前实现状态", "已形成 loading 类时序异常的基础检测链路");
  addBullets(ctx, slide, [
    "已实现视频抽帧、帧间变化计算和阶段性证据输出",
    "已实现无响应、黑白屏、长时间加载检测",
    "已实现页面加载失败独立检测器",
    "已支持结构化输出异常类型、判定依据和关键证据",
    "已完成基础单元测试验证",
  ], 95, 170, 980, { fontSize: 23, lineH: 60 });
  addFooter(ctx, slide, 7);
  return slide;
}

function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "方案特点", "将可解释性、准确性和推理成本放在同一套检测框架中考虑");
  addTable(ctx, slide, ["特点", "说明"], [
    ["CV 优先", "视觉状态类异常不默认调用 VLM，降低耗时和 token 消耗"],
    ["VLM 补充", "失败提示、弹窗等语义型问题由 VLM 判断"],
    ["职责拆分", "Loading 状态检测和失败提示检测分开实现"],
    ["结构化报告", "输出异常类型、证据指标、候选帧和模型响应"],
  ], 115, 180, [300, 720], 76);
  addFooter(ctx, slide, 8);
  return slide;
}

function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "需要甲方确认的问题", "这些信息会影响阈值设置、候选帧数量和 VLM 调用策略");
  addBullets(ctx, slide, [
    "输入视频是否默认只包含一个关键操作场景？",
    "长时间加载的时间阈值应设置为多少秒？",
    "页面加载失败是否有内部业务文案词表？",
    "是否可以提供真实正负样本视频？",
    "单次检测的耗时上限和 token 上限是多少？",
  ], 100, 170, 980, { fontSize: 24, lineH: 65 });
  addFooter(ctx, slide, 9);
  return slide;
}

function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  addTitle(ctx, slide, "下一步计划", "进入真实样本评测和参数校准阶段");
  addBullets(ctx, slide, [
    "构建 loading 类真实视频测试集，覆盖四类异常场景",
    "统计每类检测项的通过率、Precision、Recall",
    "记录单视频耗时、VLM 调用次数和 token 消耗",
    "优化候选帧筛选策略，并根据真实数据调整检测阈值",
  ], 100, 170, 980, { fontSize: 23, lineH: 62 });
  addBox(ctx, slide, {
    text: "阶段性总结：当前方案通过 CV 优先、VLM 补充的方式，在保证可解释性的同时控制推理成本。",
    x: 120,
    y: 520,
    w: 980,
    h: 74,
    fontSize: 23,
    bold: true,
    fill: C.light,
  });
  addFooter(ctx, slide, 10);
  return slide;
}

const fns = [slide01, slide02, slide03, slide04, slide05, slide06, slide07, slide08, slide09, slide10];

await ensureArtifactToolWorkspace(workspace);
const artifact = await importArtifactTool(workspace);
const { Presentation, PresentationFile } = artifact;
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ctx = createSlideContext(artifact, {
  slideSize: { width: 1280, height: 720 },
  workspaceDir: workspace,
  outputDir: path.dirname(outputPptx),
  assetDir: path.join(workspace, "assets"),
  titleFont: "Arial",
  bodyFont: "Arial",
  monoFont: "Courier New",
});

for (const fn of fns) await fn(presentation, ctx);
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
console.log(JSON.stringify({ outputPptx, slideCount: presentation.slides.count, previewDir }, null, 2));
