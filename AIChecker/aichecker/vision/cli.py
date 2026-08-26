"""vision_gui_agent MVP 入口（JSON 输入 + 统一评估链路）。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None  # type: ignore[assignment]

from aichecker.vision.checkers.count_change import ControlBounds, CountChangeDetector
from aichecker.vision.evaluator import VisionEvaluator, resolve_vlm_backend_config
from aichecker.vision.checkers.load_failure import LoadFailurePromptDetector
from aichecker.vision.checkers.list_refresh import ListRefreshDetector
from aichecker.vision.checkers.loading import LoadingDetector
from aichecker.vision.checkers.seek_playback import SeekPlaybackDetector
from aichecker.vision.checkers.video_play import VideoPlayDetector
from aichecker.vision.baseline_health import BASELINE_TASK_TYPES, VideoBaselineHealthOrchestrator
from aichecker.vision.pipeline import (
    CountChangeTaskDetector,
    FramePairTaskDetector,
    LoadFailurePromptTaskDetector,
    ListRefreshTaskDetector,
    LoadingTaskDetector,
    PipelineOrchestrator,
    SeekPlaybackTaskDetector,
    ToastTaskDetector,
    VideoBaselineHealthTaskDetector,
    VideoPlayTaskDetector,
)
from aichecker.vision.perception import ExtractedFrame, FrameExtractor
from aichecker.vision.preprocessor import GuiPreprocessor
from aichecker.vision.checkers.toast import ToastMessageDetector
from aichecker.vision.utils.logger import setup_logger


@dataclass
class VideoTaskInput:
    """单个任务输入。"""

    task_id: str
    mode: str = "full"
    task_type: str = "general"
    task_type_scope: list[str] | None = None
    video_file: str | None = None
    before_image: str | None = None
    after_image: str | None = None
    sample_interval_sec: float = 1.0
    start_sec: float = 0.0
    end_sec: float | None = None
    control_bounds: ControlBounds | None = None
    expected_count_change: str = "any_change"
    metric_hints: list[str] | None = None
    control_name_hint: str | None = None
    source_base_dir: Path | None = None
    expected_toast_keywords: list[str] | None = None
    expected_list_refresh: bool = True
    expected_result_text: str | None = None
    seek_timestamp_sec: float | None = None
    play_timestamp_sec: float | None = None


@dataclass
class RunOptions:
    """运行参数。"""

    output_root: Path | None
    input_json: str | None
    input_file: Path | None


def load_runtime_env(project_dir: Path) -> None:
    """通过 python-dotenv 加载项目根目录 .env。"""
    if load_dotenv is None:
        return
    env_path = project_dir / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def parse_args() -> RunOptions:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="vision_gui_agent MVP (JSON input)")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None, help="输出根目录")
    parser.add_argument(
        "--input-json",
        dest="input_json",
        type=str,
        default=None,
        help="JSON 字符串输入",
    )
    parser.add_argument("--input-file", dest="input_file", type=str, default=None, help="JSON 文件路径")
    args = parser.parse_args()
    return RunOptions(
        output_root=Path(args.output_dir).expanduser() if args.output_dir else None,
        input_json=args.input_json,
        input_file=Path(args.input_file).expanduser() if args.input_file else None,
    )


def _resolve_expected_change(payload: dict[str, Any]) -> str:
    """
    解析数量变化业务预期。
    expected_passed 是测试集标注字段，不参与线上任务语义解析。
    """
    if payload.get("expected_count_change") is not None:
        return str(payload["expected_count_change"])
    return "any_change"


def _resolve_expected_result_text(payload: dict[str, Any]) -> str | None:
    """读取自然语言预期结果。"""
    for key in ("expected_result", "expected_result_text", "expectation", "expected_behavior"):
        value = payload.get(key)
        if value not in ("", None):
            return str(value)
    return None


def _resolve_expected_list_refresh(payload: dict[str, Any]) -> bool:
    """
    解析列表刷新业务预期。
    expected_passed 是测试集标注字段，不参与线上任务语义解析。
    """
    if payload.get("expected_list_refresh") is not None:
        return bool(payload["expected_list_refresh"])
    expected_text = _resolve_expected_result_text(payload)
    if expected_text:
        lowered = expected_text.strip().lower()
        negative_markers = (
            "不刷新",
            "不要刷新",
            "无需刷新",
            "不应刷新",
            "不发生刷新",
            "保持不变",
            "no refresh",
            "not refresh",
            "should not refresh",
        )
        if any(marker in lowered for marker in negative_markers):
            return False
    return True


def _parse_bounds(payload: dict[str, Any]) -> ControlBounds | None:
    """
    兼容 bounds 输入：
    1) target_region_bounds: {x,y,width,height} 或 [x1,y1,x2,y2]
    2) control_bounds: {x,y,width,height}
    3) bounds: [x1,y1,x2,y2] 或 {x,y,width,height}
    """
    if payload.get("target_region_bounds") is not None:
        bounds_raw = payload["target_region_bounds"]
        if isinstance(bounds_raw, list):
            return ControlBounds.from_list(bounds_raw)
        if isinstance(bounds_raw, dict):
            return ControlBounds.from_payload(bounds_raw)
        raise ValueError(f"不支持的 target_region_bounds 类型: {type(bounds_raw)}")
    if payload.get("control_bounds") is not None:
        return ControlBounds.from_payload(payload["control_bounds"])
    if payload.get("bounds") is not None:
        bounds_raw = payload["bounds"]
        if isinstance(bounds_raw, list):
            return ControlBounds.from_list(bounds_raw)
        if isinstance(bounds_raw, dict):
            return ControlBounds.from_payload(bounds_raw)
        raise ValueError(f"不支持的 bounds 类型: {type(bounds_raw)}")
    return None


def parse_task_from_payload(payload: dict[str, Any], base_dir: Path | None = None) -> VideoTaskInput:
    """将 JSON payload 转为强类型任务对象。"""
    task_type = str(payload.get("task_type", payload.get("type", payload.get("prompt_type", "general")))).strip().lower()
    targeted_task_types = {
        "list_refresh",
        "seek_playback",
        "video_seek_playback",
        "video_seek",
        "video_play",
        "video_playback",
        "video_play_check",
    } | set(BASELINE_TASK_TYPES)
    default_mode = "targeted" if task_type in targeted_task_types else "full"
    mode = str(payload.get("mode", default_mode)).strip().lower()
    if mode not in {"full", "targeted"}:
        raise ValueError(f"mode 仅支持 full/targeted，当前为: {mode}")
    task_type_scope = payload.get("task_type_scope")
    normalized_scope: list[str] | None = None
    if task_type_scope is not None:
        if not isinstance(task_type_scope, list):
            raise ValueError("task_type_scope 必须是数组")
        normalized_scope = [str(x).strip().lower() for x in task_type_scope if str(x).strip()]
        if not normalized_scope:
            normalized_scope = None
    before_image = payload.get("before_image", payload.get("screenshot_a"))
    after_image = payload.get("after_image", payload.get("screenshot_b"))
    missing: list[str] = []
    if mode == "targeted":
        if task_type == "count_change":
            if before_image in ("", None):
                missing.append("before_image|screenshot_a")
            if after_image in ("", None):
                missing.append("after_image|screenshot_b")
            if payload.get("control_bounds") is None and payload.get("bounds") is None:
                missing.append("control_bounds|bounds")
        elif task_type == "list_refresh":
            if before_image in ("", None):
                missing.append("before_image|screenshot_a")
            if after_image in ("", None):
                missing.append("after_image|screenshot_b")
        else:
            if payload.get("video_file") in ("", None):
                missing.append("video_file")
    else:
        has_video = payload.get("video_file") not in ("", None)
        if task_type == "list_refresh":
            if before_image in ("", None):
                missing.append("before_image|screenshot_a")
            if after_image in ("", None):
                missing.append("after_image|screenshot_b")
        has_count_pair = (
            before_image not in ("", None)
            and after_image not in ("", None)
            and (
                task_type == "list_refresh"
                or payload.get("target_region_bounds") is not None
                or payload.get("control_bounds") is not None
                or payload.get("bounds") is not None
            )
        )
        if task_type != "list_refresh" and not has_video and not has_count_pair:
            missing.append("video_file 或 (before_image+after_image+bounds/control_bounds)")
    if missing:
        raise ValueError(f"mode={mode}, task_type={task_type} 时输入 JSON 缺少必填字段: {missing}")

    return VideoTaskInput(
        task_id=str(payload.get("task_id", "video_task")),
        mode=mode,
        task_type=task_type,
        task_type_scope=normalized_scope,
        video_file=str(payload["video_file"]) if payload.get("video_file") else None,
        before_image=str(before_image) if before_image else None,
        after_image=str(after_image) if after_image else None,
        sample_interval_sec=float(
            payload.get(
                "sample_interval_sec",
                0.5 if task_type in {"seek_playback", "video_seek_playback", "video_seek"} else 1.0,
            )
        ),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
        control_bounds=_parse_bounds(payload),
        expected_count_change=_resolve_expected_change(payload),
        metric_hints=[str(x) for x in payload.get("metric_hints", [])] if payload.get("metric_hints") else None,
        control_name_hint=(
            str(payload.get("control_name_hint"))
            if payload.get("control_name_hint")
            else (str(payload.get("label")) if payload.get("label") else None)
        ),
        source_base_dir=base_dir,
        expected_toast_keywords=(
            [str(x) for x in payload.get("expected_toast_keywords", [])]
            if payload.get("expected_toast_keywords")
            else ([str(x) for x in payload.get("toast_keywords", [])] if payload.get("toast_keywords") else None)
        ),
        expected_list_refresh=_resolve_expected_list_refresh(payload),
        expected_result_text=_resolve_expected_result_text(payload),
        seek_timestamp_sec=(
            float(payload["seek_timestamp_sec"])
            if payload.get("seek_timestamp_sec") is not None
            else (
                float(payload["seek_time_sec"])
                if payload.get("seek_time_sec") is not None
                else None
            )
        ),
        play_timestamp_sec=(
            float(payload["play_timestamp_sec"])
            if payload.get("play_timestamp_sec") is not None
            else (
                float(payload["play_time_sec"])
                if payload.get("play_time_sec") is not None
                else None
            )
        ),
    )


def load_task_input(options: RunOptions) -> VideoTaskInput:
    """读取 JSON 输入（优先级：CLI --input-json > CLI --input-file > ENV VGA_INPUT_JSON）。"""
    payload: dict[str, Any]
    if options.input_json:
        payload = json.loads(options.input_json)
        return parse_task_from_payload(payload, base_dir=None)
    if options.input_file:
        payload = json.loads(options.input_file.read_text(encoding="utf-8"))
        return parse_task_from_payload(payload, base_dir=options.input_file.parent)
    env_input = os.getenv("VGA_INPUT_JSON")
    if env_input:
        payload = json.loads(env_input)
        return parse_task_from_payload(payload, base_dir=None)
    raise ValueError("未提供 JSON 输入。请使用 --input-json、--input-file 或 VGA_INPUT_JSON。")


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    """保存单次执行报告为 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _build_frames_from_image_pair(
    before_image_path: Path,
    after_image_path: Path,
) -> list[ExtractedFrame]:
    """
    将 before/after 图片对归一为两帧序列，统一后续 detector 输入。
    """
    return [
        ExtractedFrame(
            image_path=before_image_path,
            timestamp_sec=0.0,
            fps=0.0,
            frame_count=2,
            duration_sec=0.0,
            target_frame_index=0,
            actual_frame_index=0,
            width=0,
            height=0,
        ),
        ExtractedFrame(
            image_path=after_image_path,
            timestamp_sec=1.0,
            fps=0.0,
            frame_count=2,
            duration_sec=0.0,
            target_frame_index=1,
            actual_frame_index=1,
            width=0,
            height=0,
        ),
    ]


def run() -> None:
    """执行完整工作流。"""
    run_start = time.perf_counter()
    options = parse_args()
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parents[1]
    load_runtime_env(project_dir)
    task = load_task_input(options)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_env = os.getenv("VGA_OUTPUT_DIR")
    output_root = options.output_root
    if output_root is None and output_root_env:
        output_root = Path(output_root_env).expanduser()
    if output_root is None:
        output_root = project_dir / "data" / "runs" / run_id
    if not output_root.is_absolute():
        output_root = (project_dir / output_root).resolve()

    frames_dir = output_root / "frames"
    reports_dir = output_root / "reports"
    debug_dir = output_root / "debug"
    logs_dir = output_root / "logs"
    prompt_log_path = debug_dir / f"vlm_prompts_{run_id}.jsonl"
    prompt_text_path = debug_dir / f"full_prompts_{run_id}.txt"

    logger = setup_logger(log_file_path=logs_dir / "vision_gui_agent.log")
    logger.info("vision_gui_agent 启动")
    logger.info("本次运行输出根目录: %s", output_root)

    backend_config = resolve_vlm_backend_config()
    if not backend_config.api_key:
        raise EnvironmentError("未读取到 VLM API key。请检查 .env 或当前终端环境变量。")
    logger.info(
        "VLM backend: backend=%s, model=%s, base_url=%s",
        backend_config.backend,
        backend_config.model,
        backend_config.base_url or "",
    )
    debug_enabled = os.getenv("VGA_DEBUG", "1").strip() in {"1", "true", "True", "YES", "yes"}
    log_full_data_url = os.getenv("VGA_LOG_FULL_DATA_URL", "0").strip() in {"1", "true", "True", "YES", "yes"}
    enable_preprocess = os.getenv("VGA_ENABLE_PREPROCESS", "1").strip() in {"1", "true", "True", "YES", "yes"}
    preprocess_max_images = int(os.getenv("VGA_PREPROCESS_MAX_EXTRA_IMAGES", "2"))
    toast_top_k_candidates = int(os.getenv("VGA_TOAST_TOP_K_CANDIDATES", "3"))
    toast_prompt_version = os.getenv("VGA_TOAST_PROMPT_VERSION", "current").strip() or "current"
    toast_min_contour_area = int(os.getenv("VGA_TOAST_MIN_CONTOUR_AREA", "1200"))
    toast_size_target_ratio = float(os.getenv("VGA_TOAST_SCORE_SIZE_TARGET_RATIO", "0.10"))
    toast_size_tolerance = float(os.getenv("VGA_TOAST_SCORE_SIZE_TOLERANCE", "0.10"))
    toast_position_center_ratio = float(os.getenv("VGA_TOAST_SCORE_POSITION_CENTER_RATIO", "0.50"))
    toast_position_tolerance = float(os.getenv("VGA_TOAST_SCORE_POSITION_TOLERANCE", "0.50"))
    toast_weight_size = float(os.getenv("VGA_TOAST_SCORE_WEIGHT_SIZE", "0.55"))
    toast_weight_position = float(os.getenv("VGA_TOAST_SCORE_WEIGHT_POSITION", "0.0"))
    toast_weight_motion = float(os.getenv("VGA_TOAST_SCORE_WEIGHT_MOTION", "0.45"))
    toast_motion_norm_ratio = float(os.getenv("VGA_TOAST_SCORE_MOTION_NORM_RATIO", "0.20"))
    toast_dynamic_penalty_threshold = float(os.getenv("VGA_TOAST_SCORE_DYNAMIC_PENALTY_THRESHOLD", "0.45"))
    toast_dynamic_penalty_scale = float(os.getenv("VGA_TOAST_SCORE_DYNAMIC_PENALTY_SCALE", "1.2"))
    toast_dynamic_penalty_max = float(os.getenv("VGA_TOAST_SCORE_DYNAMIC_PENALTY_MAX", "0.55"))
    toast_candidate_max_area_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_MAX_AREA_RATIO", "0.12"))
    toast_candidate_max_height_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_MAX_HEIGHT_RATIO", "0.30"))
    toast_candidate_min_aspect_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_MIN_ASPECT_RATIO", "1.60"))
    toast_candidate_expand_px = int(os.getenv("VGA_TOAST_CANDIDATE_EXPAND_PX", "10"))
    toast_candidate_full_width_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_FULL_WIDTH_RATIO", "0.96"))
    toast_candidate_edge_touch_px = int(os.getenv("VGA_TOAST_CANDIDATE_EDGE_TOUCH_PX", "3"))
    toast_high_dynamic_threshold = float(os.getenv("VGA_TOAST_HIGH_DYNAMIC_THRESHOLD", "0.20"))
    toast_band_search_ratio = float(os.getenv("VGA_TOAST_BAND_SEARCH_RATIO", "0.22"))
    toast_band_min_area_scale = float(os.getenv("VGA_TOAST_BAND_MIN_AREA_SCALE", "0.40"))
    toast_transition_penalty_threshold = float(os.getenv("VGA_TOAST_TRANSITION_PENALTY_THRESHOLD", "0.08"))
    toast_transition_penalty_scale = float(os.getenv("VGA_TOAST_TRANSITION_PENALTY_SCALE", "1.5"))
    toast_transition_penalty_max = float(os.getenv("VGA_TOAST_TRANSITION_PENALTY_MAX", "0.45"))
    loading_failure_probe_top_k = int(os.getenv("VGA_LOADING_FAILURE_PROBE_TOP_K", "5"))
    loading_failure_probe_min_cv_score = float(os.getenv("VGA_LOADING_FAILURE_PROBE_MIN_CV_SCORE", "0.02"))
    loading_failure_probe_force_keep = int(os.getenv("VGA_LOADING_FAILURE_PROBE_FORCE_KEEP", "1"))
    auto_crop_black_borders = os.getenv("VGA_AUTO_CROP_BLACK_BORDERS", "1").strip() in {
        "1",
        "true",
        "True",
        "YES",
        "yes",
    }
    crop_black_threshold = int(os.getenv("VGA_CROP_BLACK_THRESHOLD", "16"))
    crop_min_nonblack_ratio_per_line = float(os.getenv("VGA_CROP_MIN_NONBLACK_RATIO_PER_LINE", "0.01"))
    crop_min_area_ratio = float(os.getenv("VGA_CROP_MIN_AREA_RATIO", "0.10"))

    evaluator = VisionEvaluator(
        api_key=backend_config.api_key,
        model=backend_config.model,
        base_url=backend_config.base_url,
        backend=backend_config.backend,
        logger=logger,
        debug=debug_enabled,
        log_full_data_url=log_full_data_url,
        prompt_log_path=prompt_log_path,
        prompt_text_path=prompt_text_path,
    )
    preprocessor = GuiPreprocessor(
        artifact_dir=debug_dir / "preprocess",
        logger=logger,
        max_extra_images=preprocess_max_images,
        toast_min_contour_area=toast_min_contour_area,
        toast_size_target_ratio=toast_size_target_ratio,
        toast_size_tolerance=toast_size_tolerance,
        toast_position_center_ratio=toast_position_center_ratio,
        toast_position_tolerance=toast_position_tolerance,
        toast_weight_size=toast_weight_size,
        toast_weight_position=toast_weight_position,
        toast_weight_motion=toast_weight_motion,
        toast_motion_norm_ratio=toast_motion_norm_ratio,
        toast_dynamic_penalty_threshold=toast_dynamic_penalty_threshold,
        toast_dynamic_penalty_scale=toast_dynamic_penalty_scale,
        toast_dynamic_penalty_max=toast_dynamic_penalty_max,
        toast_candidate_max_area_ratio=toast_candidate_max_area_ratio,
        toast_candidate_max_height_ratio=toast_candidate_max_height_ratio,
        toast_candidate_min_aspect_ratio=toast_candidate_min_aspect_ratio,
        toast_candidate_expand_px=toast_candidate_expand_px,
        toast_candidate_full_width_ratio=toast_candidate_full_width_ratio,
        toast_candidate_edge_touch_px=toast_candidate_edge_touch_px,
        toast_high_dynamic_threshold=toast_high_dynamic_threshold,
        toast_band_search_ratio=toast_band_search_ratio,
        toast_band_min_area_scale=toast_band_min_area_scale,
        toast_transition_penalty_threshold=toast_transition_penalty_threshold,
        toast_transition_penalty_scale=toast_transition_penalty_scale,
        toast_transition_penalty_max=toast_transition_penalty_max,
    )
    count_change_detector = CountChangeDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
        crops_dir=debug_dir / "count_change_crops",
        preprocessor=preprocessor,
        enable_preprocess=enable_preprocess,
    )
    list_refresh_detector = ListRefreshDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
        crops_dir=debug_dir / "list_refresh_crops",
    )
    toast_detector = ToastMessageDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
        preprocessor=preprocessor,
        enable_preprocess=enable_preprocess,
        top_k_candidates=toast_top_k_candidates,
        prompt_version=toast_prompt_version,
    )
    loading_detector = LoadingDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
    )
    seek_playback_detector = SeekPlaybackDetector(
        logger=logger,
    )
    video_play_detector = VideoPlayDetector(
        logger=logger,
    )
    load_failure_detector = LoadFailurePromptDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
        probe_top_k=loading_failure_probe_top_k,
        probe_min_cv_score=loading_failure_probe_min_cv_score,
        probe_force_keep=loading_failure_probe_force_keep,
    )
    baseline_health_orchestrator = VideoBaselineHealthOrchestrator(
        loading_detector=loading_detector,
        load_failure_detector=load_failure_detector,
    )
    extractor = FrameExtractor(
        auto_crop_black_borders=auto_crop_black_borders,
        black_pixel_threshold=crop_black_threshold,
        min_nonblack_ratio_per_line=crop_min_nonblack_ratio_per_line,
        min_crop_area_ratio=crop_min_area_ratio,
    )

    logger.info(
        "任务输入: task_id=%s, mode=%s, task_type=%s, task_type_scope=%s",
        task.task_id,
        task.mode,
        task.task_type,
        task.task_type_scope,
    )
    logger.info(
        (
            "前处理开关: enable_preprocess=%s, max_extra_images=%s, "
            "toast_top_k_candidates=%s, toast_prompt_version=%s, "
            "loading_failure_probe_top_k=%s, loading_failure_probe_min_cv_score=%.4f, loading_failure_probe_force_keep=%s"
        ),
        enable_preprocess,
        preprocess_max_images,
        toast_top_k_candidates,
        toast_prompt_version,
        loading_failure_probe_top_k,
        loading_failure_probe_min_cv_score,
        loading_failure_probe_force_keep,
    )
    logger.info(
        (
            "toast打分配置: min_area=%s, size(target=%.3f,tol=%.3f,w=%.3f), "
            "position(center=%.3f,tol=%.3f,w=%.3f), motion(norm=%.3f,w=%.3f), "
            "penalty(threshold=%.3f,scale=%.3f,max=%.3f), "
            "candidate_filter(max_area=%.3f,max_height=%.3f,min_aspect=%.3f,expand_px=%s,full_width=%.3f,edge_px=%s), "
            "high_dynamic(threshold=%.3f,band_ratio=%.3f,min_area_scale=%.3f), "
            "transition_penalty(threshold=%.3f,scale=%.3f,max=%.3f)"
        ),
        toast_min_contour_area,
        toast_size_target_ratio,
        toast_size_tolerance,
        toast_weight_size,
        toast_position_center_ratio,
        toast_position_tolerance,
        toast_weight_position,
        toast_motion_norm_ratio,
        toast_weight_motion,
        toast_dynamic_penalty_threshold,
        toast_dynamic_penalty_scale,
        toast_dynamic_penalty_max,
        toast_candidate_max_area_ratio,
        toast_candidate_max_height_ratio,
        toast_candidate_min_aspect_ratio,
        toast_candidate_expand_px,
        toast_candidate_full_width_ratio,
        toast_candidate_edge_touch_px,
        toast_high_dynamic_threshold,
        toast_band_search_ratio,
        toast_band_min_area_scale,
        toast_transition_penalty_threshold,
        toast_transition_penalty_scale,
        toast_transition_penalty_max,
    )
    logger.info(
        "抽帧裁剪配置: auto_crop_black_borders=%s, black_threshold=%s, min_nonblack_ratio_per_line=%.4f, min_area_ratio=%.4f",
        auto_crop_black_borders,
        crop_black_threshold,
        crop_min_nonblack_ratio_per_line,
        crop_min_area_ratio,
    )
    logger.info("Prompt 构造方式: python_builder, requested_type=%s", task.task_type)

    sampled_frames: list[ExtractedFrame] = []
    frame_extract_elapsed_ms = 0.0
    if task.video_file:
        t_frame_start = time.perf_counter()
        video_path = Path(task.video_file)
        if not video_path.is_absolute():
            base = task.source_base_dir or project_dir
            video_path = (base / video_path).resolve()
        logger.info("视频输入: video_file=%s, interval=%.2fs", task.video_file, task.sample_interval_sec)
        sampled_frames = extractor.extract_frames_by_interval(
            video_path=video_path,
            output_dir=frames_dir,
            prefix=task.task_id,
            interval_sec=task.sample_interval_sec,
            start_sec=task.start_sec,
            end_sec=task.end_sec,
        )
        frame_extract_elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        logger.info("抽帧完成: 共 %s 帧", len(sampled_frames))
    elif task.before_image and task.after_image:
        before_image_path = Path(task.before_image or "")
        after_image_path = Path(task.after_image or "")
        if not before_image_path.is_absolute():
            base = task.source_base_dir or project_dir
            before_image_path = (base / before_image_path).resolve()
        if not after_image_path.is_absolute():
            base = task.source_base_dir or project_dir
            after_image_path = (base / after_image_path).resolve()
        if not before_image_path.exists() or not after_image_path.exists():
            raise FileNotFoundError(
                f"count_change 输入图片不存在: before={before_image_path}, after={after_image_path}"
            )
        sampled_frames = _build_frames_from_image_pair(
            before_image_path=before_image_path,
            after_image_path=after_image_path,
        )
        logger.info("输入图片对: before=%s, after=%s", before_image_path, after_image_path)

    orchestrator = PipelineOrchestrator(
        detectors=[
            CountChangeTaskDetector(detector=count_change_detector, logger=logger),
            ListRefreshTaskDetector(detector=list_refresh_detector, logger=logger),
            ToastTaskDetector(detector=toast_detector),
            SeekPlaybackTaskDetector(detector=seek_playback_detector),
            VideoPlayTaskDetector(detector=video_play_detector),
            VideoBaselineHealthTaskDetector(orchestrator=baseline_health_orchestrator),
            LoadingTaskDetector(detector=loading_detector),
            LoadFailurePromptTaskDetector(detector=load_failure_detector),
            FramePairTaskDetector(evaluator=evaluator, logger=logger),
        ]
    )
    t_pipeline_start = time.perf_counter()
    pipeline_result = orchestrator.run(
        task=task,
        sampled_frames=sampled_frames,
    )
    pipeline_elapsed_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
    segment_results: list[dict[str, Any]] = pipeline_result.segment_results
    video_bug_detected = pipeline_result.video_bug_detected
    resolved_task_intent = pipeline_result.resolved_task_intent
    token_usage_summary = evaluator.get_token_usage_summary()

    report = {
        "project": "vision_gui_agent",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "task_id": task.task_id,
            "mode": task.mode,
            "task_type": task.task_type,
            "task_type_scope": task.task_type_scope,
            "video_file": task.video_file,
            "before_image": task.before_image,
            "after_image": task.after_image,
            "resolved_task_intent": resolved_task_intent,
            "sample_interval_sec": task.sample_interval_sec,
            "start_sec": task.start_sec,
            "end_sec": task.end_sec,
            "control_bounds": (
                {
                    "x": task.control_bounds.x,
                    "y": task.control_bounds.y,
                    "width": task.control_bounds.width,
                    "height": task.control_bounds.height,
                }
                if task.control_bounds
                else None
            ),
            "target_region_bounds": (
                {
                    "x": task.control_bounds.x,
                    "y": task.control_bounds.y,
                    "width": task.control_bounds.width,
                    "height": task.control_bounds.height,
                }
                if task.task_type == "list_refresh" and task.control_bounds
                else None
            ),
            "expected_count_change": task.expected_count_change,
            "metric_hints": task.metric_hints,
            "control_name_hint": task.control_name_hint,
            "expected_toast_keywords": task.expected_toast_keywords,
            "expected_list_refresh": task.expected_list_refresh,
            "expected_result_text": task.expected_result_text,
            "seek_timestamp_sec": task.seek_timestamp_sec,
        },
        "video_level_result": {
            "bug_detected": video_bug_detected,
            "reason": "任一已执行 detector 判定为 bug，则视频级结果为存在问题。",
            "total_sampled_frames": len(sampled_frames),
            "total_segments": len(segment_results),
        },
        "token_usage_summary": token_usage_summary,
        "debug_artifacts": {
            "output_root": str(output_root),
            "frames_dir": str(frames_dir),
            "logs_dir": str(logs_dir),
            "vlm_prompt_log": str(prompt_log_path),
            "vlm_full_prompt_text": str(prompt_text_path),
            "prompt_builder": "aichecker.vision.prompt_builders.build_prompt_for_type",
            "selected_detector": pipeline_result.selected_detector,
            "detector_runs": pipeline_result.detector_runs,
            "timing": {
                "frame_extract_elapsed_ms": round(frame_extract_elapsed_ms, 2),
                "pipeline_elapsed_ms": round(pipeline_elapsed_ms, 2),
                "total_elapsed_ms": round((time.perf_counter() - run_start) * 1000.0, 2),
            },
        },
        "sampled_frames": [
            {
                "image_path": str(frame.image_path),
                "timestamp_sec": frame.timestamp_sec,
                "target_frame_index": frame.target_frame_index,
                "actual_frame_index": frame.actual_frame_index,
                "fps": frame.fps,
                "width": frame.width,
                "height": frame.height,
            }
            for frame in sampled_frames
        ],
        "segment_results": segment_results,
    }
    report_path = save_report(report=report, output_dir=reports_dir)
    total_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
    logger.info("视频级判定: bug_detected=%s", video_bug_detected)
    logger.info(
        (
            "Token统计: calls=%s, with_usage=%s, without_usage=%s, "
            "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s"
        ),
        token_usage_summary["prompt_call_count"],
        token_usage_summary["calls_with_usage"],
        token_usage_summary["calls_without_usage"],
        token_usage_summary["total_prompt_tokens"],
        token_usage_summary["total_completion_tokens"],
        token_usage_summary["total_tokens"],
    )
    logger.info(
        "运行耗时: total=%.2fms, frame_extract=%.2fms, pipeline=%.2fms",
        total_elapsed_ms,
        frame_extract_elapsed_ms,
        pipeline_elapsed_ms,
    )
    logger.info("测试报告已生成: %s", report_path)


if __name__ == "__main__":
    run()
