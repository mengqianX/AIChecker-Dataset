"""本仓库 testcase/ 回归测试的共享工具。"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import cv2
import pytest

from aichecker.vision.perception import ExtractedFrame, FrameExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROJECT_ROOT = REPO_ROOT / "AIChecker"
TESTAGENT_ROOT = Path(os.getenv("TESTAGENT_ROOT", str(REPO_ROOT))).resolve()


class _NoVlmEvaluator:
    """避免 loading 检测在 CV 不确定时触发真实 VLM 调用。"""

    def evaluate_json(self, **_kwargs: Any) -> Any:
        class _Result:
            parsed_json = {
                "bug_detected": False,
                "reason": "mock_no_vlm",
                "decision_basis": "mock_no_vlm",
                "anomaly_type": "none",
                "load_failed": False,
                "failure_type": "none",
                "evidence_text": "",
                "confidence": 0.0,
            }
            raw_response = "{}"

        return _Result()


def set_checker_report_meta(request: pytest.FixtureRequest, **kwargs: Any) -> None:
    current = getattr(request.node, "_checker_report_meta", {})
    current.update(kwargs)
    request.node._checker_report_meta = current


def record_evaluator_token_usage(request: pytest.FixtureRequest, evaluator: Any) -> None:
    """把 VisionEvaluator 累计 token 写入 pytest report meta。"""
    if evaluator is None or not hasattr(evaluator, "get_token_usage_summary"):
        return
    summary = evaluator.get_token_usage_summary() or {}
    set_checker_report_meta(
        request,
        prompt_call_count=int(summary.get("prompt_call_count", 0) or 0),
        calls_with_usage=int(summary.get("calls_with_usage", 0) or 0),
        prompt_tokens=int(summary.get("total_prompt_tokens", 0) or 0),
        completion_tokens=int(summary.get("total_completion_tokens", 0) or 0),
        total_tokens=int(summary.get("total_tokens", 0) or 0),
    )


def record_cli_report_metrics(request: pytest.FixtureRequest, report: dict[str, Any]) -> None:
    """从 vision.cli 报告中提取 token / 检测耗时到 pytest report meta。"""
    token = report.get("token_usage_summary") or {}
    timing = ((report.get("debug_artifacts") or {}).get("timing") or {})
    set_checker_report_meta(
        request,
        prompt_call_count=int(token.get("prompt_call_count", 0) or 0),
        calls_with_usage=int(token.get("calls_with_usage", 0) or 0),
        prompt_tokens=int(token.get("total_prompt_tokens", 0) or 0),
        completion_tokens=int(token.get("total_completion_tokens", 0) or 0),
        total_tokens=int(token.get("total_tokens", 0) or 0),
        detect_elapsed_ms=timing.get("total_elapsed_ms", ""),
    )


def testcase_json_dir(category: str) -> Path:
    return TESTAGENT_ROOT / "testcase" / category / "json"


def resolve_path(json_path: Path, rel_path: str) -> Path:
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()


def load_case(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("video_file"):
        data["video_file"] = str(resolve_path(json_path, str(data["video_file"])))
    data.setdefault("sample_interval_sec", 1.0)
    data.setdefault("start_sec", 0.0)
    return data


def collect_case_jsons(category: str) -> list[Path]:
    json_dir = testcase_json_dir(category)
    if not json_dir.exists():
        return []
    return sorted(json_dir.glob("*.json"), key=lambda p: p.name)


def extract_expected_passed(payload: dict[str, Any], json_path: Path) -> bool:
    if "expected_passed" in payload:
        return bool(payload["expected_passed"])
    raise AssertionError(
        f"No groundtruth found in {json_path}. "
        "Test case must include 'expected_passed' (boolean) field."
    )


def get_frame_extractor() -> FrameExtractor:
    auto_crop = os.getenv("VGA_AUTO_CROP_BLACK_BORDERS", "0").strip() in {
        "1",
        "true",
        "True",
        "YES",
        "yes",
    }
    return FrameExtractor(auto_crop_black_borders=auto_crop)


@contextmanager
def extract_video_frames(
    *,
    video_path: Path,
    prefix: str,
    sample_interval_sec: float,
    start_sec: float,
    end_sec: float | None = None,
) -> Iterator[list[ExtractedFrame]]:
    extractor = get_frame_extractor()
    with tempfile.TemporaryDirectory(prefix="testagent_frames_") as tmp:
        frames = extractor.extract_frames_by_interval(
            video_path=video_path,
            output_dir=Path(tmp),
            prefix=prefix,
            interval_sec=sample_interval_sec,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        yield frames


def require_testagent_root() -> None:
    if not TESTAGENT_ROOT.exists():
        pytest.skip(f"testcase root not found: {TESTAGENT_ROOT}")


def no_vlm_evaluator() -> _NoVlmEvaluator:
    return _NoVlmEvaluator()


def scale_bounds_to_frame(
    bounds: list[Any] | tuple[Any, ...],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """
    将 case.bounds [x1,y1,x2,y2] 映射到当前帧分辨率。

    TestAgent 标注常用设备坐标（如 1080x2340），而抽帧视频可能是 720x1584。
    """
    if len(bounds) != 4:
        raise ValueError(f"bounds 长度必须为 4，当前为 {len(bounds)}: {bounds}")
    x1, y1, x2, y2 = [float(v) for v in bounds]
    if x2 <= frame_width and y2 <= frame_height:
        sx = sy = 1.0
    else:
        candidates: list[tuple[float, float]] = []
        for ref_w, ref_h in (
            (1080.0, 2340.0),
            (1080.0, 2400.0),
            (1170.0, 2532.0),
            (1080.0, 1080.0 * frame_height / max(1, frame_width)),
        ):
            candidates.append((frame_width / ref_w, frame_height / ref_h))
        # 最后回退：按越界轴独立缩放
        candidates.append(
            (
                frame_width / max(x2, float(frame_width)),
                frame_height / max(y2, float(frame_height)),
            )
        )
        sx = sy = 1.0
        for cand_sx, cand_sy in candidates:
            if x2 * cand_sx <= frame_width + 1 and y2 * cand_sy <= frame_height + 1:
                sx, sy = cand_sx, cand_sy
                break

    left = max(0, int(round(x1 * sx)))
    top = max(0, int(round(y1 * sy)))
    right = min(frame_width, int(round(x2 * sx)))
    bottom = min(frame_height, int(round(y2 * sy)))
    if right - left < 2 or bottom - top < 2:
        raise ValueError(
            f"bounds 映射后区域过小: raw={bounds}, frame={frame_width}x{frame_height}, "
            f"mapped=({left},{top},{right},{bottom})"
        )
    return left, top, right, bottom


def crop_frames_to_bounds(
    frames: list[ExtractedFrame],
    bounds: list[Any] | tuple[Any, ...],
    output_dir: Path,
    *,
    min_side_px: int = 48,
) -> list[ExtractedFrame]:
    """
    按 bounds 裁剪抽帧结果；过小 ROI 会向四周扩边，减轻压缩噪声误触发。
    """
    if not frames:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames[0].image_path))
    if first is None:
        raise RuntimeError(f"无法读取图片: {frames[0].image_path}")
    frame_h, frame_w = first.shape[:2]
    left, top, right, bottom = scale_bounds_to_frame(bounds, frame_w, frame_h)

    width = right - left
    height = bottom - top
    if width < min_side_px or height < min_side_px:
        pad_x = max(0, (min_side_px - width + 1) // 2)
        pad_y = max(0, (min_side_px - height + 1) // 2)
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(frame_w, right + pad_x)
        bottom = min(frame_h, bottom + pad_y)

    cropped: list[ExtractedFrame] = []
    for idx, frame in enumerate(frames):
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {frame.image_path}")
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError(f"空裁剪区域: bounds mapped=({left},{top},{right},{bottom})")
        out_path = output_dir / f"{frame.image_path.stem}_roi_{idx:03d}.png"
        if not cv2.imwrite(str(out_path), crop):
            raise RuntimeError(f"写入裁剪帧失败: {out_path}")
        cropped.append(
            replace(
                frame,
                image_path=out_path,
                width=int(crop.shape[1]),
                height=int(crop.shape[0]),
            )
        )
    return cropped
