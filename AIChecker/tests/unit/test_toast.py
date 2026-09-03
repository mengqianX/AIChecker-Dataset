from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aichecker.vision.checkers.toast import ToastMessageDetector
from aichecker.vision.evaluator import JsonEvaluationResult
from aichecker.vision.perception import ExtractedFrame
from aichecker.vision.preprocessor import GuiPreprocessor
from aichecker.vision.prompt_builders import build_prompt_for_type, render_toast_user_prompt


def _blank(path: Path, color: tuple[int, int, int] = (40, 40, 40)) -> None:
    cv2.imwrite(str(path), np.full((640, 360, 3), color, dtype=np.uint8))


def _with_bar(
    path: Path,
    *,
    color: tuple[int, int, int] = (40, 40, 40),
        bar_color: tuple[int, int, int] = (0, 0, 0),
    y0: int = 500,
    y1: int = 560,
    x0: int = 30,
    x1: int = 330,
) -> None:
    image = np.full((640, 360, 3), color, dtype=np.uint8)
    image[y0:y1, x0:x1] = bar_color
    cv2.imwrite(str(path), image)


def test_toast_prompt_has_no_roi_text() -> None:
    prompt = render_toast_user_prompt(
        prompt_pack=build_prompt_for_type("toast", {}),
        context={
            "task_intent": "检测 toast",
            "candidate_timestamp_sec": 1.5,
            "keywords_text": "无",
        },
    )
    assert "roi" not in prompt.lower()
    assert "前处理" not in prompt
    pack = build_prompt_for_type("toast", {})
    assert "先只根据图1的图标/按钮文案" in pack.system_prompt
    assert "inferred_expected_toast_text" in pack.system_prompt
    assert "再看图2有没有 toast" in pack.system_prompt
    assert "已回收" in pack.system_prompt
    assert "地图导航指引" in pack.system_prompt
    assert "禁止 unknown" in pack.system_prompt
    assert "图标/按钮文案" in pack.system_prompt
    assert "另一种操作" in pack.system_prompt
    assert "图1控件本身" in prompt
    assert "选择标签" in prompt
    assert "引导" in prompt
    assert "两张完整帧" in prompt
    assert "图3" not in pack.system_prompt
    assert "图3" not in prompt
    assert ToastMessageDetector._build_toast_image_role_labels() == [
        "图1:动作前完整帧（必须用于推断操作语义）",
        "图2:候选完整帧（toast出现时）",
    ]


def test_same_page_rectangular_overlay_ranks_toast_frame(tmp_path: Path) -> None:
    paths = []
    for idx in range(6):
        path = tmp_path / f"f{idx}.png"
        if idx == 3:
            _with_bar(path)
        else:
            _blank(path)
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    best = max(scores, key=lambda item: float(item["score"]))
    assert best["index"] in {2, 3, 4}
    assert best["score"] > 0.2
    assert best["source"] in {"pair_before_center", "pair_center_after"}


def test_navigation_then_toast_uses_new_page_side(tmp_path: Path) -> None:
    paths = []
    for idx in range(6):
        path = tmp_path / f"f{idx}.png"
        if idx <= 1:
            _blank(path, color=(40, 40, 40))
        elif idx == 3:
            _with_bar(path, color=(90, 90, 90))
        else:
            _blank(path, color=(90, 90, 90))
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    toast_score = float(scores[3]["score"])
    nav_score = float(scores[1]["score"])
    assert toast_score > nav_score
    assert toast_score > 0.2
    assert scores[3]["source"] in {"pair_before_center", "pair_center_after"}


def test_chronic_ad_slot_does_not_outrank_one_shot_toast(tmp_path: Path) -> None:
    paths = []
    for idx in range(8):
        path = tmp_path / f"f{idx}.png"
        ad_color = (10, 10, 200) if idx % 2 == 0 else (10, 200, 10)
        if idx == 6:
            _with_bar(path, bar_color=(0, 0, 0))
            image = cv2.imread(str(path))
            image[20:80, 20:340] = ad_color
            cv2.imwrite(str(path), image)
        else:
            image = np.full((640, 360, 3), 40, dtype=np.uint8)
            image[20:80, 20:340] = ad_color
            cv2.imwrite(str(path), image)
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    hot = scores[0]["hot_mask"]
    assert hot["enabled"] is True
    assert hot["hot_cell_count"] >= 1
    toast_best = max(float(scores[5]["score"]), float(scores[6]["score"]))
    ad_best = max(float(item["score"]) for item in scores[:5])
    assert toast_best >= ad_best


def test_select_score_peaks_keeps_separated_high_scores() -> None:
    scores = [
        {"index": 1, "score": 0.9},
        {"index": 2, "score": 0.88},
        {"index": 10, "score": 0.7},
        {"index": 11, "score": 0.2},
    ]
    selected = ToastMessageDetector._select_score_peaks(scores, max_peaks=2, min_separation=3)
    assert selected == [1, 10]


def test_select_score_peaks_skips_near_zero_and_does_not_pad() -> None:
    scores = [
        {"index": 0, "score": 0.0},
        {"index": 1, "score": 0.01},
        {"index": 2, "score": 0.0},
        {"index": 8, "score": 0.81},
    ]
    selected = ToastMessageDetector._select_score_peaks(scores, max_peaks=2, min_separation=3, min_score=0.2)
    assert selected == [8]


def test_select_score_peaks_skips_vlm_when_all_scores_are_zero() -> None:
    scores = [
        {"index": 0, "score": 0.0},
        {"index": 1, "score": 0.0},
        {"index": 2, "score": 0.0},
    ]
    selected = ToastMessageDetector._select_score_peaks(scores, max_peaks=2, min_separation=3, min_score=0.2)
    assert selected == []


def _with_center_pill(
    path: Path,
    *,
    color: tuple[int, int, int] = (220, 220, 220),
    pill_color: tuple[int, int, int] = (40, 40, 40),
    y0: int = 270,
    y1: int = 360,
    x0: int = 80,
    x1: int = 280,
) -> None:
    image = np.full((640, 360, 3), color, dtype=np.uint8)
    image[y0:y1, x0:x1] = pill_color
    cv2.imwrite(str(path), image)


def test_center_floating_toast_is_selected(tmp_path: Path) -> None:
    paths = []
    for idx in range(8):
        path = tmp_path / f"f{idx}.png"
        if 3 <= idx <= 5:
            _with_center_pill(path)
        else:
            cv2.imwrite(str(path), np.full((640, 360, 3), 220, dtype=np.uint8))
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    peaks = ToastMessageDetector._select_score_peaks(scores, max_peaks=2, min_separation=3, min_score=0.2)
    assert peaks, "居中 toast 必须能进峰"
    assert any(idx in {3, 4, 5} for idx in peaks)
    toast_best = max(float(scores[idx]["score"]) for idx in (3, 4, 5))
    assert toast_best > 0.2


def test_later_toast_outranks_earlier_persistent_dialog_bar(tmp_path: Path) -> None:
    paths = []
    for idx in range(12):
        path = tmp_path / f"f{idx}.png"
        image = np.full((640, 360, 3), 40, dtype=np.uint8)
        if 1 <= idx <= 6:
            image[180:260, 40:320] = (230, 230, 230)
        if idx == 9:
            image[500:560, 30:330] = (0, 0, 0)
        cv2.imwrite(str(path), image)
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    peaks = ToastMessageDetector._select_score_peaks(scores, max_peaks=2, min_separation=3, min_score=0.2)
    assert any(idx in {8, 9, 10} for idx in peaks)
    toast_best = max(float(scores[idx]["score"]) for idx in (8, 9, 10))
    assert toast_best > 0.2
    dialog_idxs = set(range(1, 7))
    assert not (len(peaks) >= 2 and all(idx in dialog_idxs for idx in peaks))


def test_center_toast_after_page_change_is_selected(tmp_path: Path) -> None:
    paths = []
    for idx in range(8):
        path = tmp_path / f"f{idx}.png"
        if idx <= 2:
            _blank(path, color=(40, 40, 40))
        elif 4 <= idx <= 5:
            _with_center_pill(path, color=(210, 210, 210))
        else:
            cv2.imwrite(str(path), np.full((640, 360, 3), 210, dtype=np.uint8))
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    peaks = ToastMessageDetector._select_score_peaks(scores, max_peaks=2, min_separation=3, min_score=0.2)
    assert any(idx in {4, 5, 6} for idx in peaks)
    toast_best = max(float(scores[idx]["score"]) for idx in (4, 5))
    assert toast_best > 0.2


def test_disappearing_overlay_outranks_persistent_bubble(tmp_path: Path) -> None:
    paths = []
    for idx in range(10):
        path = tmp_path / f"f{idx}.png"
        image = np.full((640, 360, 3), 40, dtype=np.uint8)
        if idx == 2:
            image[500:560, 30:330] = (0, 0, 0)
        if idx >= 5:
            image[80:160, 40:320] = (20, 90, 220)
        cv2.imwrite(str(path), image)
        paths.append(path)

    scores = GuiPreprocessor().score_toast_sequence(paths)
    toast_best = max(float(scores[idx]["score"]) for idx in (1, 2, 3))
    bubble_best = max(float(item["score"]) for item in scores[5:])
    assert toast_best > 0.2
    assert toast_best > bubble_best
    bubble_tags = {item.get("transience") for item in scores[5:] if item.get("transience")}
    assert "still_present" in bubble_tags or "still_present_short" in bubble_tags


def test_apply_tag_recycle_conflict_marks_bug() -> None:
    candidates = [
        {
            "toast_visible": True,
            "toast_text": "1 已回收",
            "action_semantic": "unknown",
            "inferred_expected_toast_text": "",
            "expectation_met": False,
            "is_uncertain_action": True,
            "reason": "动作不可观测",
        },
        {
            "toast_visible": False,
            "toast_text": "",
            "action_semantic": "选择标签",
            "inferred_expected_toast_text": "",
            "expectation_met": False,
            "is_uncertain_action": False,
            "reason": "对话框确认标签",
        },
    ]
    ToastMessageDetector._apply_cross_candidate_conflicts(candidates)
    recycle = candidates[0]
    assert recycle["toast_visible"] is True
    assert recycle["expectation_met"] is False
    assert recycle["is_uncertain_action"] is False
    assert "标签" in recycle["inferred_expected_toast_text"] or recycle["inferred_expected_toast_text"] == "标签已添加"


def test_write_vlm_preview_downscales_to_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "full.png"
    cv2.imwrite(str(source), np.full((1920, 1080, 3), 40, dtype=np.uint8))
    dest = tmp_path / "preview.jpg"
    preview = GuiPreprocessor().write_vlm_preview(source, dest)
    image = cv2.imread(str(preview))
    assert image is not None
    assert max(image.shape[0], image.shape[1]) <= 512
    assert preview.suffix.lower() == ".jpg"


class _StubToastEvaluator:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def evaluate_json(self, **kwargs):  # noqa: ANN003
        self.last_kwargs = kwargs
        return JsonEvaluationResult(
            parsed_json={
                "toast_visible": True,
                "toast_text": "已保存",
                "action_semantic": "保存笔记",
                "inferred_expected_toast_text": "已保存",
                "expectation_met": True,
                "reverse_inference_risk": "low",
                "action_evidence_from_frame12": "点击保存",
                "toast_evidence_from_frame2": "底部提示条",
                "reason": "ok",
                "confidence": 0.9,
            },
            raw_response="{}",
        )


def _frame(path: Path, idx: int) -> ExtractedFrame:
    return ExtractedFrame(
        image_path=path,
        timestamp_sec=idx * 0.5,
        fps=2.0,
        frame_count=8,
        duration_sec=4.0,
        target_frame_index=idx,
        actual_frame_index=idx,
        width=360,
        height=640,
    )


def test_detect_records_timing_breakdown(tmp_path: Path) -> None:
    frames = []
    for idx in range(6):
        path = tmp_path / f"f{idx}.png"
        if idx == 3:
            _with_bar(path)
        else:
            _blank(path)
        frames.append(_frame(path, idx))

    evaluator = _StubToastEvaluator()
    detector = ToastMessageDetector(
        evaluator=evaluator,  # type: ignore[arg-type]
        preprocessor=GuiPreprocessor(artifact_dir=tmp_path / "pre"),
        enable_preprocess=True,
        top_k_candidates=1,
    )
    result = detector.detect(frames, task_id="toast_timing")
    assert evaluator.last_kwargs is not None
    assert not evaluator.last_kwargs.get("extra_image_paths")
    assert evaluator.last_kwargs["image_role_labels"] == [
        "图1:动作前完整帧（必须用于推断操作语义）",
        "图2:候选完整帧（toast出现时）",
    ]
    before_image = Path(str(evaluator.last_kwargs["before_image"]))
    after_image = Path(str(evaluator.last_kwargs["after_image"]))
    assert before_image.suffix.lower() == ".png"
    assert after_image.suffix.lower() == ".png"
    assert before_image.parent == tmp_path
    assert after_image.parent == tmp_path
    timing = result.timing or {}
    assert timing["frame_count"] == 6
    assert timing["vlm_call_count"] == 1
    assert timing["vlm_eval_elapsed_total_ms"] >= 0
    assert timing["scoring_elapsed_ms"] >= 0
    assert timing["preview_elapsed_ms"] == 0
    assert timing["detect_elapsed_ms"] >= timing["vlm_eval_elapsed_total_ms"]
    assert timing["vlm_calls"][0]["ok"] is True
    assert "idx" in timing["vlm_calls"][0]
    assert timing["vlm_max_long_edge"] == 0

