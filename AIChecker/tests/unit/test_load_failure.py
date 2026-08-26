"""Unit tests for LoadFailurePromptDetector observation normalization."""

from __future__ import annotations

from aichecker.vision.checkers.load_failure import LoadFailurePromptDetector


class _NoVlmEvaluator:
    def evaluate_json(self, **_kwargs):  # pragma: no cover - unused in these tests
        raise AssertionError("VLM should not be called in normalize unit tests")


def _detector() -> LoadFailurePromptDetector:
    return LoadFailurePromptDetector(evaluator=_NoVlmEvaluator())


def test_normalize_uses_failure_text_visible_without_keyword_match() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "加载出错 请点击重试",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "个人页内容区显示加载出错重试提示",
            "confidence": 0.9,
        }
    )
    assert result["load_failed"] is True
    assert result["text_failure_type"] == "request_failed"
    assert result["failure_type"] == "request_failed"


def test_normalize_blank_visual_state_without_text_is_not_vlm_hit() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": False,
            "evidence_text": "",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "后帧仍为大面积空白，仅显示还没有内容",
            "confidence": 0.8,
        }
    )
    assert result["load_failed"] is False
    assert result["failure_type"] == "none"


def test_normalize_ignores_failure_flag_when_reason_denies_text() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "图2和图3均未显示任何明确的失败文案或错误弹窗。页面仅显示还没有内容。",
            "confidence": 0.7,
        }
    )
    assert result["load_failed"] is False
    assert result["failure_type"] == "none"


def test_normalize_page_recovered_clears_transient_failure() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "网络异常",
            "page_recovered": True,
            "visual_state": "normal",
            "reason": "后帧已恢复列表内容",
            "confidence": 0.7,
        }
    )
    assert result["load_failed"] is False
    assert result["failure_type"] == "none"


def test_normalize_loading_only_is_not_failure() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": False,
            "evidence_text": "",
            "page_recovered": False,
            "visual_state": "loading",
            "reason": "仅见加载动画",
            "confidence": 0.6,
        }
    )
    assert result["load_failed"] is False
    assert result["failure_type"] == "none"


def test_classify_keywords_still_used_for_typed_failure() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "网络异常，请稍后重试",
            "page_recovered": False,
            "visual_state": "normal",
            "reason": "出现网络异常提示",
            "confidence": 0.85,
        }
    )
    assert result["load_failed"] is True
    assert result["failure_type"] == "network_error"


def test_normalize_service_error_toast_without_keyword() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "服务异常，请稍后再试",
            "page_recovered": False,
            "visual_state": "normal",
            "reason": "底部出现服务异常 toast",
            "confidence": 0.88,
        }
    )
    assert result["load_failed"] is True
    assert result["failure_type"] == "request_failed"


def test_normalize_soft_failure_copy_seems_wrong() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "似乎出了点问题",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "关注列表页显示似乎出了点问题和重新加载",
            "confidence": 0.9,
        }
    )
    assert result["load_failed"] is True
    assert result["failure_type"] == "request_failed"


def test_normalize_cannot_connect_network() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "无法连接到网络",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "整页网络错误并提供立即重试",
            "confidence": 0.92,
        }
    )
    assert result["load_failed"] is True
    assert result["failure_type"] == "network_error"


def test_normalize_trusts_vlm_even_when_evidence_has_no_keyword() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "页面打不开啦，点这里再试一次",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "错误页提示页面打不开并提供再试",
            "confidence": 0.8,
        }
    )
    assert result["load_failed"] is True
    assert result["failure_type"] == "request_failed"


def test_normalize_demotes_achievement_popup_evidence() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "你的累计阅读数超过 200 万",
            "page_recovered": False,
            "visual_state": "normal",
            "reason": "成就弹窗被误认为失败提示",
            "confidence": 0.9,
        }
    )
    assert result["load_failed"] is False
    assert result["failure_text_visible"] is False


def test_normalize_demotes_empty_filter_result_with_retry() -> None:
    detector = _detector()
    result = detector._normalize_probe_result(
        {
            "failure_text_visible": True,
            "evidence_text": "没有符合条件的订单 点击重试",
            "page_recovered": False,
            "visual_state": "blank",
            "reason": "空结果页被误认为加载失败",
            "confidence": 0.9,
        }
    )
    assert result["load_failed"] is False
    assert result["failure_text_visible"] is False


def test_select_probe_indices_always_keeps_first_candidate() -> None:
    detector = _detector()
    scores = [
        {"index": 11, "cv_score": 0.9},
        {"index": 10, "cv_score": 0.8},
        {"index": 9, "cv_score": 0.7},
        {"index": 0, "cv_score": 0.01},
    ]
    assert detector._select_probe_indices(scores) == [0, 9, 10, 11]
