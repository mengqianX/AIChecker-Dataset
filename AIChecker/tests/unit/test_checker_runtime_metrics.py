from __future__ import annotations

from types import SimpleNamespace

import pytest

from testagent_case_utils import record_cli_report_metrics, record_evaluator_token_usage, set_checker_report_meta


def test_record_evaluator_token_usage_writes_meta(request: pytest.FixtureRequest) -> None:
    evaluator = SimpleNamespace(
        get_token_usage_summary=lambda: {
            "prompt_call_count": 2,
            "calls_with_usage": 2,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
            "total_tokens": 120,
        }
    )
    record_evaluator_token_usage(request, evaluator)
    meta = request.node._checker_report_meta
    assert meta["prompt_call_count"] == 2
    assert meta["prompt_tokens"] == 100
    assert meta["completion_tokens"] == 20
    assert meta["total_tokens"] == 120


def test_record_cli_report_metrics_writes_meta(request: pytest.FixtureRequest) -> None:
    report = {
        "token_usage_summary": {
            "prompt_call_count": 1,
            "calls_with_usage": 1,
            "total_prompt_tokens": 50,
            "total_completion_tokens": 10,
            "total_tokens": 60,
        },
        "debug_artifacts": {"timing": {"total_elapsed_ms": 123.4}},
    }
    record_cli_report_metrics(request, report)
    meta = request.node._checker_report_meta
    assert meta["total_tokens"] == 60
    assert meta["detect_elapsed_ms"] == 123.4


def test_aggregate_checker_rows_runtime_and_tokens() -> None:
    # Import from conftest helpers via module path used by pytest.
    import importlib.util
    from pathlib import Path

    conftest_path = Path(__file__).resolve().parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("checker_conftest", conftest_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {
            "actual_passed": "通过(Pass)",
            "duration_sec": "1.5",
            "prompt_call_count": "1",
            "prompt_tokens": "100",
            "completion_tokens": "20",
            "total_tokens": "120",
        },
        {
            "actual_passed": "失败(Fail)",
            "duration_sec": "0.5",
            "prompt_call_count": "0",
            "prompt_tokens": "0",
            "completion_tokens": "0",
            "total_tokens": "0",
        },
        {
            "actual_passed": "跳过(SKIPPED)",
            "duration_sec": "9.0",
            "prompt_call_count": "9",
            "total_tokens": "999",
        },
    ]
    stats = mod._aggregate_checker_rows(rows)
    assert stats["case_count"] == 3
    assert stats["executed_count"] == 2
    assert stats["skipped_count"] == 1
    assert stats["total_duration_sec"] == 2.0
    assert stats["avg_duration_sec"] == 1.0
    assert stats["cases_with_llm"] == 1
    assert stats["total_tokens"] == 120
    assert stats["avg_tokens_per_llm_case"] == 120.0


def test_print_runtime_summary_lists_per_case_timing(capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util
    from pathlib import Path

    conftest_path = Path(__file__).resolve().parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("checker_conftest_timing", conftest_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod._print_runtime_summary(
        {
            "toast": [
                {
                    "case_id": "toast_omninote-2-f",
                    "actual_passed": "失败(Fail)",
                    "duration_sec": "17.3",
                    "frame_extract_elapsed_ms": "490",
                    "scoring_elapsed_ms": "310",
                    "preview_elapsed_ms": "90",
                    "vlm_eval_elapsed_ms": "16500",
                    "detect_elapsed_ms": "17000",
                    "vlm_call_count": "2",
                    "vlm_call_details": "idx5=8000ms,idx8=8500ms",
                    "prompt_call_count": "2",
                    "prompt_tokens": "2000",
                    "completion_tokens": "100",
                    "total_tokens": "2100",
                },
                {
                    "case_id": "toast_feishu-2",
                    "actual_passed": "通过(Pass)",
                    "duration_sec": "6.58",
                    "frame_extract_elapsed_ms": "480",
                    "scoring_elapsed_ms": "280",
                    "preview_elapsed_ms": "50",
                    "vlm_eval_elapsed_ms": "5600",
                    "detect_elapsed_ms": "5900",
                    "vlm_call_count": "1",
                    "vlm_call_details": "idx4=5600ms",
                    "prompt_call_count": "1",
                    "prompt_tokens": "2000",
                    "completion_tokens": "80",
                    "total_tokens": "2080",
                },
            ]
        }
    )
    out = capsys.readouterr().out
    assert "toast_omninote-2-f" in out
    assert "idx5=8000ms,idx8=8500ms" in out
    assert "toast_feishu-2" in out


def test_set_checker_report_meta_merges(request: pytest.FixtureRequest) -> None:
    set_checker_report_meta(request, checker="toast", app="x")
    set_checker_report_meta(request, total_tokens=3)
    meta = request.node._checker_report_meta
    assert meta["checker"] == "toast"
    assert meta["total_tokens"] == 3
