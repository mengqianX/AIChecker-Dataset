from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REGRESSION_DIR = Path(__file__).resolve().parent / "regression"
if str(_REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_DIR))

# 加载 .env 文件中的环境变量（如果存在）
try:
    from dotenv import load_dotenv
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    # 尝试从多个位置加载 .env 文件
    env_files = [
        REPO_ROOT / ".env",
        REPO_ROOT / "AIChecker" / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)  # override=False 表示不覆盖已存在的环境变量
            break
except ImportError:
    # python-dotenv 未安装时忽略
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = REPO_ROOT / "reports" / "history"

RESULT_FIELDNAMES = [
    "run_id",
    "run_at",
    "git_commit",
    "checker",
    "app",
    "case_id",
    "case_file",
    "template_image",
    "target_image",
    "threshold",
    "expected_passed",
    "expected_bounds",
    "actual_passed",
    "status",
    "similarity",
    "actual_bounds",
    "error",
    "duration_sec",
    "prompt_call_count",
    "calls_with_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "detect_elapsed_ms",
    "frame_extract_elapsed_ms",
    "scoring_elapsed_ms",
    "preview_elapsed_ms",
    "vlm_eval_elapsed_ms",
    "vlm_call_count",
    "vlm_call_details",
    "preview_template_image",
    "preview_target_image",
    "preview_match_result_image",
    "preview_button_before_image",
    "preview_button_after_image",
]

RUN_FIELDNAMES = [
    "run_id",
    "run_at",
    "branch",
    "commit",
    "note",
    "checker",
    "case_count",
    "executed_count",
    "skipped_count",
    "total_duration_sec",
    "avg_duration_sec",
    "total_prompt_calls",
    "cases_with_llm",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "avg_tokens_per_case",
    "avg_tokens_per_llm_case",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, cwd=REPO_ROOT).strip()
    except Exception:
        return "unknown"


def _ensure_csv_header(csv_file: Path, fieldnames: List[str]) -> None:
    if csv_file.exists():
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
        if existing_header == fieldnames:
            return
        existing_rows: List[Dict[str, Any]] = []
        if existing_header:
            with csv_file.open("r", newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_rows(csv_file: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    _ensure_csv_header(csv_file, fieldnames)
    with csv_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _to_expected_text(value: Any) -> str:
    if value is True:
        return "通过(Pass)"
    if value is False:
        return "失败(Fail)"
    return "N/A"


def _status_from_expected_actual(expected: str, actual: str) -> str:
    if expected == "N/A":
        return "预期未知(UNKNOWN_EXPECTED)"
    if actual in ("通过(Pass)", "失败(Fail)"):
        ok = (expected == "通过(Pass)" and actual == "通过(Pass)") or (
            expected == "失败(Fail)" and actual == "失败(Fail)"
        )
        return "一致(MATCH)" if ok else "不一致(MISMATCH)"
    if actual == "跳过(SKIPPED)":
        return "未知(UNKNOWN)"
    return "异常(ERROR)"


def _checker_from_test_path(test_path: str) -> str | None:
    if "testcase_image_match_test.py" in test_path or "test_image_match_cases.py" in test_path:
        return "image_match"
    if "test_image_match_feature_cases.py" in test_path:
        return "image_match_feature"
    if "test_button_cases.py" in test_path:
        return "button_color"
    if "test_count_change_cases.py" in test_path:
        return "count_change"
    if "test_progress_cases.py" in test_path:
        return "progress_change"
    if "test_toggle_cases.py" in test_path:
        return "toggle"
    if "test_black_white_screen_cases.py" in test_path:
        return "black_white_screen"
    if "test_no_response_cases.py" in test_path:
        return "no_response"
    if "test_long_loading_cases.py" in test_path:
        return "long_loading"
    if "test_page_load_failure_cases.py" in test_path:
        return "page_load_failure"
    if "test_list_refresh_cases.py" in test_path:
        return "list_refresh"
    if "test_seek_playback_cases.py" in test_path:
        return "seek_playback"
    if "test_toast_cases.py" in test_path:
        return "toast"
    if "test_video_play_cases.py" in test_path:
        return "video_play"
    return None


def _runs_csv_for_checker(checker: str) -> Path:
    return HISTORY_DIR / checker / "test_runs.csv"


def _results_csv_for_checker(checker: str) -> Path:
    return HISTORY_DIR / checker / "test_case_results.csv"


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _aggregate_checker_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    executed = [r for r in rows if str(r.get("actual_passed", "")) != "跳过(SKIPPED)"]
    skipped_count = len(rows) - len(executed)
    durations = [_as_float(r.get("duration_sec")) for r in executed]
    durations = [d for d in durations if d is not None]
    total_duration = sum(durations) if durations else 0.0
    avg_duration = (total_duration / len(durations)) if durations else 0.0

    total_prompt_calls = sum(_as_int(r.get("prompt_call_count")) for r in executed)
    total_prompt_tokens = sum(_as_int(r.get("prompt_tokens")) for r in executed)
    total_completion_tokens = sum(_as_int(r.get("completion_tokens")) for r in executed)
    total_tokens = sum(_as_int(r.get("total_tokens")) for r in executed)
    cases_with_llm = sum(1 for r in executed if _as_int(r.get("prompt_call_count")) > 0)

    return {
        "case_count": len(rows),
        "executed_count": len(executed),
        "skipped_count": skipped_count,
        "total_duration_sec": round(total_duration, 3),
        "avg_duration_sec": round(avg_duration, 3),
        "total_prompt_calls": total_prompt_calls,
        "cases_with_llm": cases_with_llm,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_case": round(total_tokens / len(executed), 2) if executed else 0.0,
        "avg_tokens_per_llm_case": (
            round(total_tokens / cases_with_llm, 2) if cases_with_llm else 0.0
        ),
    }


def _avg_ms(rows: List[Dict[str, Any]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt_avg_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _fmt_ms_cell(row: Dict[str, Any], key: str) -> str:
    return _fmt_avg_ms(_as_float(row.get(key)))


def _print_per_case_timings(rows: List[Dict[str, Any]]) -> None:
    timed = [
        row
        for row in rows
        if str(row.get("actual_passed", "")) != "跳过(SKIPPED)"
        and (
            _as_float(row.get("detect_elapsed_ms")) is not None
            or _as_float(row.get("vlm_eval_elapsed_ms")) is not None
        )
    ]
    if not timed:
        return
    timed.sort(key=lambda row: _as_float(row.get("duration_sec")) or 0.0, reverse=True)
    print(
        f"    {'case':<34} {'wall_s':>7} {'extract':>8} {'cv':>7} "
        f"{'preview':>8} {'vlm':>8} {'xn':>3} {'detect':>8}  calls"
    )
    for row in timed:
        wall = _as_float(row.get("duration_sec"))
        wall_txt = "-" if wall is None else f"{wall:.2f}"
        calls = row.get("vlm_call_count") or "-"
        details = str(row.get("vlm_call_details") or "").strip() or "-"
        print(
            f"    {str(row.get('case_id') or '')[:34]:<34} {wall_txt:>7} "
            f"{_fmt_ms_cell(row, 'frame_extract_elapsed_ms'):>8} "
            f"{_fmt_ms_cell(row, 'scoring_elapsed_ms'):>7} "
            f"{_fmt_ms_cell(row, 'preview_elapsed_ms'):>8} "
            f"{_fmt_ms_cell(row, 'vlm_eval_elapsed_ms'):>8} "
            f"{str(calls):>3} "
            f"{_fmt_ms_cell(row, 'detect_elapsed_ms'):>8}  {details}"
        )


def _print_runtime_summary(by_checker: Dict[str, List[Dict[str, Any]]]) -> None:
    if not by_checker:
        return
    print("\n======== Checker Runtime / Token Summary ========")
    print("注: avg_tok_llm = 仅对真正调用了 LLM 的用例求平均；avg_tok_all = 总token/全部执行用例(含0)")
    header = (
        f"{'checker':<22} {'n':>4} {'exec':>4} {'skip':>4} "
        f"{'total_s':>9} {'avg_s':>8} {'llm':>4} {'calls':>6} "
        f"{'tok_total':>10} {'avg_tok_llm':>11} {'avg_tok_all':>11}"
    )
    print(header)
    print("-" * len(header))
    for checker in sorted(by_checker):
        checker_rows = by_checker[checker]
        stats = _aggregate_checker_rows(checker_rows)
        print(
            f"{checker:<22} {stats['case_count']:4d} {stats['executed_count']:4d} "
            f"{stats['skipped_count']:4d} {stats['total_duration_sec']:9.2f} "
            f"{stats['avg_duration_sec']:8.2f} {stats['cases_with_llm']:4d} "
            f"{stats['total_prompt_calls']:6d} {stats['total_tokens']:10d} "
            f"{stats['avg_tokens_per_llm_case']:11.1f} {stats['avg_tokens_per_case']:11.1f}"
        )
        executed = [row for row in checker_rows if str(row.get("actual_passed", "")) != "跳过(SKIPPED)"]
        extract_avg = _avg_ms(executed, "frame_extract_elapsed_ms")
        scoring_avg = _avg_ms(executed, "scoring_elapsed_ms")
        preview_avg = _avg_ms(executed, "preview_elapsed_ms")
        vlm_avg = _avg_ms(executed, "vlm_eval_elapsed_ms")
        detect_avg = _avg_ms(executed, "detect_elapsed_ms")
        if any(value is not None for value in (extract_avg, scoring_avg, preview_avg, vlm_avg, detect_avg)):
            print(
                "  timing avg_ms: "
                f"extract={_fmt_avg_ms(extract_avg)} cv={_fmt_avg_ms(scoring_avg)} "
                f"preview={_fmt_avg_ms(preview_avg)} vlm={_fmt_avg_ms(vlm_avg)} detect={_fmt_avg_ms(detect_avg)}"
            )
            _print_per_case_timings(executed)
    print("=================================================\n")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    run_at = _now_iso()
    run_id = run_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    session.config._checker_report_run_meta = {
        "run_id": run_id,
        "run_at": run_at,
        "commit": _git(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }
    session.config._checker_report_rows = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    rep = outcome.get_result()

    test_path = str(item.fspath)
    checker = _checker_from_test_path(test_path)
    if checker is None:
        return
    # For skipped-by-marker in setup phase.
    if rep.when not in ("call", "setup"):
        return
    if hasattr(item, "_image_report_recorded"):
        return
    if rep.when == "setup" and not rep.skipped:
        return

    # Backward compatible: support old _image_report_meta and new _checker_report_meta.
    meta: Dict[str, Any] = getattr(item, "_checker_report_meta", {})
    if not meta:
        meta = getattr(item, "_image_report_meta", {})
    run_meta: Dict[str, str] = item.config._checker_report_run_meta  # type: ignore[attr-defined]

    app = str(meta.get("app", "unknown"))
    case_id = str(meta.get("case_id", item.name))
    case_file = str(meta.get("case_file", ""))
    expected_text = _to_expected_text(meta.get("expected_passed"))
    similarity = meta.get("similarity", "")
    actual_bounds = meta.get("actual_bounds", "")
    expected_bounds = meta.get("expected_bounds", "N/A")
    if isinstance(expected_bounds, list) and len(expected_bounds) == 4:
        expected_bounds = f"[{expected_bounds[0]}, {expected_bounds[1]}, {expected_bounds[2]}, {expected_bounds[3]}]"
    elif not expected_bounds:
        expected_bounds = "N/A"

    if rep.skipped:
        actual_text = "跳过(SKIPPED)"
        error_text = str(rep.longrepr) if rep.longrepr else ""
    elif rep.failed:
        if meta.get("actual_passed") in (True, False):
            actual_text = "通过(Pass)" if meta["actual_passed"] else "失败(Fail)"
        else:
            actual_text = "异常(ERROR)"
        error_text = str(rep.longrepr) if rep.longrepr else ""
    else:
        if meta.get("actual_passed") in (True, False):
            actual_text = "通过(Pass)" if meta["actual_passed"] else "失败(Fail)"
        else:
            actual_text = "通过(Pass)"
        error_text = ""

    duration_sec = ""
    if getattr(call, "duration", None) is not None:
        duration_sec = f"{float(call.duration):.4f}"

    status = _status_from_expected_actual(expected_text, actual_text)
    row = {
        "run_id": run_meta["run_id"],
        "run_at": run_meta["run_at"],
        "git_commit": run_meta["commit"],
        "checker": str(meta.get("checker", checker)),
        "app": app,
        "case_id": case_id,
        "case_file": case_file,
        "template_image": meta.get("template_image", ""),
        "target_image": meta.get("target_image", ""),
        "threshold": meta.get("threshold", ""),
        "expected_passed": expected_text,
        "expected_bounds": expected_bounds,
        "actual_passed": actual_text,
        "status": status,
        "similarity": similarity,
        "actual_bounds": actual_bounds,
        "error": error_text.replace("\n", " "),
        "duration_sec": duration_sec,
        "prompt_call_count": meta.get("prompt_call_count", ""),
        "calls_with_usage": meta.get("calls_with_usage", ""),
        "prompt_tokens": meta.get("prompt_tokens", ""),
        "completion_tokens": meta.get("completion_tokens", ""),
        "total_tokens": meta.get("total_tokens", ""),
        "detect_elapsed_ms": meta.get("detect_elapsed_ms", ""),
        "frame_extract_elapsed_ms": meta.get("frame_extract_elapsed_ms", ""),
        "scoring_elapsed_ms": meta.get("scoring_elapsed_ms", ""),
        "preview_elapsed_ms": meta.get("preview_elapsed_ms", ""),
        "vlm_eval_elapsed_ms": meta.get("vlm_eval_elapsed_ms", ""),
        "vlm_call_count": meta.get("vlm_call_count", ""),
        "vlm_call_details": meta.get("vlm_call_details", ""),
        "preview_template_image": meta.get("preview_template_image", ""),
        "preview_target_image": meta.get("preview_target_image", ""),
        "preview_match_result_image": meta.get("preview_match_result_image", ""),
        "preview_button_before_image": meta.get("preview_button_before_image", ""),
        "preview_button_after_image": meta.get("preview_button_after_image", ""),
    }
    item.config._checker_report_rows.append(row)  # type: ignore[attr-defined]
    item._image_report_recorded = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    rows: List[Dict[str, Any]] = getattr(session.config, "_checker_report_rows", [])
    run_meta: Dict[str, str] = getattr(session.config, "_checker_report_run_meta", {})
    if not run_meta:
        return

    by_checker: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        checker = str(row.get("checker", "")).strip()
        if not checker:
            continue
        by_checker.setdefault(checker, []).append(row)

    for checker, checker_rows in by_checker.items():
        stats = _aggregate_checker_rows(checker_rows)
        _append_rows(
            _runs_csv_for_checker(checker),
            RUN_FIELDNAMES,
            [
                {
                    "run_id": run_meta["run_id"],
                    "run_at": run_meta["run_at"],
                    "branch": run_meta["branch"],
                    "commit": run_meta["commit"],
                    "note": "pytest session",
                    "checker": checker,
                    **stats,
                }
            ],
        )
        _append_rows(
            _results_csv_for_checker(checker),
            RESULT_FIELDNAMES,
            checker_rows,
        )

    _print_runtime_summary(by_checker)
