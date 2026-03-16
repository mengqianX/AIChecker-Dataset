from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

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
    return None


def _runs_csv_for_checker(checker: str) -> Path:
    return HISTORY_DIR / checker / "test_runs.csv"


def _results_csv_for_checker(checker: str) -> Path:
    return HISTORY_DIR / checker / "test_case_results.csv"


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
        _append_rows(
            _runs_csv_for_checker(checker),
            ["run_id", "run_at", "branch", "commit", "note", "checker", "case_count"],
            [
                {
                    "run_id": run_meta["run_id"],
                    "run_at": run_meta["run_at"],
                    "branch": run_meta["branch"],
                    "commit": run_meta["commit"],
                    "note": "pytest session",
                    "checker": checker,
                    "case_count": len(checker_rows),
                }
            ],
        )
        _append_rows(
            _results_csv_for_checker(checker),
            [
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
                "preview_template_image",
                "preview_target_image",
                "preview_match_result_image",
                "preview_button_before_image",
                "preview_button_after_image",
            ],
            checker_rows,
        )
