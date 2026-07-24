import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_CASE_DIR = REPO_ROOT / "testcase" / "image_match" / "jsons"
REPORT_DIR = REPO_ROOT / "reports"
HISTORY_DIR = REPORT_DIR / "history"
LEGACY_RUNS_CSV = HISTORY_DIR / "test_runs.csv"
LEGACY_RESULTS_CSV = HISTORY_DIR / "test_case_results.csv"
DEFAULT_IMAGE_OUTPUT_ROOT = REPO_ROOT / "AIChecker" / "tests" / "image_match_output"
SUPPORTED_CHECKERS = ("image_match", "button_color", "count_change", "progress_change", "video_play")
PAIR_IMAGE_CHECKERS = frozenset({"count_change", "progress_change"})
NO_PREVIEW_IMAGE_CHECKERS = frozenset({"video_play"})
ISSUE_STATUSES = {"不一致(MISMATCH)", "异常(ERROR)"}
GOOD_STATUS = "一致(MATCH)"
UNKNOWN_STATUSES = {"未知(UNKNOWN)", "预期未知(UNKNOWN_EXPECTED)"}
STATUS_PRIORITY = {
    "异常(ERROR)": 0,
    "不一致(MISMATCH)": 1,
    "跳过(SKIPPED)": 2,
    "未知(UNKNOWN)": 3,
    "预期未知(UNKNOWN_EXPECTED)": 4,
    "一致(MATCH)": 5,
}
HISTORY_LABEL_PRIORITY = {
    "新回归": 0,
    "持续失败": 1,
    "首次失败": 2,
    "首次异常": 3,
    "已修复": 4,
    "波动中": 5,
    "稳定通过": 6,
    "首次通过": 7,
    "待确认": 8,
    "N/A": 9,
}


def testagent_root() -> Path:
    return Path(os.getenv("TESTAGENT_ROOT", str(REPO_ROOT.parent / "TestAgent"))).resolve()


CASE_JSON_ROOTS = {
    "image_match": REPO_ROOT / "testcase" / "image_match" / "jsons",
    "button_color": REPO_ROOT / "testcase" / "button_color_change" / "jsons",
    "count_change": REPO_ROOT / "testcase" / "count_change" / "jsons",
    "progress_change": REPO_ROOT / "testcase" / "progress_bar_change" / "jsons",
    "video_play": testagent_root() / "testcase" / "video_play" / "json",
}


def report_paths_for_checker(checker: str) -> Dict[str, Path]:
    out_dir = REPORT_DIR / checker
    return {
        "dir": out_dir,
        "latest": out_dir / "LATEST_SNAPSHOT.html",
        "timeline": out_dir / "HISTORY_TIMELINE.html",
        "failure": out_dir / "FAILURE_VIEW.html",
    }


def history_paths_for_checker(checker: str) -> Dict[str, Path]:
    out_dir = HISTORY_DIR / checker
    return {
        "dir": out_dir,
        "runs_csv": out_dir / "test_runs.csv",
        "results_csv": out_dir / "test_case_results.csv",
    }


def infer_checker(row: Dict[str, Any]) -> str:
    checker = str(row.get("checker", "")).strip()
    if checker in SUPPORTED_CHECKERS:
        return checker
    case_file = str(row.get("case_file", ""))
    lowered = case_file.lower()
    if "image_match" in lowered:
        return "image_match"
    if "button_color_change" in lowered or "test_button_cases.py" in lowered:
        return "button_color"
    if "count_change" in lowered or "test_count_change_cases.py" in lowered:
        return "count_change"
    if "progress_change" in lowered or "test_progress_cases.py" in lowered:
        return "progress_change"
    if "video_play" in lowered or "test_video_play_cases.py" in lowered:
        return "video_play"
    return "unknown"


def _read_csv_rows_if_exists(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_row_matches_checker(run_row: Dict[str, str], checker: str) -> bool:
    checker_value = str(run_row.get("checker", "")).strip()
    if not checker_value:
        return True
    parts = [p.strip() for p in checker_value.split(",") if p.strip()]
    if not parts:
        return True
    return checker in parts


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_git_info() -> Dict[str, str]:
    def read_cmd(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, cwd=str(REPO_ROOT)).strip()
        except Exception:
            return "unknown"

    return {
        "commit": read_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "branch": read_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def try_import_checker():
    try:
        from AIChecker.aichecker.checkers.image_match_checker import check_image_match

        return check_image_match, None
    except Exception as exc:
        return None, str(exc)


def expected_to_text(value: Any) -> str:
    if value is True:
        return "通过(Pass)"
    if value is False:
        return "失败(Fail)"
    return "N/A"


def format_bounds(bounds: Any) -> str:
    if isinstance(bounds, list) and len(bounds) == 4:
        return f"[{bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}]"
    return "N/A"


def resolve_data_path(case_file: Path, rel: str) -> str:
    return str((case_file.parent / rel).resolve())


def discover_cases() -> List[Path]:
    files = sorted(ROOT_CASE_DIR.glob("**/*.json"))
    return [p for p in files if p.is_file()]


def load_case(case_file: Path) -> Dict[str, Any]:
    with case_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(case_file: Path, payload: Dict[str, Any], checker, checker_error: Optional[str]) -> Dict[str, Any]:
    case_id = case_file.stem
    app = case_file.parent.name
    template_raw = str(payload.get("template_image", "N/A"))
    target_raw = str(payload.get("target_image", "N/A"))
    expected_passed = payload.get("expected_passed")
    expected_bounds = payload.get("expected_bounds")
    threshold = payload.get("similarity_threshold", 0.9)

    row = {
        "app": app,
        "case_id": case_id,
        "case_file": str(case_file),
        "template_image": os.path.basename(template_raw),
        "target_image": os.path.basename(target_raw),
        "threshold": str(threshold),
        "expected_passed": expected_to_text(expected_passed),
        "expected_bounds": format_bounds(expected_bounds),
        "actual_passed": "跳过(SKIPPED)",
        "status": "未知(UNKNOWN)",
        "similarity": "",
        "actual_bounds": "",
        "error": "",
    }

    if checker is None:
        row["error"] = f"检查器不可用: {checker_error}"
        return row

    try:
        run_payload = dict(payload)
        if "template_image" in run_payload:
            run_payload["template_image"] = resolve_data_path(case_file, run_payload["template_image"])
        if "target_image" in run_payload:
            run_payload["target_image"] = resolve_data_path(case_file, run_payload["target_image"])
        result = checker(run_payload)
        actual_passed = bool(result.passed)
        row["actual_passed"] = "通过(Pass)" if actual_passed else "失败(Fail)"
        row["similarity"] = f"{float(result.details.get('similarity', 0.0)):.4f}"
        row["actual_bounds"] = format_bounds(result.details.get("bounds"))
        if expected_passed in (True, False):
            row["status"] = "一致(MATCH)" if actual_passed == expected_passed else "不一致(MISMATCH)"
        else:
            row["status"] = "预期未知(UNKNOWN_EXPECTED)"
    except Exception as exc:
        row["actual_passed"] = "异常(ERROR)"
        row["status"] = "异常(ERROR)"
        row["error"] = str(exc)
    return row


def ensure_csv_header(csv_file: Path, fieldnames: List[str]) -> None:
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


def append_csv_rows(csv_file: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    ensure_csv_header(csv_file, fieldnames)
    with csv_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def html_page(title: str, body: str, script: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{escape(title)}</title>
<style>
:root{{color-scheme:light;--bg:#f3f7fb;--card:#ffffff;--border:#dbe5f0;--text:#18212f;--muted:#65758b;--accent:#0f766e;--accent-soft:#e6fffa;--danger:#b42318;--danger-soft:#fef3f2;--warning:#b54708;--warning-soft:#fff7ed;--ok:#166534;--ok-soft:#ecfdf3;--shadow:0 10px 30px rgba(15,23,42,0.06)}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;padding:24px;line-height:1.5;color:var(--text);background:linear-gradient(180deg,#f8fbff 0%,var(--bg) 100%)}}
.page{{max-width:1600px;margin:0 auto}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;margin-bottom:18px;box-shadow:var(--shadow)}}
.hero{{background:linear-gradient(135deg,#ffffff 0%,#eef8ff 58%,#f6fffb 100%)}}
h1,h2,h3{{margin:0 0 12px}}
h1{{font-size:28px;line-height:1.2}}
h2{{font-size:19px;margin-top:0}}
h3{{font-size:15px;margin-bottom:8px}}
p{{margin:0}}
ul{{margin:0;padding-left:20px}}
.muted{{color:var(--muted)}}
.meta-list{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px 18px;padding:0;list-style:none}}
.meta-list li{{background:rgba(255,255,255,0.72);border:1px solid var(--border);border-radius:12px;padding:10px 12px}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.metric-card{{border:1px solid var(--border);border-radius:14px;padding:14px;background:#fbfdff}}
.metric-label{{font-size:12px;color:var(--muted);margin-bottom:6px}}
.metric-value{{font-size:26px;font-weight:700;line-height:1.1}}
.metric-sub{{font-size:12px;color:var(--muted);margin-top:6px}}
.metric-card.bad{{background:var(--danger-soft);border-color:#f2c7c3}}
.metric-card.ok{{background:var(--ok-soft);border-color:#bbe6ca}}
.metric-card.warn{{background:var(--warning-soft);border-color:#f4d9b3}}
.tag{{display:inline-flex;align-items:center;gap:6px;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid transparent}}
.tag-ok{{color:var(--ok);background:var(--ok-soft);border-color:#bbe6ca}}
.tag-bad{{color:var(--danger);background:var(--danger-soft);border-color:#f2c7c3}}
.tag-unknown{{color:var(--warning);background:var(--warning-soft);border-color:#f4d9b3}}
.tag-info{{color:#155eef;background:#eff4ff;border-color:#c7d7fe}}
.toolbar{{display:flex;flex-wrap:wrap;align-items:center;gap:10px}}
.toolbar input,.toolbar select{{border:1px solid #cbd5e1;border-radius:10px;padding:9px 12px;background:#fff;min-height:40px;font:inherit;color:var(--text)}}
.toolbar input{{min-width:min(340px,100%);flex:1 1 280px}}
.toolbar button{{border:1px solid #cbd5e1;border-radius:10px;background:#fff;padding:9px 12px;cursor:pointer;font:inherit}}
.toolbar-stats{{margin-left:auto;color:var(--muted);font-size:13px}}
.split{{display:grid;grid-template-columns:1.35fr 1fr;gap:18px}}
.stack{{display:grid;gap:18px}}
.app-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.app-summary-scroll{{max-height:720px;overflow-y:auto;overflow-x:hidden;padding-right:6px;scrollbar-gutter:stable}}
.mini-card{{border:1px solid var(--border);border-radius:14px;padding:14px;background:#fbfdff}}
.mini-kpis{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}
.mini-kpis span{{font-size:12px;padding:3px 8px;border-radius:999px;background:#eef2ff;color:#334155}}
.table-wrap{{overflow:auto;border:1px solid var(--border);border-radius:14px;background:#fff}}
table{{border-collapse:collapse;width:100%;min-width:980px}}
th,td{{border-bottom:1px solid #e8eef5;padding:9px 10px;text-align:left;font-size:13px;white-space:nowrap;vertical-align:top}}
thead th{{position:sticky;top:0;background:#f8fbff;z-index:1}}
tbody tr:hover{{background:#f8fbff}}
tr.issue-row{{background:#fff9f8}}
tr.recovered-row{{background:#f4fff7}}
code{{background:#f2f6fb;padding:2px 6px;border-radius:6px;font-size:12px}}
.subtle-code{{display:inline-block;background:#f8fafc;color:#334155;border:1px solid #e2e8f0}}
.error-cell{{white-space:normal!important;overflow-wrap:anywhere;word-break:break-word;min-width:320px;max-width:560px;line-height:1.5}}
.error-summary-line{{display:block;margin-bottom:3px}}
.error-summary-line:last-child{{margin-bottom:0}}
.error-details{{margin-top:6px}}
.error-details > summary{{cursor:pointer;color:#155eef;font-size:12px;user-select:none}}
.error-details > summary:hover{{text-decoration:underline}}
.error-full{{margin:6px 0 0;padding:8px 10px;background:#f8fbff;border:1px solid var(--border);border-radius:8px;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:12px;line-height:1.45}}
.secondary{{color:var(--muted);font-size:12px}}
.section-head{{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin-bottom:12px}}
.anchor-links{{display:flex;flex-wrap:wrap;gap:8px}}
.anchor-links a{{display:inline-flex;padding:7px 10px;border-radius:999px;border:1px solid var(--border);background:#fff;color:var(--text);text-decoration:none;font-size:12px}}
.empty{{padding:18px;color:var(--muted)}}
img{{display:block}}
.img-thumb{{border:1px solid #d7e2ee;border-radius:8px;background:#fff}}
.nowrap{{white-space:nowrap}}
@media (max-width: 960px) {{
  body{{padding:16px}}
  .split{{grid-template-columns:1fr}}
  .toolbar-stats{{width:100%;margin-left:0}}
  .meta-list{{grid-template-columns:1fr}}
}}
</style>
</head><body><div class="page">{body}</div>{script}</body></html>"""


def status_class(status: str) -> str:
    if GOOD_STATUS in status:
        return "tag-ok"
    if status in ISSUE_STATUSES:
        return "tag-bad"
    return "tag-unknown"


def is_issue_status(status: str) -> bool:
    return status in ISSUE_STATUSES


def is_good_status(status: str) -> bool:
    return status == GOOD_STATUS


def status_priority(status: str) -> int:
    return STATUS_PRIORITY.get(status, 99)


def history_label_priority(label: str) -> int:
    return HISTORY_LABEL_PRIORITY.get(label, 99)


def format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def extract_core_error_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    signal_pattern = re.compile(
        r"(AssertionError:|RuntimeError:|ValueError:|TypeError:|KeyError:|ImportError:|FileNotFoundError:|ModuleNotFoundError:|NameError:|Skipped:|(?<![A-Za-z])Error:)"
    )
    matches = list(signal_pattern.finditer(text))
    if matches:
        text = text[matches[-1].start():]
    text = text.replace("E       ", "").replace("E   ", "").strip()
    cut_patterns = [
        r"\s+E\s+assert\s+.+$",
        r"\s+\+\s+where\s+.+$",
        r"\s+tests?/[\w./:-]+.*$",
    ]
    for pattern in cut_patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


def summarize_error_message(value: Any) -> str:
    core_text = extract_core_error_text(value)
    if not core_text:
        return "-"

    assertion_match = re.match(
        r"^(?P<exc>AssertionError):\s*(?P<case>[^:]+):\s*expected passed=(?P<expected>[^,]+).*?got passed=(?P<actual>[^,]+),\s*basis=(?P<basis>.+)$",
        core_text,
    )
    if assertion_match:
        case_name = assertion_match.group("case").strip()
        expected = assertion_match.group("expected").strip()
        actual = assertion_match.group("actual").strip()
        basis = assertion_match.group("basis").strip()
        return "\n".join(
            [
                assertion_match.group("exc"),
                f"用例: {case_name}",
                f"预期: {expected}",
                f"实际: {actual}",
                f"依据: {basis}",
            ]
        )

    generic_match = re.match(
        r"^(?P<exc>AssertionError|RuntimeError|ValueError|TypeError|KeyError|ImportError|FileNotFoundError|ModuleNotFoundError|NameError|Skipped|Error):\s*(?P<message>.+)$",
        core_text,
    )
    if generic_match:
        exc_name = generic_match.group("exc").strip()
        message = generic_match.group("message").strip()
        return "\n".join([exc_name, f"信息: {message}"])

    return core_text


def compact_error_summary(value: Any, max_len: int = 180) -> str:
    summary = clean_text(summarize_error_message(value).replace("\n", " | "))
    if len(summary) <= max_len:
        return summary
    return summary[: max_len - 1] + "…"


def readable_timestamp(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def existing_path_str(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw)
    except Exception:
        return ""
    return str(p.resolve()) if p.exists() else ""


@lru_cache(maxsize=None)
def case_lookup_for_checker(checker: str) -> Dict[tuple[str, str], Path]:
    root = CASE_JSON_ROOTS.get(checker)
    if root is None or not root.exists():
        return {}
    lookup: Dict[tuple[str, str], Path] = {}
    for case_file in sorted(root.rglob("*.json")):
        rel = case_file.relative_to(root)
        app = rel.parts[0] if len(rel.parts) > 1 else case_file.parent.name
        lookup[(app.lower(), case_file.stem.lower())] = case_file
    return lookup


def resolve_case_file_for_row(row: Dict[str, Any]) -> Optional[Path]:
    raw_case_file = str(row.get("case_file", "")).strip()
    if raw_case_file:
        case_path = Path(raw_case_file)
        if case_path.exists():
            return case_path
    checker = infer_checker(row)
    lookup = case_lookup_for_checker(checker)
    app = str(row.get("app", "")).strip().lower()
    case_id = str(row.get("case_id", "")).strip().lower()
    if not case_id:
        return None
    if app and (app, case_id) in lookup:
        return lookup[(app, case_id)]
    for (lookup_app, lookup_case), path in lookup.items():
        if lookup_case == case_id and (not app or lookup_app == app):
            return path
    return None


def resolve_case_image(case_file: Path, raw_path: Any) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        p = (case_file.parent / p).resolve()
    return str(p) if p.exists() else ""


def resolve_output_images(image_output_root: Path, app: str, case_id: str) -> Dict[str, str]:
    candidates = [
        image_output_root / app / case_id,
        image_output_root / app,
        image_output_root / case_id,
    ]
    image_names = {
        "preview_template_image": "template.png",
        "preview_target_image": "target.png",
        "preview_match_result_image": "match_result.png",
    }
    for base in candidates:
        if not base.exists():
            continue
        result: Dict[str, str] = {}
        found_any = False
        for key, filename in image_names.items():
            p = base / filename
            if p.exists():
                result[key] = str(p.resolve())
                found_any = True
            else:
                result[key] = ""
        if found_any:
            return result
    return {key: "" for key in image_names}


def repair_preview_images(rows: List[Dict[str, Any]], image_output_root: Path) -> None:
    payload_cache: Dict[Path, Dict[str, Any]] = {}
    for row in rows:
        checker = infer_checker(row)
        case_file = resolve_case_file_for_row(row)
        existing_template = existing_path_str(row.get("preview_template_image", ""))
        existing_target = existing_path_str(row.get("preview_target_image", ""))
        existing_match = existing_path_str(row.get("preview_match_result_image", ""))
        existing_button_before = existing_path_str(row.get("preview_button_before_image", ""))
        existing_button_after = existing_path_str(row.get("preview_button_after_image", ""))

        if existing_template:
            row["preview_template_image"] = existing_template
        if existing_target:
            row["preview_target_image"] = existing_target
        if existing_match:
            row["preview_match_result_image"] = existing_match
        if existing_button_before:
            row["preview_button_before_image"] = existing_button_before
        if existing_button_after:
            row["preview_button_after_image"] = existing_button_after

        payload: Optional[Dict[str, Any]] = None
        if case_file is not None:
            try:
                payload = payload_cache.get(case_file)
                if payload is None:
                    payload = load_case(case_file)
                    payload_cache[case_file] = payload
            except Exception:
                payload = None

        if payload is not None:
            if not row.get("preview_template_image"):
                if checker == "video_play":
                    row["preview_template_image"] = resolve_case_image(case_file, payload.get("video_file", ""))
                else:
                    key = "template_image" if checker == "image_match" else "screenshot_a"
                    row["preview_template_image"] = resolve_case_image(case_file, payload.get(key, ""))
            if not row.get("preview_target_image") and checker != "video_play":
                key = "target_image" if checker == "image_match" else "screenshot_b"
                row["preview_target_image"] = resolve_case_image(case_file, payload.get(key, ""))

        if checker == "image_match" and not row.get("preview_match_result_image"):
            row.update({k: v or row.get(k, "") for k, v in resolve_output_images(image_output_root, str(row.get("app", "")), str(row.get("case_id", ""))).items()})

        if checker == "button_color":
            debug_dir = REPO_ROOT / "AIChecker" / "debug" / "crops" / f"{row.get('app', '')}_{row.get('case_id', '')}"
            before_crop = debug_dir / "button_crop_before.png"
            after_crop = debug_dir / "button_crop_after.png"
            if before_crop.exists():
                row["preview_button_before_image"] = str(before_crop.resolve())
            if after_crop.exists():
                row["preview_button_after_image"] = str(after_crop.resolve())


def build_case_history_stats(rows: Iterable[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    by_case: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("app", "")), str(row.get("case_id", "")))
        by_case[key].append(row)

    stats: Dict[tuple[str, str], Dict[str, Any]] = {}
    for key, case_rows in by_case.items():
        ordered = sorted(case_rows, key=lambda item: (str(item.get("run_at", "")), str(item.get("run_id", ""))))
        statuses = [str(item.get("status", "")) for item in ordered]
        latest = ordered[-1]

        first_failed_at = ""
        first_fixed_at = ""
        failed_seen = False
        for item in ordered:
            status = str(item.get("status", ""))
            if not failed_seen and is_issue_status(status):
                first_failed_at = str(item.get("run_at", ""))
                failed_seen = True
            if failed_seen and is_good_status(status):
                first_fixed_at = str(item.get("run_at", ""))
                break

        failure_streak = 0
        for status in reversed(statuses):
            if is_issue_status(status):
                failure_streak += 1
            else:
                break

        last_passed_at = ""
        last_failed_at = ""
        for item in reversed(ordered):
            status = str(item.get("status", ""))
            if not last_passed_at and is_good_status(status):
                last_passed_at = str(item.get("run_at", ""))
            if not last_failed_at and is_issue_status(status):
                last_failed_at = str(item.get("run_at", ""))
            if last_passed_at and last_failed_at:
                break

        recent_statuses = statuses[-5:]
        recent_buckets = [
            "good" if is_good_status(status) else "issue" if is_issue_status(status) else "other"
            for status in recent_statuses
        ]
        recent_flip_count = sum(1 for prev, cur in zip(recent_buckets, recent_buckets[1:]) if prev != cur)

        stats[key] = {
            "case_id": key[1],
            "app": key[0],
            "total_runs": len(ordered),
            "latest_status": str(latest.get("status", "")),
            "latest_run_id": str(latest.get("run_id", "")),
            "latest_run_at": str(latest.get("run_at", "")),
            "first_failed_at": first_failed_at,
            "first_fixed_at": first_fixed_at,
            "last_passed_at": last_passed_at,
            "last_failed_at": last_failed_at,
            "failure_streak": failure_streak,
            "recent_flip_count": recent_flip_count,
            "recent_statuses": recent_statuses,
        }
    return stats


def infer_history_label(current_status: str, previous_status: str, recent_flip_count: int) -> str:
    if is_issue_status(current_status):
        if is_good_status(previous_status):
            return "新回归"
        if is_issue_status(previous_status):
            return "持续失败"
        if current_status == "异常(ERROR)":
            return "首次异常"
        return "首次失败"
    if is_good_status(current_status):
        if is_issue_status(previous_status):
            return "已修复"
        if is_good_status(previous_status):
            if recent_flip_count > 0:
                return "波动中"
            return "稳定通过"
        return "首次通过"
    return "待确认"


def enrich_rows_with_history(rows: List[Dict[str, Any]], history_rows: List[Dict[str, Any]], current_run_id: str) -> None:
    history_by_case: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in history_rows:
        if str(row.get("run_id", "")) == current_run_id:
            continue
        key = (str(row.get("app", "")), str(row.get("case_id", "")))
        history_by_case[key].append(row)

    history_stats = build_case_history_stats(history_rows)
    for row in rows:
        key = (str(row.get("app", "")), str(row.get("case_id", "")))
        previous_rows = sorted(
            history_by_case.get(key, []),
            key=lambda item: (str(item.get("run_at", "")), str(item.get("run_id", ""))),
        )
        previous = previous_rows[-1] if previous_rows else {}
        previous_status = str(previous.get("status", ""))

        combined_recent = previous_rows[-4:] + [row]
        combined_stats = build_case_history_stats(combined_recent).get(key, {})
        recent_flip_count = int(combined_stats.get("recent_flip_count", 0))
        failure_streak = int(combined_stats.get("failure_streak", 0))

        label = infer_history_label(str(row.get("status", "")), previous_status, recent_flip_count)
        row["history_label"] = label
        row["previous_status"] = previous_status or "-"
        row["last_good_at"] = combined_stats.get("last_passed_at", "") or "-"
        row["last_issue_at"] = combined_stats.get("last_failed_at", "") or "-"
        row["failure_streak"] = str(failure_streak if failure_streak else "")
        row["recent_flip_count"] = str(recent_flip_count if recent_flip_count else "")
        row["error_summary"] = summarize_error_message(row.get("error", ""))
        row["error_signature"] = compact_error_summary(row.get("error", ""))

        reference_stats = history_stats.get(key, {})
        if not row.get("last_good_at") or row["last_good_at"] == "-":
            row["last_good_at"] = reference_stats.get("last_passed_at", "") or "-"
        if not row.get("last_issue_at") or row["last_issue_at"] == "-":
            row["last_issue_at"] = reference_stats.get("last_failed_at", "") or "-"


def sort_rows_for_display(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            status_priority(str(row.get("status", ""))),
            history_label_priority(str(row.get("history_label", "N/A"))),
            str(row.get("app", "")).lower(),
            str(row.get("case_id", "")).lower(),
        ),
    )


def count_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    rows_list = list(rows)
    return {
        "total": len(rows_list),
        "match": sum(1 for row in rows_list if str(row.get("status", "")) == GOOD_STATUS),
        "mismatch": sum(1 for row in rows_list if str(row.get("status", "")) == "不一致(MISMATCH)"),
        "error": sum(1 for row in rows_list if str(row.get("status", "")) == "异常(ERROR)"),
        "skipped": sum(1 for row in rows_list if str(row.get("actual_passed", "")) == "跳过(SKIPPED)"),
        "unknown": sum(1 for row in rows_list if str(row.get("status", "")) in UNKNOWN_STATUSES),
        "issues": sum(1 for row in rows_list if is_issue_status(str(row.get("status", "")))),
    }


def render_metric_grid(metrics: List[Dict[str, str]]) -> str:
    cards = []
    for metric in metrics:
        variant = metric.get("variant", "")
        cards.append(
            f'<div class="metric-card {escape(variant)}">'
            f'<div class="metric-label">{escape(metric["label"])}</div>'
            f'<div class="metric-value">{escape(metric["value"])}</div>'
            f'<div class="metric-sub">{escape(metric.get("sub", ""))}</div>'
            "</div>"
        )
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def render_filter_controls(scope: str, rows: Iterable[Dict[str, Any]], placeholder: str) -> str:
    rows_list = list(rows)
    apps = sorted({str(row.get("app", "")).strip() for row in rows_list if str(row.get("app", "")).strip()})
    statuses = sorted({str(row.get("status", "")).strip() for row in rows_list if str(row.get("status", "")).strip()}, key=status_priority)
    history_labels = sorted(
        {str(row.get("history_label", "")).strip() for row in rows_list if str(row.get("history_label", "")).strip()},
        key=history_label_priority,
    )
    app_options = "".join(f'<option value="{escape(app)}">{escape(app)}</option>' for app in apps)
    status_options = "".join(f'<option value="{escape(status)}">{escape(status)}</option>' for status in statuses)
    history_options = "".join(f'<option value="{escape(label)}">{escape(label)}</option>' for label in history_labels)
    return (
        '<div class="card"><div class="section-head">'
        '<div><h2>快速筛选</h2><p class="muted">支持按应用、状态、历史标签和关键字过滤，排查大批量结果更快。</p></div>'
        "</div>"
        f'<div class="toolbar" data-filter-scope="{escape(scope)}">'
        f'<input type="search" data-filter-input="{escape(scope)}" placeholder="{escape(placeholder)}" />'
        f'<select data-filter-app="{escape(scope)}"><option value="">全部应用</option>{app_options}</select>'
        f'<select data-filter-status="{escape(scope)}"><option value="">全部状态</option>{status_options}</select>'
        f'<select data-filter-history="{escape(scope)}"><option value="">全部历史标签</option>{history_options}</select>'
        f'<button type="button" data-filter-reset="{escape(scope)}">重置</button>'
        f'<div class="toolbar-stats">显示 <strong data-filter-count="{escape(scope)}">{len(rows_list)}</strong> / <strong data-filter-total="{escape(scope)}">{len(rows_list)}</strong></div>'
        "</div></div>"
    )


def build_filter_script() -> str:
    return """<script>
document.addEventListener("DOMContentLoaded", function () {
  function apply(scope) {
    const input = document.querySelector('[data-filter-input="' + scope + '"]');
    const app = document.querySelector('[data-filter-app="' + scope + '"]');
    const status = document.querySelector('[data-filter-status="' + scope + '"]');
    const history = document.querySelector('[data-filter-history="' + scope + '"]');
    const rows = Array.from(document.querySelectorAll('tr[data-row-scope="' + scope + '"]'));
    let visible = 0;
    rows.forEach(function (row) {
      const haystack = [
        row.dataset.app || "",
        row.dataset.case || "",
        row.dataset.status || "",
        row.dataset.history || "",
        row.dataset.error || ""
      ].join(" ").toLowerCase();
      const textOk = !input || !input.value || haystack.includes(input.value.toLowerCase());
      const appOk = !app || !app.value || row.dataset.app === app.value;
      const statusOk = !status || !status.value || row.dataset.status === status.value;
      const historyOk = !history || !history.value || row.dataset.history === history.value;
      const show = textOk && appOk && statusOk && historyOk;
      row.hidden = !show;
      if (show) visible += 1;
    });
    document.querySelectorAll('[data-app-card="' + scope + '"]').forEach(function (card) {
      const hasVisible = card.querySelector('tr[data-row-scope="' + scope + '"]:not([hidden])');
      card.hidden = !hasVisible;
    });
    const countNode = document.querySelector('[data-filter-count="' + scope + '"]');
    if (countNode) countNode.textContent = String(visible);
  }

  document.querySelectorAll('[data-filter-scope]').forEach(function (toolbar) {
    const scope = toolbar.dataset.filterScope;
    ["input", "change"].forEach(function (eventName) {
      toolbar.querySelectorAll("input,select").forEach(function (node) {
        node.addEventListener(eventName, function () { apply(scope); });
      });
    });
    const reset = toolbar.querySelector('[data-filter-reset="' + scope + '"]');
    if (reset) {
      reset.addEventListener("click", function () {
        toolbar.querySelectorAll("input").forEach(function (node) { node.value = ""; });
        toolbar.querySelectorAll("select").forEach(function (node) { node.value = ""; });
        apply(scope);
      });
    }
    apply(scope);
  });
});
</script>"""


def render_image_cell(page_dir: Path, image_path: Any, alt_text: str, *, is_template: bool = False) -> str:
    existing = existing_path_str(image_path)
    if not existing:
        return "<td>-</td>"
    try:
        href = os.path.relpath(existing, page_dir)
    except ValueError:
        href = existing
    style = (
        "width:72px;height:72px;object-fit:contain;background:#fff;"
        if is_template
        else "height:72px;max-width:180px;object-fit:contain;background:#fff;"
    )
    return (
        f'<td><a href="{escape(href)}" target="_blank" rel="noopener noreferrer">'
        f'<img class="img-thumb" src="{escape(href)}" alt="{escape(alt_text)}" style="{style}" /></a></td>'
    )


def row_data_attrs(row: Dict[str, Any], scope: str) -> str:
    return (
        f'data-row-scope="{escape(scope)}" '
        f'data-app="{escape(str(row.get("app", "")))}" '
        f'data-case="{escape(str(row.get("case_id", "")))}" '
        f'data-status="{escape(str(row.get("status", "")))}" '
        f'data-history="{escape(str(row.get("history_label", "")))}" '
        f'data-error="{escape(clean_text(str(row.get("error_summary", ""))))}"'
    )


def row_css_class(row: Dict[str, Any]) -> str:
    if is_issue_status(str(row.get("status", ""))):
        return "issue-row"
    if str(row.get("history_label", "")) == "已修复":
        return "recovered-row"
    return ""


def app_summary_cards(rows: List[Dict[str, Any]]) -> str:
    by_app: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_app[str(row.get("app", ""))].append(row)
    cards = []
    for app, app_rows in sorted(by_app.items(), key=lambda item: (-sum(1 for row in item[1] if is_issue_status(str(row.get("status", "")))), item[0].lower())):
        counts = count_rows(app_rows)
        cards.append(
            '<div class="mini-card">'
            f"<h3>{escape(app)}</h3>"
            f'<p class="muted">{counts["total"]} 个用例，问题率 {escape(format_percent(counts["issues"], max(counts["total"], 1)))}</p>'
            '<div class="mini-kpis">'
            f'<span>问题 {counts["issues"]}</span>'
            f'<span>一致 {counts["match"]}</span>'
            f'<span>跳过 {counts["skipped"]}</span>'
            "</div></div>"
        )
    if not cards:
        return '<div class="empty">暂无应用统计。</div>'
    return '<div class="app-summary-scroll"><div class="app-grid">' + "".join(cards) + "</div></div>"


def render_error_summary_cell(row: Dict[str, Any]) -> str:
    summary = str(row.get("error_summary", "-") or "-")
    full_error_raw = str(row.get("error", "") or "")
    full_error = clean_text(full_error_raw or "-")
    summary_lines = "".join(
        f'<span class="error-summary-line">{escape(line)}</span>'
        for line in summary.splitlines()
        if line.strip()
    )
    if not summary_lines:
        summary_lines = '<span class="error-summary-line">-</span>'
    details_html = ""
    if full_error_raw.strip():
        details_html = (
            '<details class="error-details">'
            "<summary>展开完整错误</summary>"
            f'<pre class="error-full">{escape(full_error_raw)}</pre>'
            "</details>"
        )
    return f'<td class="error-cell" title="{escape(full_error or "-")}">{summary_lines}{details_html}</td>'


def issue_digest_table(rows: List[Dict[str, Any]], page_dir: Path, checker: str, scope: str) -> str:
    bad_rows = [row for row in sort_rows_for_display(rows) if is_issue_status(str(row.get("status", "")))]
    image_headers = "<th>Template</th><th>Target</th><th>Match Result</th>"
    if checker == "button_color":
        image_headers = "<th>原图 Before</th><th>原图 After</th><th>按钮 Before</th><th>按钮 After</th>"
    elif checker in PAIR_IMAGE_CHECKERS:
        image_headers = "<th>原图 Before</th><th>原图 After</th>"
    elif checker in NO_PREVIEW_IMAGE_CHECKERS:
        image_headers = ""

    issue_empty_colspan = 6 if checker in NO_PREVIEW_IMAGE_CHECKERS else 10
    parts = [
        '<div class="card"><div class="section-head">'
        '<div><h2>优先关注问题</h2><p class="muted">按严重性和历史标签排序，优先把新回归和持续失败排到前面。</p></div>'
        f'<div class="anchor-links"><a href="#all-details">跳到全量明细</a></div></div>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>App</th><th>Case ID</th><th>历史标签</th><th>状态</th><th>最近通过</th><th>错误摘要</th>"
        f"{image_headers}</tr></thead><tbody>"
    ]
    if not bad_rows:
        parts.append(f'<tr><td colspan="{issue_empty_colspan}">当前运行无不一致/异常</td></tr>')
    else:
        for row in bad_rows:
            cls = status_class(str(row.get("status", "")))
            attrs = row_data_attrs(row, scope)
            tr_class = row_css_class(row)
            parts.append(f'<tr class="{escape(tr_class)}" {attrs}>')
            parts.append(f"<td>{escape(str(row.get('app', '')))}</td>")
            parts.append(f"<td><code>{escape(str(row.get('case_id', '')))}</code></td>")
            parts.append(f'<td><span class="tag tag-info">{escape(str(row.get("history_label", "N/A")))}</span></td>')
            parts.append(f'<td><span class="tag {cls}">{escape(str(row.get("status", "")))}</span></td>')
            parts.append(f"<td>{escape(readable_timestamp(str(row.get('last_good_at', '-'))))}</td>")
            parts.append(render_error_summary_cell(row))
            if checker == "button_color":
                parts.append(render_image_cell(page_dir, row.get("preview_template_image", ""), f"{row.get('case_id', '')} before"))
                parts.append(render_image_cell(page_dir, row.get("preview_target_image", ""), f"{row.get('case_id', '')} after"))
                parts.append(render_image_cell(page_dir, row.get("preview_button_before_image", ""), f"{row.get('case_id', '')} button before", is_template=True))
                parts.append(render_image_cell(page_dir, row.get("preview_button_after_image", ""), f"{row.get('case_id', '')} button after", is_template=True))
            elif checker in PAIR_IMAGE_CHECKERS:
                parts.append(render_image_cell(page_dir, row.get("preview_template_image", ""), f"{row.get('case_id', '')} before"))
                parts.append(render_image_cell(page_dir, row.get("preview_target_image", ""), f"{row.get('case_id', '')} after"))
            elif checker not in NO_PREVIEW_IMAGE_CHECKERS:
                parts.append(render_image_cell(page_dir, row.get("preview_template_image", ""), f"{row.get('case_id', '')} template", is_template=True))
                parts.append(render_image_cell(page_dir, row.get("preview_target_image", ""), f"{row.get('case_id', '')} target"))
                parts.append(render_image_cell(page_dir, row.get("preview_match_result_image", ""), f"{row.get('case_id', '')} match result"))
            parts.append("</tr>")
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def render_detail_tables(rows: List[Dict[str, Any]], page_dir: Path, checker: str, scope: str) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("app", ""))].append(row)

    image_headers = "<th>Template</th><th>Target</th><th>Match Result</th>"
    no_data_colspan = 15
    if checker == "button_color":
        image_headers = "<th>原图 Before</th><th>原图 After</th><th>按钮 Before</th><th>按钮 After</th>"
        no_data_colspan = 16
    elif checker in PAIR_IMAGE_CHECKERS:
        image_headers = "<th>原图 Before</th><th>原图 After</th>"
        no_data_colspan = 14
    elif checker in NO_PREVIEW_IMAGE_CHECKERS:
        image_headers = ""
        no_data_colspan = 11

    parts = ['<div class="card" id="all-details"><div class="section-head"><div><h2>全量明细</h2><p class="muted">问题会排在每个应用的前面，便于边筛选边比对截图。</p></div></div></div>']
    for app in sorted(grouped.keys(), key=lambda name: (-sum(1 for row in grouped[name] if is_issue_status(str(row.get("status", "")))), name.lower())):
        app_rows = sort_rows_for_display(grouped[app])
        parts.append(
            f'<div class="card" data-app-card="{escape(scope)}"><div class="section-head">'
            f'<div><h2>{escape(app)} ({len(app_rows)} 个用例)</h2>'
            f'<p class="muted">问题 {sum(1 for row in app_rows if is_issue_status(str(row.get("status", ""))))}，一致 {sum(1 for row in app_rows if str(row.get("status", "")) == GOOD_STATUS)}</p></div>'
            "</div>"
        )
        parts.append(
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Case ID</th><th>历史标签</th><th>预期</th><th>实际</th><th>状态</th><th>相似度</th>"
            "<th>阈值</th><th>预期框</th><th>实际框</th><th>错误摘要</th><th>最近通过</th>"
            f"{image_headers}</tr></thead><tbody>"
        )
        if not app_rows:
            parts.append(f'<tr><td colspan="{no_data_colspan}">暂无数据</td></tr>')
        else:
            for row in app_rows:
                cls = status_class(str(row.get("status", "")))
                tr_class = row_css_class(row)
                attrs = row_data_attrs(row, scope)
                parts.append(f'<tr class="{escape(tr_class)}" {attrs}>')
                parts.append(f"<td><code>{escape(str(row.get('case_id', '')))}</code></td>")
                parts.append(f'<td><span class="tag tag-info">{escape(str(row.get("history_label", "N/A")))}</span></td>')
                parts.append(f"<td>{escape(str(row.get('expected_passed', '')))}</td>")
                parts.append(f"<td>{escape(str(row.get('actual_passed', '')))}</td>")
                parts.append(f'<td><span class="tag {cls}">{escape(str(row.get("status", "")))}</span></td>')
                parts.append(f"<td>{escape(str(row.get('similarity', '') or '-'))}</td>")
                parts.append(f"<td>{escape(str(row.get('threshold', '') or '-'))}</td>")
                parts.append(f"<td><code>{escape(str(row.get('expected_bounds', '') or '-'))}</code></td>")
                parts.append(f"<td><code>{escape(str(row.get('actual_bounds', '') or '-'))}</code></td>")
                parts.append(render_error_summary_cell(row))
                parts.append(f"<td>{escape(readable_timestamp(str(row.get('last_good_at', '-'))))}</td>")
                if checker == "button_color":
                    parts.append(render_image_cell(page_dir, row.get("preview_template_image", ""), f"{row.get('case_id', '')} before"))
                    parts.append(render_image_cell(page_dir, row.get("preview_target_image", ""), f"{row.get('case_id', '')} after"))
                    parts.append(render_image_cell(page_dir, row.get("preview_button_before_image", ""), f"{row.get('case_id', '')} button before", is_template=True))
                    parts.append(render_image_cell(page_dir, row.get("preview_button_after_image", ""), f"{row.get('case_id', '')} button after", is_template=True))
                elif checker in PAIR_IMAGE_CHECKERS:
                    parts.append(render_image_cell(page_dir, row.get("preview_template_image", ""), f"{row.get('case_id', '')} before"))
                    parts.append(render_image_cell(page_dir, row.get("preview_target_image", ""), f"{row.get('case_id', '')} after"))
                elif checker not in NO_PREVIEW_IMAGE_CHECKERS:
                    parts.append(render_image_cell(page_dir, row.get("preview_template_image", ""), f"{row.get('case_id', '')} template", is_template=True))
                    parts.append(render_image_cell(page_dir, row.get("preview_target_image", ""), f"{row.get('case_id', '')} target"))
                    parts.append(render_image_cell(page_dir, row.get("preview_match_result_image", ""), f"{row.get('case_id', '')} match result"))
                parts.append("</tr>")
        parts.append("</tbody></table></div></div>")
    return "".join(parts)


def read_history_rows(checker: str) -> List[Dict[str, str]]:
    paths = history_paths_for_checker(checker)
    split_rows = _read_csv_rows_if_exists(paths["results_csv"])
    legacy_rows = [row for row in _read_csv_rows_if_exists(LEGACY_RESULTS_CSV) if infer_checker(row) == checker]
    return split_rows + legacy_rows


def load_latest_run_rows_from_history(checker: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    paths = history_paths_for_checker(checker)
    run_rows = _read_csv_rows_if_exists(paths["runs_csv"])
    all_results = _read_csv_rows_if_exists(paths["results_csv"])

    run_rows.extend([row for row in _read_csv_rows_if_exists(LEGACY_RUNS_CSV) if _run_row_matches_checker(row, checker)])
    all_results.extend([row for row in _read_csv_rows_if_exists(LEGACY_RESULTS_CSV) if infer_checker(row) == checker])

    if not run_rows or not all_results:
        return [], {}

    by_run: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in all_results:
        if infer_checker(row) != checker:
            continue
        run_id = row.get("run_id", "")
        if run_id:
            by_run[run_id].append(row)

    candidate_runs = [row for row in run_rows if row.get("run_id") in by_run]
    if not candidate_runs:
        return [], {}

    pytest_candidates = [row for row in candidate_runs if "pytest" in str(row.get("note", "")).lower()]
    latest_run = pytest_candidates[-1] if pytest_candidates else candidate_runs[-1]
    result_rows = by_run.get(latest_run.get("run_id", ""), [])
    run_meta = {
        "run_id": latest_run.get("run_id", "unknown"),
        "run_at": latest_run.get("run_at", "unknown"),
        "commit": latest_run.get("commit", "unknown"),
        "branch": latest_run.get("branch", "unknown"),
        "note": latest_run.get("note", ""),
    }
    return result_rows, run_meta


def write_latest_snapshot(rows: List[Dict[str, Any]], run_meta: Dict[str, str], latest_html: Path, checker: str) -> None:
    counts = count_rows(rows)
    latest_html.parent.mkdir(parents=True, exist_ok=True)
    scope = "snapshot"
    new_regressions = sum(1 for row in rows if str(row.get("history_label", "")) == "新回归")
    recovered = sum(1 for row in rows if str(row.get("history_label", "")) == "已修复")
    flaky_rows = sum(1 for row in rows if str(row.get("recent_flip_count", "")).strip())

    body_parts = [
        '<div class="card hero"><div class="section-head">'
        f'<div><h1>最新测试快照 ({escape(checker)})</h1><p class="muted">先看本次是否出现新回归，再快速下钻到具体应用和截图。</p></div>'
        '<div class="anchor-links"><a href="#all-details">全量明细</a><a href="FAILURE_VIEW.html">问题视图</a><a href="HISTORY_TIMELINE.html">历史时间线</a></div>'
        "</div>"
        '<ul class="meta-list">'
        f"<li><strong>运行 ID</strong><br /><code>{escape(run_meta['run_id'])}</code></li>"
        f"<li><strong>运行时间</strong><br />{escape(readable_timestamp(run_meta['run_at']))}</li>"
        f"<li><strong>Git Branch</strong><br /><code>{escape(run_meta['branch'])}</code></li>"
        f"<li><strong>Git Commit</strong><br /><code>{escape(run_meta['commit'])}</code></li>"
        f"<li><strong>备注</strong><br />{escape(run_meta['note'] or 'N/A')}</li>"
        "</ul></div>"
    ]

    body_parts.append(
        render_metric_grid(
            [
                {"label": "总用例数", "value": str(counts["total"]), "sub": f"覆盖 {len({str(row.get('app', '')) for row in rows})} 个应用"},
                {"label": "一致率", "value": format_percent(counts["match"], max(counts["total"], 1)), "sub": f"一致 {counts['match']} / 总计 {counts['total']}", "variant": "ok"},
                {"label": "问题数", "value": str(counts["issues"]), "sub": f"不一致 {counts['mismatch']}，异常 {counts['error']}", "variant": "bad"},
                {"label": "新回归", "value": str(new_regressions), "sub": "上次通过，这次失败/异常", "variant": "bad"},
                {"label": "已修复", "value": str(recovered), "sub": "上次失败/异常，这次恢复一致", "variant": "ok"},
                {"label": "最近波动", "value": str(flaky_rows), "sub": "最近 5 次运行中状态发生切换", "variant": "warn"},
            ]
        )
    )

    body_parts.append(
        '<div class="split">'
        f'<div class="stack">{issue_digest_table(rows, latest_html.parent, checker, scope)}</div>'
        f'<div class="stack"><div class="card"><h2>按应用概览</h2><p class="muted">优先把问题多的应用放在前面。</p>{app_summary_cards(rows)}</div></div>'
        "</div>"
    )
    body_parts.append(render_filter_controls(scope, rows, "搜索 app / case / 状态 / 错误摘要"))
    body_parts.append(render_detail_tables(rows, latest_html.parent, checker, scope))
    latest_html.write_text(html_page(f"最新测试快照({checker})", "".join(body_parts), build_filter_script()), encoding="utf-8")


def write_history_timeline(checker: str, timeline_html: Path) -> None:
    rows = read_history_rows(checker)
    timeline_html.parent.mkdir(parents=True, exist_ok=True)
    history_stats = build_case_history_stats(rows)
    runs = sorted({(str(row.get("run_id", "")), str(row.get("run_at", ""))) for row in rows})
    latest_run_id = runs[-1][0] if runs else ""
    latest_run_at = runs[-1][1] if runs else ""
    latest_rows = [row for row in rows if str(row.get("run_id", "")) == latest_run_id]
    latest_counts = count_rows(latest_rows)

    persistent_failures = [
        stat for stat in history_stats.values() if is_issue_status(str(stat.get("latest_status", ""))) and int(stat.get("failure_streak", 0)) >= 2
    ]
    unstable_cases = [stat for stat in history_stats.values() if int(stat.get("recent_flip_count", 0)) > 0]
    repaired_cases = [
        stat for stat in history_stats.values() if str(stat.get("latest_status", "")) == GOOD_STATUS and str(stat.get("last_failed_at", ""))
    ]
    repaired_cases = sorted(repaired_cases, key=lambda stat: str(stat.get("latest_run_at", "")), reverse=True)[:20]

    body_parts = [
        '<div class="card hero"><div class="section-head">'
        f'<div><h1>测试历史时间线 ({escape(checker)})</h1><p class="muted">看清哪些问题是新回归，哪些是长期未修，以及哪些用例本身不稳定。</p></div>'
        '<div class="anchor-links"><a href="#persistent-failures">持续失败</a><a href="#repair-tracking">修复追踪</a><a href="#recent-details">最近明细</a></div>'
        "</div></div>"
    ]
    body_parts.append(
        render_metric_grid(
            [
                {"label": "历史运行次数", "value": str(len(runs)), "sub": f"最近一次: {readable_timestamp(latest_run_at)}"},
                {"label": "累计用例数", "value": str(len(history_stats)), "sub": f"当前最新运行 ID: {latest_run_id or '-'}"},
                {"label": "最新运行问题数", "value": str(latest_counts["issues"]), "sub": f"不一致 {latest_counts['mismatch']}，异常 {latest_counts['error']}", "variant": "bad"},
                {"label": "持续失败用例", "value": str(len(persistent_failures)), "sub": "连续至少 2 次失败/异常", "variant": "warn"},
                {"label": "波动用例", "value": str(len(unstable_cases)), "sub": "最近 5 次运行有状态切换", "variant": "warn"},
                {"label": "已修复用例", "value": str(len(repaired_cases)), "sub": "历史出现过问题，当前已恢复一致", "variant": "ok"},
            ]
        )
    )

    body_parts.append(f'<div class="card" id="persistent-failures"><h2>持续失败用例</h2><p class="muted">这类问题通常优先级最高，因为它们不是偶发波动。</p>')
    body_parts.append('<div class="table-wrap"><table><thead><tr><th>App</th><th>Case ID</th><th>连续失败次数</th><th>最近通过时间</th><th>最新状态</th><th>最新运行 ID</th></tr></thead><tbody>')
    if not persistent_failures:
        body_parts.append('<tr><td colspan="6">当前没有连续失败 2 次及以上的用例。</td></tr>')
    else:
        for stat in sorted(persistent_failures, key=lambda item: (-int(item.get("failure_streak", 0)), str(item.get("app", "")).lower(), str(item.get("case_id", "")).lower())):
            cls = status_class(str(stat.get("latest_status", "")))
            body_parts.append(
                "<tr>"
                f"<td>{escape(str(stat.get('app', '')))}</td>"
                f"<td><code>{escape(str(stat.get('case_id', '')))}</code></td>"
                f"<td>{escape(str(stat.get('failure_streak', '0')))}</td>"
                f"<td>{escape(readable_timestamp(str(stat.get('last_passed_at', ''))))}</td>"
                f'<td><span class="tag {cls}">{escape(str(stat.get("latest_status", "")))}</span></td>'
                f"<td><code>{escape(str(stat.get('latest_run_id', '')))}</code></td>"
                "</tr>"
            )
    body_parts.append("</tbody></table></div></div>")

    body_parts.append('<div class="card"><h2>最近波动用例</h2><p class="muted">如果一个用例频繁在通过和失败之间切换，通常意味着阈值、样本或环境稳定性需要复查。</p>')
    body_parts.append('<div class="table-wrap"><table><thead><tr><th>App</th><th>Case ID</th><th>最近 5 次切换次数</th><th>最近失败时间</th><th>最近通过时间</th><th>最新状态</th></tr></thead><tbody>')
    if not unstable_cases:
        body_parts.append('<tr><td colspan="6">当前没有检测到最近 5 次内发生状态切换的用例。</td></tr>')
    else:
        for stat in sorted(unstable_cases, key=lambda item: (-int(item.get("recent_flip_count", 0)), str(item.get("app", "")).lower(), str(item.get("case_id", "")).lower()))[:50]:
            cls = status_class(str(stat.get("latest_status", "")))
            body_parts.append(
                "<tr>"
                f"<td>{escape(str(stat.get('app', '')))}</td>"
                f"<td><code>{escape(str(stat.get('case_id', '')))}</code></td>"
                f"<td>{escape(str(stat.get('recent_flip_count', '0')))}</td>"
                f"<td>{escape(readable_timestamp(str(stat.get('last_failed_at', ''))))}</td>"
                f"<td>{escape(readable_timestamp(str(stat.get('last_passed_at', ''))))}</td>"
                f'<td><span class="tag {cls}">{escape(str(stat.get("latest_status", "")))}</span></td>'
                "</tr>"
            )
    body_parts.append("</tbody></table></div></div>")

    body_parts.append(f'<div class="card" id="repair-tracking"><h2>用例修复追踪</h2><p class="muted">关注首次失败、首次修复以及当前状态，适合回看修复过程。</p>')
    body_parts.append('<div class="table-wrap"><table><thead><tr><th>App</th><th>Case ID</th><th>首次失败时间</th><th>首次修复通过时间</th><th>总运行次数</th><th>最新状态</th><th>最新运行 ID</th></tr></thead><tbody>')
    for key in sorted(history_stats.keys(), key=lambda item: (item[0].lower(), item[1].lower())):
        stat = history_stats[key]
        cls = status_class(str(stat.get("latest_status", "")))
        body_parts.append(
            "<tr>"
            f"<td>{escape(str(stat.get('app', '')))}</td>"
            f"<td><code>{escape(str(stat.get('case_id', '')))}</code></td>"
            f"<td>{escape(readable_timestamp(str(stat.get('first_failed_at', ''))))}</td>"
            f"<td>{escape(readable_timestamp(str(stat.get('first_fixed_at', ''))))}</td>"
            f"<td>{escape(str(stat.get('total_runs', '0')))}</td>"
            f'<td><span class="tag {cls}">{escape(str(stat.get("latest_status", "")))}</span></td>'
            f"<td><code>{escape(str(stat.get('latest_run_id', '')))}</code></td>"
            "</tr>"
        )
    body_parts.append("</tbody></table></div></div>")

    body_parts.append(f'<div class="card" id="recent-details"><h2>最近运行明细（最近 200 条）</h2><p class="muted">这是按时间倒序的轻量视图，适合快速回看最近几次跑测。</p>')
    body_parts.append('<div class="table-wrap"><table><thead><tr><th>运行时间</th><th>运行 ID</th><th>App</th><th>Case ID</th><th>预期</th><th>实际</th><th>状态</th></tr></thead><tbody>')
    for row in sorted(rows, key=lambda item: (str(item.get("run_at", "")), str(item.get("run_id", ""))), reverse=True)[:200]:
        cls = status_class(str(row.get("status", "")))
        body_parts.append(
            "<tr>"
            f"<td>{escape(readable_timestamp(str(row.get('run_at', ''))))}</td>"
            f"<td><code>{escape(str(row.get('run_id', '')))}</code></td>"
            f"<td>{escape(str(row.get('app', '')))}</td>"
            f"<td><code>{escape(str(row.get('case_id', '')))}</code></td>"
            f"<td>{escape(str(row.get('expected_passed', '')))}</td>"
            f"<td>{escape(str(row.get('actual_passed', '')))}</td>"
            f'<td><span class="tag {cls}">{escape(str(row.get("status", "")))}</span></td>'
            "</tr>"
        )
    body_parts.append("</tbody></table></div></div>")
    timeline_html.write_text(html_page(f"测试历史时间线({checker})", "".join(body_parts)), encoding="utf-8")


def write_failure_view(rows: List[Dict[str, Any]], run_meta: Dict[str, str], failure_html: Path, checker: str) -> None:
    scope = "failures"
    bad_rows = [row for row in sort_rows_for_display(rows) if is_issue_status(str(row.get("status", "")))]
    counts = count_rows(rows)
    new_regressions = sum(1 for row in bad_rows if str(row.get("history_label", "")) == "新回归")
    persistent = sum(1 for row in bad_rows if str(row.get("history_label", "")) == "持续失败")

    image_headers = "<th>Template</th><th>Target</th><th>Match Result</th>"
    if checker == "button_color":
        image_headers = "<th>原图 Before</th><th>原图 After</th><th>按钮 Before</th><th>按钮 After</th>"
    elif checker in PAIR_IMAGE_CHECKERS:
        image_headers = "<th>原图 Before</th><th>原图 After</th>"
    elif checker in NO_PREVIEW_IMAGE_CHECKERS:
        image_headers = ""

    body_parts = [
        '<div class="card hero"><div class="section-head">'
        f'<div><h1>不一致与异常视图 ({escape(checker)})</h1><p class="muted">只保留需要处理的问题项，适合回归后第一时间排障。</p></div>'
        '<div class="anchor-links"><a href="LATEST_SNAPSHOT.html">返回快照</a><a href="HISTORY_TIMELINE.html">查看历史</a></div>'
        "</div>"
        '<ul class="meta-list">'
        f"<li><strong>运行 ID</strong><br /><code>{escape(run_meta['run_id'])}</code></li>"
        f"<li><strong>运行时间</strong><br />{escape(readable_timestamp(run_meta['run_at']))}</li>"
        f"<li><strong>问题数</strong><br />{len(bad_rows)} / {counts['total']}</li>"
        f"<li><strong>新回归</strong><br />{new_regressions}</li>"
        f"<li><strong>持续失败</strong><br />{persistent}</li>"
        "</ul></div>"
    ]
    body_parts.append(render_filter_controls(scope, bad_rows, "搜索 app / case / 历史标签 / 错误摘要"))

    no_data_colspan = 15
    if checker == "button_color":
        no_data_colspan = 16
    elif checker in PAIR_IMAGE_CHECKERS:
        no_data_colspan = 14
    elif checker in NO_PREVIEW_IMAGE_CHECKERS:
        no_data_colspan = 10

    body_parts.append('<div class="card"><div class="section-head"><div><h2>问题明细</h2><p class="muted">默认已按新回归、持续失败、应用名称排序。</p></div></div>')
    body_parts.append('<div class="table-wrap"><table><thead><tr>'
                      "<th>App</th><th>Case ID</th><th>历史标签</th><th>预期</th><th>实际</th><th>状态</th><th>相似度</th><th>阈值</th><th>最近通过</th><th>错误摘要</th>"
                      f"{image_headers}</tr></thead><tbody>")
    if not bad_rows:
        body_parts.append(f'<tr><td colspan="{no_data_colspan}">当前运行无不一致/异常</td></tr>')
    else:
        for row in bad_rows:
            cls = status_class(str(row.get("status", "")))
            attrs = row_data_attrs(row, scope)
            tr_class = row_css_class(row)
            body_parts.append(f'<tr class="{escape(tr_class)}" {attrs}>')
            body_parts.append(f"<td>{escape(str(row.get('app', '')))}</td>")
            body_parts.append(f"<td><code>{escape(str(row.get('case_id', '')))}</code></td>")
            body_parts.append(f'<td><span class="tag tag-info">{escape(str(row.get("history_label", "N/A")))}</span></td>')
            body_parts.append(f"<td>{escape(str(row.get('expected_passed', '')))}</td>")
            body_parts.append(f"<td>{escape(str(row.get('actual_passed', '')))}</td>")
            body_parts.append(f'<td><span class="tag {cls}">{escape(str(row.get("status", "")))}</span></td>')
            body_parts.append(f"<td>{escape(str(row.get('similarity', '') or '-'))}</td>")
            body_parts.append(f"<td>{escape(str(row.get('threshold', '') or '-'))}</td>")
            body_parts.append(f"<td>{escape(readable_timestamp(str(row.get('last_good_at', '-'))))}</td>")
            body_parts.append(render_error_summary_cell(row))
            if checker == "button_color":
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_template_image", ""), f"{row.get('case_id', '')} before"))
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_target_image", ""), f"{row.get('case_id', '')} after"))
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_button_before_image", ""), f"{row.get('case_id', '')} button before", is_template=True))
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_button_after_image", ""), f"{row.get('case_id', '')} button after", is_template=True))
            elif checker in PAIR_IMAGE_CHECKERS:
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_template_image", ""), f"{row.get('case_id', '')} before"))
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_target_image", ""), f"{row.get('case_id', '')} after"))
            elif checker not in NO_PREVIEW_IMAGE_CHECKERS:
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_template_image", ""), f"{row.get('case_id', '')} template", is_template=True))
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_target_image", ""), f"{row.get('case_id', '')} target"))
                body_parts.append(render_image_cell(failure_html.parent, row.get("preview_match_result_image", ""), f"{row.get('case_id', '')} match result"))
            body_parts.append("</tr>")
    body_parts.append("</tbody></table></div></div>")
    failure_html.write_text(html_page(f"不一致与异常视图({checker})", "".join(body_parts), build_filter_script()), encoding="utf-8")


def run(note: str, image_output_root: Path, source: str, checker_name: str) -> None:
    paths = report_paths_for_checker(checker_name)
    history_paths = history_paths_for_checker(checker_name)
    history_rows = read_history_rows(checker_name)

    if source == "pytest":
        run_rows, run_meta = load_latest_run_rows_from_history(checker_name)
        if not run_rows:
            raise RuntimeError(f"未找到 {checker_name} 的 pytest 结果。请先执行 pytest，再运行 --source pytest")
        if note:
            run_meta["note"] = note
        repair_preview_images(run_rows, image_output_root)
        enrich_rows_with_history(run_rows, history_rows, run_meta["run_id"])
        write_latest_snapshot(run_rows, run_meta, paths["latest"], checker_name)
        write_history_timeline(checker_name, paths["timeline"])
        write_failure_view(run_rows, run_meta, paths["failure"], checker_name)
        print(f"已生成: {paths['latest']}")
        print(f"已生成: {paths['timeline']}")
        print(f"已生成: {paths['failure']}")
        print(f"来源: pytest 历史结果（{checker_name}，未重新执行 checker）")
        return

    if checker_name != "image_match":
        raise RuntimeError("当前 --source checker 仅支持 image_match。button_color/count_change 请使用 --source pytest")

    cases = discover_cases()
    check_image_match, checker_error = try_import_checker()
    git_info = get_git_info()
    run_at = now_iso()
    run_id = run_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    run_meta = {
        "run_id": run_id,
        "run_at": run_at,
        "commit": git_info["commit"],
        "branch": git_info["branch"],
        "note": note,
    }
    run_rows: List[Dict[str, Any]] = []
    for case_file in cases:
        payload = load_case(case_file)
        case_row = evaluate_case(case_file, payload, check_image_match, checker_error)
        if str(payload.get("type", "")).lower() == "image_match":
            case_row.update(resolve_output_images(image_output_root, case_row["app"], case_row["case_id"]))
        else:
            case_row["preview_template_image"] = ""
            case_row["preview_target_image"] = ""
            case_row["preview_match_result_image"] = ""
        case_row["run_id"] = run_id
        case_row["run_at"] = run_at
        case_row["git_commit"] = git_info["commit"]
        case_row["checker"] = checker_name
        run_rows.append(case_row)

    repair_preview_images(run_rows, image_output_root)
    enrich_rows_with_history(run_rows, history_rows, run_id)

    append_csv_rows(
        history_paths["runs_csv"],
        ["run_id", "run_at", "branch", "commit", "note", "checker", "case_count"],
        [{**run_meta, "checker": checker_name, "case_count": len(run_rows)}],
    )
    append_csv_rows(
        history_paths["results_csv"],
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
        run_rows,
    )
    write_latest_snapshot(run_rows, run_meta, paths["latest"], checker_name)
    write_history_timeline(checker_name, paths["timeline"])
    write_failure_view(run_rows, run_meta, paths["failure"], checker_name)
    print(f"已生成: {paths['latest']}")
    print(f"已生成: {paths['timeline']}")
    print(f"已生成: {paths['failure']}")
    print(f"已更新: {history_paths['runs_csv']}")
    print(f"已更新: {history_paths['results_csv']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最新测试快照和历史时间线报告。")
    parser.add_argument("--note", default="", help="本次运行的备注。")
    parser.add_argument("--image-output-root", default=str(DEFAULT_IMAGE_OUTPUT_ROOT), help="image_match 调试输出目录。")
    parser.add_argument("--source", choices=["checker", "pytest"], default="checker", help="报告数据来源。")
    parser.add_argument("--checker", choices=list(SUPPORTED_CHECKERS), default="image_match", help="要生成报告的 checker。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.note, Path(args.image_output_root), args.source, args.checker)
