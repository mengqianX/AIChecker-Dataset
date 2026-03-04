import argparse
import csv
import glob
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_CASE_DIR = REPO_ROOT / "testcase/image_match/jsons"
REPORT_DIR = REPO_ROOT / "reports"
HISTORY_DIR = REPORT_DIR / "history"
LEGACY_RUNS_CSV = HISTORY_DIR / "test_runs.csv"
LEGACY_RESULTS_CSV = HISTORY_DIR / "test_case_results.csv"
DEFAULT_IMAGE_OUTPUT_ROOT = REPO_ROOT / "AIChecker/tests/image_match_output"
SUPPORTED_CHECKERS = ("image_match", "button_color", "count_change")


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
            row["status"] = "一致(MATCH)" if (actual_passed == expected_passed) else "不一致(MISMATCH)"
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


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{escape(title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;margin:24px;line-height:1.45;color:#1f2937;background:#f8fafc}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin-bottom:16px}}
h1,h2{{margin:0 0 12px}} h1{{font-size:24px}} h2{{font-size:18px;margin-top:16px}}
ul{{margin:0;padding-left:20px}} .muted{{color:#6b7280}}
.table-wrap{{overflow-x:auto;border:1px solid #e5e7eb;border-radius:8px;background:#fff}}
table{{border-collapse:collapse;width:100%;min-width:900px}}
th,td{{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;font-size:13px;white-space:nowrap;vertical-align:top}}
thead th{{position:sticky;top:0;background:#f3f4f6;z-index:1}}
.tag-ok{{color:#166534;font-weight:600}} .tag-bad{{color:#b91c1c;font-weight:600}} .tag-unknown{{color:#92400e;font-weight:600}}
code{{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12px}}
.error-cell{{white-space:normal!important;overflow-wrap:anywhere;word-break:break-word;width:clamp(420px,46vw,980px);max-width:clamp(420px,46vw,980px);min-width:420px;line-height:1.4;max-height:7em;overflow-y:auto;overflow-x:hidden;display:block}}
</style></head><body>{body}</body></html>"""


def status_class(status: str) -> str:
    if "一致(MATCH)" in status:
        return "tag-ok"
    if "不一致(MISMATCH)" in status or "异常(ERROR)" in status:
        return "tag-bad"
    return "tag-unknown"


def resolve_output_images(image_output_root: Path, app: str, case_id: str) -> Dict[str, str]:
    candidates = [image_output_root / app / case_id, image_output_root / app, image_output_root / case_id]
    image_names = {"preview_template_image": "template.png", "preview_target_image": "target.png", "preview_match_result_image": "match_result.png"}
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


def write_latest_snapshot(rows: List[Dict[str, Any]], run_meta: Dict[str, str], latest_html: Path, checker: str) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["app"]].append(r)
    match_count = sum(1 for r in rows if r["status"] == "一致(MATCH)")
    mismatch_count = sum(1 for r in rows if r["status"] == "不一致(MISMATCH)")
    error_count = sum(1 for r in rows if r["status"] == "异常(ERROR)")
    skipped_count = sum(1 for r in rows if r["actual_passed"] == "跳过(SKIPPED)")

    html_parts: List[str] = []
    latest_html.parent.mkdir(parents=True, exist_ok=True)
    html_parts.append(f'<div class="card"><h1>最新测试快照 ({escape(checker)})</h1><ul>')
    html_parts.append(f"<li><strong>运行ID (Run ID)</strong>: <code>{escape(run_meta['run_id'])}</code></li>")
    html_parts.append(f"<li><strong>运行时间 (UTC)</strong>: <code>{escape(run_meta['run_at'])}</code></li>")
    html_parts.append(f"<li><strong>Git Branch</strong>: <code>{escape(run_meta['branch'])}</code></li>")
    html_parts.append(f"<li><strong>Git Commit</strong>: <code>{escape(run_meta['commit'])}</code></li>")
    html_parts.append(f"<li><strong>备注 (Note)</strong>: {escape(run_meta['note'] or 'N/A')}</li></ul></div>")

    html_parts.append('<div class="card"><h2>运行结果统计</h2><ul>')
    html_parts.append(f"<li><strong>总用例数</strong>: {len(rows)}</li>")
    html_parts.append(f"<li><strong>一致 (MATCH)</strong>: {match_count}</li>")
    html_parts.append(f"<li><strong>不一致 (MISMATCH)</strong>: {mismatch_count}</li>")
    html_parts.append(f"<li><strong>异常 (ERROR)</strong>: {error_count}</li>")
    html_parts.append(f"<li><strong>跳过 (SKIPPED)</strong>: {skipped_count}</li></ul></div>")

    for app in sorted(grouped.keys()):
        app_rows = sorted(grouped[app], key=lambda x: x["case_id"])
        image_headers = "<th>Template</th><th>Target</th><th>Match Result</th>"
        if checker == "button_color":
            image_headers = "<th>原图 Before</th><th>原图 After</th><th>按钮 Before</th><th>按钮 After</th>"
        html_parts.append(f'<div class="card"><h2>{escape(app.capitalize())} ({len(app_rows)} 个用例)</h2>')
        html_parts.append('<div class="table-wrap"><table><thead><tr>'
                          f"<th>Case ID</th><th>预期</th><th>实际</th><th>状态</th><th>相似度</th><th>预期框</th><th>实际框</th>{image_headers}"
                          "</tr></thead><tbody>")
        for row in app_rows:
            cls = status_class(row["status"])

            def render_image_cell(abs_path: str, alt_text: str, *, is_template: bool = False) -> str:
                if not abs_path:
                    return "<td>-</td>"
                rel = os.path.relpath(abs_path, latest_html.parent)
                href = escape(rel)
                alt = escape(alt_text)
                image_style = "width:72px; height:72px; object-fit:contain; background:#fff;" if is_template else "height:72px; max-width:180px; object-fit:contain; background:#fff;"
                return f'<td><a href="{href}" target="_blank" rel="noopener noreferrer"><img src="{href}" alt="{alt}" style="{image_style} border:1px solid #e5e7eb; border-radius:6px;" /></a></td>'

            html_parts.append("<tr>")
            html_parts.append(f"<td><code>{escape(row['case_id'])}</code></td><td>{escape(row['expected_passed'])}</td><td>{escape(row['actual_passed'])}</td>")
            html_parts.append(f'<td><span class="{cls}">{escape(row["status"])}</span></td><td>{escape(row["similarity"] or "-")}</td>')
            html_parts.append(f"<td><code>{escape(row['expected_bounds'])}</code></td><td><code>{escape(row['actual_bounds'] or '-')}</code></td>")
            if checker == "button_color":
                html_parts.append(render_image_cell(row.get("preview_template_image", ""), f"{row['case_id']} before"))
                html_parts.append(render_image_cell(row.get("preview_target_image", ""), f"{row['case_id']} after"))
                html_parts.append(render_image_cell(row.get("preview_button_before_image", ""), f"{row['case_id']} button before", is_template=True))
                html_parts.append(render_image_cell(row.get("preview_button_after_image", ""), f"{row['case_id']} button after", is_template=True))
            else:
                html_parts.append(render_image_cell(row.get("preview_template_image", ""), f"{row['case_id']} template", is_template=True))
                html_parts.append(render_image_cell(row.get("preview_target_image", ""), f"{row['case_id']} target"))
                html_parts.append(render_image_cell(row.get("preview_match_result_image", ""), f"{row['case_id']} match result"))
            html_parts.append("</tr>")
        html_parts.append("</tbody></table></div></div>")
    latest_html.write_text(html_page(f"最新测试快照({checker})", "".join(html_parts)), encoding="utf-8")


def read_history_rows(checker: str) -> List[Dict[str, str]]:
    paths = history_paths_for_checker(checker)
    split_rows = _read_csv_rows_if_exists(paths["results_csv"])
    legacy_rows = [r for r in _read_csv_rows_if_exists(LEGACY_RESULTS_CSV) if infer_checker(r) == checker]
    return split_rows + legacy_rows


def load_latest_run_rows_from_history(checker: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    paths = history_paths_for_checker(checker)
    run_rows = _read_csv_rows_if_exists(paths["runs_csv"])
    if not run_rows:
        run_rows = []
    all_results = _read_csv_rows_if_exists(paths["results_csv"])

    # Backward compatibility: legacy shared CSV files.
    run_rows.extend([r for r in _read_csv_rows_if_exists(LEGACY_RUNS_CSV) if _run_row_matches_checker(r, checker)])
    all_results.extend([r for r in _read_csv_rows_if_exists(LEGACY_RESULTS_CSV) if infer_checker(r) == checker])

    if not run_rows or not all_results:
        return [], {}
    by_run: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in all_results:
        if infer_checker(r) != checker:
            continue
        rid = r.get("run_id", "")
        if rid:
            by_run[rid].append(r)
    candidate_runs = [r for r in run_rows if r.get("run_id") in by_run]
    if not candidate_runs:
        return [], {}
    pytest_candidates = [r for r in candidate_runs if "pytest" in str(r.get("note", "")).lower()]
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


def write_history_timeline(checker: str, timeline_html: Path) -> None:
    rows = read_history_rows(checker)
    by_case: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    html_parts: List[str] = []
    timeline_html.parent.mkdir(parents=True, exist_ok=True)
    html_parts.append(f'<div class="card"><h1>测试历史时间线 ({escape(checker)})</h1><p class="muted">按每次运行跟踪每个用例结果，并展示失败到通过的变化。</p></div>')
    html_parts.append('<div class="card"><h2>用例修复追踪</h2><div class="table-wrap"><table><thead><tr>'
                      "<th>Case ID</th><th>首次失败时间</th><th>首次修复通过时间</th><th>最新状态</th><th>最新运行ID</th>"
                      "</tr></thead><tbody>")
    for case_id in sorted(by_case.keys()):
        case_rows = sorted(by_case[case_id], key=lambda x: x["run_at"])
        first_failed_at = ""
        first_fixed_at = ""
        failed_seen = False
        for row in case_rows:
            is_bad = row["status"] in ("不一致(MISMATCH)", "异常(ERROR)")
            is_good = row["status"] == "一致(MATCH)"
            if not failed_seen and is_bad:
                first_failed_at = row["run_at"]
                failed_seen = True
            if failed_seen and is_good:
                first_fixed_at = row["run_at"]
                break
        latest = case_rows[-1]
        cls = status_class(latest["status"])
        html_parts.append(f"<tr><td><code>{escape(case_id)}</code></td><td>{escape(first_failed_at or '-')}</td><td>{escape(first_fixed_at or '-')}</td><td><span class=\"{cls}\">{escape(latest['status'])}</span></td><td><code>{escape(latest['run_id'])}</code></td></tr>")
    html_parts.append("</tbody></table></div></div>")
    html_parts.append('<div class="card"><h2>最近运行明细（最近 200 条）</h2><div class="table-wrap"><table><thead><tr>'
                      "<th>运行时间</th><th>运行ID</th><th>App</th><th>Case ID</th><th>预期</th><th>实际</th><th>状态</th>"
                      "</tr></thead><tbody>")
    for row in rows[-200:]:
        cls = status_class(row["status"])
        html_parts.append(f"<tr><td>{escape(row['run_at'])}</td><td><code>{escape(row['run_id'])}</code></td><td>{escape(row['app'])}</td><td><code>{escape(row['case_id'])}</code></td><td>{escape(row['expected_passed'])}</td><td>{escape(row['actual_passed'])}</td><td><span class=\"{cls}\">{escape(row['status'])}</span></td></tr>")
    html_parts.append("</tbody></table></div></div>")
    timeline_html.write_text(html_page(f"测试历史时间线({checker})", "".join(html_parts)), encoding="utf-8")


def write_failure_view(rows: List[Dict[str, Any]], run_meta: Dict[str, str], failure_html: Path, checker: str) -> None:
    bad_rows = sorted([r for r in rows if r.get("status") in ("不一致(MISMATCH)", "异常(ERROR)")], key=lambda x: (x["app"], x["case_id"]))
    html_parts: List[str] = []
    failure_html.parent.mkdir(parents=True, exist_ok=True)
    html_parts.append(f'<div class="card"><h1>不一致与异常视图 ({escape(checker)})</h1><ul>')
    html_parts.append(f"<li><strong>运行ID (Run ID)</strong>: <code>{escape(run_meta['run_id'])}</code></li>")
    html_parts.append(f"<li><strong>运行时间 (UTC)</strong>: <code>{escape(run_meta['run_at'])}</code></li>")
    html_parts.append(f"<li><strong>总问题数</strong>: {len(bad_rows)}</li></ul></div>")
    image_headers = "<th>Template</th><th>Target</th><th>Match Result</th>"
    if checker == "button_color":
        image_headers = "<th>原图 Before</th><th>原图 After</th><th>按钮 Before</th><th>按钮 After</th>"
    no_data_colspan = 13 if checker == "button_color" else 12
    html_parts.append('<div class="card"><div class="table-wrap"><table><thead><tr>'
                      f"<th>App</th><th>Case ID</th><th>预期</th><th>实际</th><th>状态</th><th>相似度</th><th>预期框</th><th>实际框</th><th>错误信息</th>{image_headers}"
                      "</tr></thead><tbody>")
    if not bad_rows:
        html_parts.append(f'<tr><td colspan="{no_data_colspan}">当前运行无不一致/异常</td></tr>')
    else:
        for row in bad_rows:
            cls = status_class(row["status"])

            def render_image_cell(abs_path: str, alt_text: str, *, is_template: bool = False) -> str:
                if not abs_path:
                    return "<td>-</td>"
                rel = os.path.relpath(abs_path, failure_html.parent)
                href = escape(rel)
                alt = escape(alt_text)
                image_style = "width:72px; height:72px; object-fit:contain; background:#fff;" if is_template else "height:72px; max-width:180px; object-fit:contain; background:#fff;"
                return f'<td><a href="{href}" target="_blank" rel="noopener noreferrer"><img src="{href}" alt="{alt}" style="{image_style} border:1px solid #e5e7eb; border-radius:6px;" /></a></td>'

            full_error = escape((row["error"] or "-").replace(chr(10), " "))
            html_parts.append("<tr>")
            html_parts.append(f"<td>{escape(row['app'])}</td><td><code>{escape(row['case_id'])}</code></td><td>{escape(row['expected_passed'])}</td><td>{escape(row['actual_passed'])}</td>")
            html_parts.append(f'<td><span class="{cls}">{escape(row["status"])}</span></td><td>{escape(row["similarity"] or "-")}</td>')
            html_parts.append(f"<td><code>{escape(row['expected_bounds'])}</code></td><td><code>{escape(row['actual_bounds'] or '-')}</code></td>")
            html_parts.append(f'<td class="error-cell" title="{full_error}">{full_error}</td>')
            if checker == "button_color":
                html_parts.append(render_image_cell(row.get("preview_template_image", ""), f"{row['case_id']} before"))
                html_parts.append(render_image_cell(row.get("preview_target_image", ""), f"{row['case_id']} after"))
                html_parts.append(render_image_cell(row.get("preview_button_before_image", ""), f"{row['case_id']} button before", is_template=True))
                html_parts.append(render_image_cell(row.get("preview_button_after_image", ""), f"{row['case_id']} button after", is_template=True))
            else:
                html_parts.append(render_image_cell(row.get("preview_template_image", ""), f"{row['case_id']} template", is_template=True))
                html_parts.append(render_image_cell(row.get("preview_target_image", ""), f"{row['case_id']} target"))
                html_parts.append(render_image_cell(row.get("preview_match_result_image", ""), f"{row['case_id']} match result"))
            html_parts.append("</tr>")
    html_parts.append("</tbody></table></div></div>")
    failure_html.write_text(html_page(f"不一致与异常视图({checker})", "".join(html_parts)), encoding="utf-8")


def run(note: str, image_output_root: Path, source: str, checker_name: str) -> None:
    paths = report_paths_for_checker(checker_name)
    history_paths = history_paths_for_checker(checker_name)
    if source == "pytest":
        run_rows, run_meta = load_latest_run_rows_from_history(checker_name)
        if not run_rows:
            raise RuntimeError(f"未找到 {checker_name} 的 pytest 结果。请先执行 pytest，再运行 --source pytest")
        if note:
            run_meta["note"] = note
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
    run_meta = {"run_id": run_id, "run_at": run_at, "commit": git_info["commit"], "branch": git_info["branch"], "note": note}
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

    append_csv_rows(
        history_paths["runs_csv"],
        ["run_id", "run_at", "branch", "commit", "note", "checker", "case_count"],
        [{**run_meta, "checker": checker_name, "case_count": len(run_rows)}],
    )
    append_csv_rows(
        history_paths["results_csv"],
        ["run_id", "run_at", "git_commit", "checker", "app", "case_id", "case_file", "template_image", "target_image", "threshold", "expected_passed", "expected_bounds", "actual_passed", "status", "similarity", "actual_bounds", "error", "preview_template_image", "preview_target_image", "preview_match_result_image", "preview_button_before_image", "preview_button_after_image"],
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
