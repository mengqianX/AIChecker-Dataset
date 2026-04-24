from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "testing" / "generate_test_report.py"
    spec = importlib.util.spec_from_file_location("generate_test_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enrich_rows_with_history_marks_regressions_and_repairs():
    report = _load_report_module()
    history_rows = [
        {
            "app": "aiqiyi",
            "case_id": "aiqiyi_1",
            "run_id": "old_1",
            "run_at": "2026-04-20T00:00:00+00:00",
            "status": "一致(MATCH)",
        },
        {
            "app": "aiqiyi",
            "case_id": "aiqiyi_2",
            "run_id": "old_1",
            "run_at": "2026-04-20T00:00:00+00:00",
            "status": "不一致(MISMATCH)",
        },
    ]
    current_rows = [
        {
            "app": "aiqiyi",
            "case_id": "aiqiyi_1",
            "run_id": "new_1",
            "run_at": "2026-04-24T00:00:00+00:00",
            "status": "异常(ERROR)",
            "error": "AssertionError: exploded",
        },
        {
            "app": "aiqiyi",
            "case_id": "aiqiyi_2",
            "run_id": "new_1",
            "run_at": "2026-04-24T00:00:00+00:00",
            "status": "一致(MATCH)",
            "error": "",
        },
    ]

    report.enrich_rows_with_history(current_rows, history_rows, "new_1")

    assert current_rows[0]["history_label"] == "新回归"
    assert current_rows[0]["previous_status"] == "一致(MATCH)"
    assert current_rows[0]["error_summary"].startswith("AssertionError")
    assert "exploded" in current_rows[0]["error_summary"]

    assert current_rows[1]["history_label"] == "已修复"
    assert current_rows[1]["previous_status"] == "不一致(MISMATCH)"
    assert current_rows[1]["last_issue_at"] == "2026-04-20T00:00:00+00:00"


def test_repair_preview_images_recovers_local_assets(tmp_path, monkeypatch):
    report = _load_report_module()

    json_root = tmp_path / "testcase" / "button_color_change" / "jsons"
    screen_root = tmp_path / "testcase" / "button_color_change" / "screens" / "demo"
    case_dir = json_root / "demo"
    case_dir.mkdir(parents=True)
    screen_root.mkdir(parents=True)

    before = screen_root / "demo_1_before.png"
    after = screen_root / "demo_1_after.png"
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    case_file = case_dir / "demo_1.json"
    case_file.write_text(
        '{"screenshot_a": "../../screens/demo/demo_1_before.png", "screenshot_b": "../../screens/demo/demo_1_after.png"}',
        encoding="utf-8",
    )

    monkeypatch.setitem(report.CASE_JSON_ROOTS, "button_color", json_root)
    report.case_lookup_for_checker.cache_clear()

    rows = [
        {
            "checker": "button_color",
            "app": "demo",
            "case_id": "demo_1",
            "case_file": "/obsolete/workspace/testcase/button_color_change/jsons/demo/demo_1.json",
            "preview_template_image": "",
            "preview_target_image": "",
        }
    ]

    report.repair_preview_images(rows, tmp_path / "unused_image_root")

    assert rows[0]["preview_template_image"] == str(before)
    assert rows[0]["preview_target_image"] == str(after)


def test_write_latest_snapshot_renders_filters_and_issue_digest(tmp_path):
    report = _load_report_module()
    report_path = tmp_path / "LATEST_SNAPSHOT.html"
    rows = [
        {
            "app": "aiqiyi",
            "case_id": "aiqiyi_3",
            "expected_passed": "通过(Pass)",
            "actual_passed": "异常(ERROR)",
            "status": "异常(ERROR)",
            "similarity": "",
            "threshold": "",
            "expected_bounds": "N/A",
            "actual_bounds": "",
            "error": "RuntimeError: demo",
            "error_summary": "RuntimeError: demo",
            "history_label": "新回归",
            "last_good_at": "2026-04-23T00:00:00+00:00",
            "preview_template_image": "",
            "preview_target_image": "",
            "preview_button_before_image": "",
            "preview_button_after_image": "",
        },
        {
            "app": "aiqiyi",
            "case_id": "aiqiyi_2",
            "expected_passed": "通过(Pass)",
            "actual_passed": "通过(Pass)",
            "status": "一致(MATCH)",
            "similarity": "",
            "threshold": "",
            "expected_bounds": "N/A",
            "actual_bounds": "",
            "error": "",
            "error_summary": "-",
            "history_label": "稳定通过",
            "last_good_at": "2026-04-24T00:00:00+00:00",
            "preview_template_image": "",
            "preview_target_image": "",
            "preview_button_before_image": "",
            "preview_button_after_image": "",
        },
    ]
    run_meta = {
        "run_id": "run_demo",
        "run_at": "2026-04-24T00:00:00+00:00",
        "branch": "main",
        "commit": "abcdef1",
        "note": "unit test",
    }

    report.write_latest_snapshot(rows, run_meta, report_path, "button_color")

    html = report_path.read_text(encoding="utf-8")
    assert "快速筛选" in html
    assert "优先关注问题" in html
    assert 'data-row-scope="snapshot"' in html
    assert "新回归" in html
    assert "app-summary-scroll" in html
