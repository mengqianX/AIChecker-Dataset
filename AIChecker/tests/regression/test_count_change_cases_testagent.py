"""
基于 testcase/count_change 目录的测试用例，验证 TestAgent count_change 链路。
JSON 中的路径相对于各 JSON 文件所在目录解析。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "count_change"
JSONS_DIR = TESTCASE_DIR / "jsons"
TEST_AGENT_ROOT = REPO_ROOT / "AIChecker" / "TestAgent"
TEST_AGENT_MAIN = TEST_AGENT_ROOT / "main.py"
REPORT_PATH_PATTERN = re.compile(r"测试报告已生成:\s*(.+)$", re.MULTILINE)


def _set_checker_report_meta(request: pytest.FixtureRequest, **kwargs) -> None:
    current = getattr(request.node, "_checker_report_meta", {})
    current.update(kwargs)
    request.node._checker_report_meta = current


def _resolve_path(json_path: Path, rel_path: str) -> Path:
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()


def _load_and_resolve_payload(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("screenshot_a", "screenshot_b"):
        if data.get(key):
            data[key] = str(_resolve_path(json_path, data[key]))
    # 让主仓库 case 能无缝走 TestAgent targeted 模式
    data.setdefault("task_type", data.get("type", "count_change"))
    data.setdefault("mode", "targeted")
    return data


def _collect_testcase_jsons():
    cases = []
    for j in sorted(JSONS_DIR.rglob("*.json")):
        if "sample" in j.stem.lower():
            continue
        rel = j.relative_to(JSONS_DIR)
        app = rel.parts[0] if rel.parts else j.stem
        cases.append((app, j.stem, j))
    return cases


def _extract_expected_passed(payload: dict[str, Any], json_path: Path) -> bool:
    if "expected_passed" in payload:
        return bool(payload["expected_passed"])
    if "label" in payload:
        return payload.get("label") == "pass"
    raise AssertionError(
        f"No groundtruth found in {json_path}. "
        "Test case must include 'expected_passed' (boolean) or 'label' (string) field."
    )


def _run_test_agent(payload: dict[str, Any]) -> tuple[int, str]:
    cmd = [sys.executable, str(TEST_AGENT_MAIN), "--input-json", json.dumps(payload, ensure_ascii=False)]
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    proc = subprocess.Popen(
        cmd,
        cwd=str(TEST_AGENT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    return proc.wait(), "".join(output_lines)


def _resolve_report_path(output: str) -> Path:
    matches = REPORT_PATH_PATTERN.findall(output)
    if not matches:
        raise RuntimeError("运行输出中未找到报告路径（测试报告已生成）")
    return Path(matches[-1].strip())


def _resolve_task_pass(report: dict[str, Any]) -> tuple[bool, str]:
    task_type = str(report.get("input", {}).get("task_type", "")).strip().lower()
    for seg in report.get("segment_results", []):
        if not isinstance(seg, dict):
            continue
        if task_type == "count_change" and seg.get("detector") == "count_change_detector":
            return (not bool(seg.get("bug_detected", False))), "count_change.bug_detected"
    bug_detected = bool(report.get("video_level_result", {}).get("bug_detected", False))
    return (not bug_detected), "video_level_result.bug_detected"


@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_count_change_from_testcase(
    platform: str, app_name: str, json_path: Path, request: pytest.FixtureRequest
):
    _set_checker_report_meta(
        request,
        checker="count_change",
        app=platform,
        case_id=app_name,
        case_file=str(json_path),
    )
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")
    if not TEST_AGENT_MAIN.exists():
        pytest.skip(f"TestAgent main.py not found: {TEST_AGENT_MAIN}")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for TestAgent count_change regression tests")

    payload = _load_and_resolve_payload(json_path)
    _set_checker_report_meta(
        request,
        template_image=Path(payload.get("screenshot_a", "N/A")).name,
        target_image=Path(payload.get("screenshot_b", "N/A")).name,
    )
    bounds = payload.get("bounds", [])
    if bounds == [0, 0, 0, 0]:
        pytest.skip(f"Skipping placeholder test case: {json_path} (bounds not set)")
    if not payload.get("screenshot_a") or not payload.get("screenshot_b"):
        pytest.skip(f"Missing screenshots in {json_path}")
    for p in (payload.get("screenshot_a"), payload.get("screenshot_b")):
        if p and not Path(p).exists():
            pytest.skip(f"Screenshot not found: {p}")

    expected_pass = _extract_expected_passed(payload, json_path)
    _set_checker_report_meta(request, expected_passed=expected_pass)

    code, output = _run_test_agent(payload)
    if code != 0 and "OPENAI_API_KEY" in output:
        pytest.skip(f"{platform}/{app_name}: missing OPENAI_API_KEY")
    assert code == 0, f"TestAgent main.py exited with code={code}\n{output}"

    report_path = _resolve_report_path(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_pass, source = _resolve_task_pass(report)

    assert actual_pass == expected_pass, (
        f"{platform}/{app_name}: Groundtruth expected_passed={expected_pass}, "
        f"but model detected passed={actual_pass}. source={source}, report={report_path}"
    )
    _set_checker_report_meta(request, actual_passed=bool(actual_pass))