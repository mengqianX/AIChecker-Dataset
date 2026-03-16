"""
基于 testcase/button_color_change 目录的测试用例，验证 check_button_color 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析；expected_color 为空时使用自动颜色变化检测。
"""
import json
from pathlib import Path

import pytest
from aichecker.checkers import check_button_color
from aichecker.utils import _encode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "button_color_change"
JSONS_DIR = TESTCASE_DIR / "jsons"


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
    if data.get("expected_color") == "":
        data["expected_color"] = None
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


@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_button_color_change_from_testcase(
    platform: str, app_name: str, json_path: Path, request: pytest.FixtureRequest
):
    _set_checker_report_meta(
        request,
        checker="button_color",
        app=platform,
        case_id=app_name,
        case_file=str(json_path),
    )
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")
    payload = _load_and_resolve_payload(json_path)
    _set_checker_report_meta(
        request,
        template_image=Path(payload.get("screenshot_a", "N/A")).name,
        target_image=Path(payload.get("screenshot_b", "N/A")).name,
        expected_bounds=payload.get("bounds", "N/A"),
    )
    if not payload.get("screenshot_a") or not payload.get("screenshot_b"):
        pytest.skip(f"Missing screenshots in {json_path}")
    for p in (payload.get("screenshot_a"), payload.get("screenshot_b")):
        if p and not Path(p).exists():
            pytest.skip(f"Screenshot not found: {p}")
    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "crops" / f"{platform}_{app_name}"
    _set_checker_report_meta(
        request,
        preview_template_image=str(Path(payload["screenshot_a"]).resolve()),
        preview_target_image=str(Path(payload["screenshot_b"]).resolve()),
    )
    result = check_button_color(payload, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))

    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
        groundtruth_source = f"expected_passed={expected_pass}"
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
        groundtruth_source = f"label={payload.get('label')}"
    else:
        pytest.skip(f"No groundtruth found in {json_path} (missing 'expected_passed' or 'label')")
    _set_checker_report_meta(
        request,
        expected_passed=expected_pass,
        actual_passed=bool(result.passed),
        preview_button_before_image=str((debug_dir / "button_crop_before.png").resolve()),
        preview_button_after_image=str((debug_dir / "button_crop_after.png").resolve()),
    )

    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: expected passed={expected_pass} ({groundtruth_source}), "
        f"got passed={result.passed}, basis={result.basis}"
    )