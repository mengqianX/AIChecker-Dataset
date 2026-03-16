"""
基于 testcase/image_match 目录的测试用例，验证 check_image_match_feature 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析。
默认跳过 Android 和 HarmonyOS 目录。
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from aichecker.checkers import check_image_match_feature
from aichecker.utils import _encode

# 仓库根目录（AIChecker/tests/regression -> AIChecker/tests -> AIChecker -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "image_match"
JSONS_DIR = TESTCASE_DIR / "jsons"
EXCLUDED_PLATFORMS = {"android", "harmony", "harmonyos"}


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
    for key in ("template_image", "target_image"):
        if data.get(key):
            data[key] = str(_resolve_path(json_path, data[key]))
    return data


def _collect_testcase_jsons():
    cases = []
    sample_path = JSONS_DIR / "sample.json"
    if sample_path.exists():
        cases.append(("sample", "sample", sample_path))

    for j in sorted(JSONS_DIR.rglob("*.json")):
        if j == sample_path:
            continue
        rel = j.relative_to(JSONS_DIR)
        if any(part.lower() in EXCLUDED_PLATFORMS for part in rel.parts):
            continue
        platform = rel.parts[0] if len(rel.parts) > 1 else "root"
        cases.append((platform, j.stem, j))
    return cases


@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_image_match_feature_from_testcase(
    platform: str,
    app_name: str,
    json_path: Path,
    request: pytest.FixtureRequest,
):
    _set_checker_report_meta(
        request,
        checker="image_match_feature",
        app=platform,
        case_id=app_name,
        case_file=str(json_path),
    )

    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")

    payload = _load_and_resolve_payload(json_path)
    _set_checker_report_meta(
        request,
        template_image=Path(payload.get("template_image", "N/A")).name,
        target_image=Path(payload.get("target_image", "N/A")).name,
        threshold=payload.get("similarity_threshold", 0.35),
        expected_passed=payload.get("expected_passed"),
        expected_bounds=payload.get("expected_bounds"),
    )

    template_image = payload.get("template_image")
    target_image = payload.get("target_image")
    if not template_image or not target_image:
        pytest.skip(f"Missing images in {json_path}")
    for img_path in (template_image, target_image):
        if img_path and not Path(img_path).exists():
            pytest.skip(f"Image not found: {img_path}")

    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
    else:
        pytest.fail(
            f"No groundtruth found in {json_path}. "
            f"Test case must include 'expected_passed' (boolean) or 'label' (string) field."
        )

    debug_dir = REPO_ROOT / "AIChecker" / "tests" / "image_match_output" / f"{app_name}_f"
    try:
        result = check_image_match_feature(payload, output=debug_dir)
    except ImportError as e:
        pytest.skip(f"{platform}/{app_name}: Missing required dependency (opencv-python). Error: {e}")
    except Exception as e:
        pytest.fail(f"{platform}/{app_name}: Error during feature image matching: {e}")

    _set_checker_report_meta(
        request,
        actual_passed=bool(result.passed),
        similarity=f"{float(result.details.get('similarity', 0.0)):.4f}",
        actual_bounds=str(result.details.get("bounds", "")),
        preview_template_image=str((debug_dir / "template.png").resolve()),
        preview_target_image=str((debug_dir / "target.png").resolve()),
        preview_match_result_image=str((debug_dir / "feature_match_result.png").resolve()),
    )

    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: "
        f"Groundtruth expected_passed={expected_pass}, but feature checker detected passed={result.passed}. "
        f"Similarity: {result.details.get('similarity', 'N/A')}, "
        f"Threshold: {result.details.get('similarity_threshold', 'N/A')}. "
        f"Model basis: {result.basis}"
    )
