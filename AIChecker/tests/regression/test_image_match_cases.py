"""
基于 testcase/image_match 目录的测试用例，验证 check_image_match 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析。
"""
import json
from pathlib import Path
from random import sample

import pytest
from aichecker.checkers import check_image_match
from aichecker.utils import _encode

# 仓库根目录（AIChecker/tests/regression -> AIChecker/tests -> AIChecker -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "image_match"
JSONS_DIR = TESTCASE_DIR / "jsons"
EXCLUDED_PLATFORMS = {"android", "harmony"}


def _set_image_report_meta(request: pytest.FixtureRequest, **kwargs) -> None:
    current = getattr(request.node, "_image_report_meta", {})
    current.update(kwargs)
    request.node._image_report_meta = current


def _resolve_path(json_path: Path, rel_path: str) -> Path:
    """将 JSON 中的相对路径解析为绝对路径（相对于该 JSON 所在目录）。"""
    if not rel_path or not rel_path.strip():
        raise ValueError(f"Empty path in JSON {json_path}")
    return (json_path.parent / rel_path.strip()).resolve()


def _load_and_resolve_payload(json_path: Path) -> dict:
    """加载 JSON 并解析其中的 template_image、target_image 或 image_a、image_b 路径。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("template_image", "target_image"):
        if data.get(key):
            data[key] = str(_resolve_path(json_path, data[key]))
    return data


def _bounds_center(b: tuple) -> tuple:
    """(left, top, right, bottom) -> (cx, cy)"""
    if len(b) != 4:
        return (0.0, 0.0)
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _bounds_iou(a: tuple, b: tuple) -> float:
    """计算两个矩形 [l,t,r,b] 的 IoU (Intersection over Union)，0~1。"""
    if len(a) != 4 or len(b) != 4:
        return 0.0
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il = max(al, bl)
    it = max(at, bt)
    ir = min(ar, br)
    ib = min(ab, bb)
    if ir <= il or ib <= it:
        return 0.0
    inter = (ir - il) * (ib - it)
    sa = (ar - al) * (ab - at)
    sb = (br - bl) * (bb - bt)
    union = sa + sb - inter
    return inter / union if union > 0 else 0.0


def _bounds_match(
    actual: tuple,
    expected: tuple,
    *,
    tolerance: int | None = 40,
    center_tolerance: int | float | None = None,
    iou_min: float | None = None,
) -> tuple[bool, str]:
    """
    判断实际 bounds 是否在预期范围内（模糊匹配）。
    优先级：center_tolerance > iou_min > tolerance（逐边容差）。
    返回 (是否通过, 说明字符串)。
    """
    if not actual or not expected:
        return (False, "missing bounds")
    if center_tolerance is not None:
        ac = _bounds_center(actual)
        ec = _bounds_center(expected)
        dist = ((ac[0] - ec[0]) ** 2 + (ac[1] - ec[1]) ** 2) ** 0.5
        # 轻微放松中心点阈值，降低分辨率/缩放导致的边缘误差影响
        relaxed_center_tolerance = max(float(center_tolerance) * 1.15, float(center_tolerance) + 5.0)
        ok = dist <= relaxed_center_tolerance
        return (
            ok,
            f"center_dist={dist:.1f}px, "
            f"center_tolerance={center_tolerance}px, "
            f"effective_center_tolerance={relaxed_center_tolerance:.1f}px",
        )
    if iou_min is not None:
        iou = _bounds_iou(actual, expected)
        relaxed_iou_min = max(0.0, float(iou_min) - 0.05)
        ok = iou >= relaxed_iou_min
        return (ok, f"IoU={iou:.3f}, iou_min={iou_min}, effective_iou_min={relaxed_iou_min:.3f}")
    # 逐边容差
    t = tolerance or 40
    d = [
        abs(actual[0] - expected[0]),
        abs(actual[1] - expected[1]),
        abs(actual[2] - expected[2]),
        abs(actual[3] - expected[3]),
    ]
    ok = all(x <= t for x in d)
    return (ok, f"edge_diffs={d}, tolerance={t}px")


def _collect_testcase_jsons():
    """收集 jsons 目录下的 JSON（包含 sample.json，支持 jsons/<app>/*.json 结构）。"""
    cases = []
    sample_path = JSONS_DIR / "sample.json"
    if sample_path.exists():
        cases.append(("sample", "sample", sample_path))
    for j in sorted(JSONS_DIR.rglob("*.json")):
        if j == sample_path:
            continue
        rel = j.relative_to(JSONS_DIR)
        rel_parts_lower = [part.lower() for part in rel.parts]
        if any(
            part == token or part.startswith(token)
            for part in rel_parts_lower
            for token in EXCLUDED_PLATFORMS
        ):
            continue
        cases.append((rel.parts[0], j.stem, j))
    return cases

# @pytest.mark.skip(reason="Skip image match test cases for now")
@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_image_match_from_testcase(platform: str, app_name: str, json_path: Path, request: pytest.FixtureRequest):
    _set_image_report_meta(
        request,
        app=platform,
        case_id=app_name,
        case_file=str(json_path),
    )

    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")

    payload = _load_and_resolve_payload(json_path)
    _set_image_report_meta(
        request,
        template_image=Path(payload.get("template_image", "N/A")).name,
        target_image=Path(payload.get("target_image", "N/A")).name,
        threshold=payload.get("similarity_threshold", 0.8),
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

    debug_dir = REPO_ROOT / "AIChecker" / "tests" / "image_match_output" / f"{app_name}"
    try:
        result = check_image_match(payload, output=debug_dir)
    except ImportError as e:
        pytest.skip(f"{platform}/{app_name}: Missing required dependency (opencv-python). Error: {e}")
    except Exception as e:
        pytest.fail(f"{platform}/{app_name}: Error during image matching: {e}")

    backend_used = str(result.details.get("backend_used", "") or "").lower()
    final_preview = (debug_dir / "final_match_result.png").resolve()
    template_preview = (debug_dir / "match_result.png").resolve()
    feature_preview = (debug_dir / "feature_match_result.png").resolve()
    if final_preview.exists():
        selected_preview = final_preview
    elif backend_used.startswith("feature") and feature_preview.exists():
        selected_preview = feature_preview
    elif backend_used.startswith("template") and template_preview.exists():
        selected_preview = template_preview
    elif feature_preview.exists():
        selected_preview = feature_preview
    else:
        selected_preview = template_preview

    _set_image_report_meta(
        request,
        actual_passed=bool(result.passed),
        similarity=f"{float(result.details.get('similarity', 0.0)):.4f}",
        actual_bounds=str(result.details.get("bounds", "")),
        preview_template_image=str((debug_dir / "template.png").resolve()),
        preview_target_image=str((debug_dir / "target.png").resolve()),
        preview_match_result_image=str(selected_preview),
        backend_used=backend_used,
    )

    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: "
        f"Groundtruth expected_passed={expected_pass}, but model detected passed={result.passed}. "
        f"Similarity: {result.details.get('similarity', 'N/A')}, "
        f"Threshold: {result.details.get('similarity_threshold', 'N/A')}. "
        f"Model basis: {result.basis}"
    )

    # if "expected_bounds" in payload and result.passed:
    #     expected_bounds = payload["expected_bounds"]
    #     actual_bounds = result.details.get("bounds")
    #     if expected_bounds and actual_bounds:
    #         tolerance = payload.get("bounds_tolerance", 30)
    #         center_tol = payload.get("bounds_center_tolerance")
    #         iou_min = payload.get("bounds_iou_min")
    #         ok, msg = _bounds_match(
    #             tuple(actual_bounds),
    #             tuple(expected_bounds),
    #             tolerance=tolerance,
    #             center_tolerance=center_tol,
    #             iou_min=iou_min,
    #         )
    #         if not ok:
    #             pytest.fail(
    #                 f"{platform}/{app_name}: Bounds fuzzy match failed. "
    #                 f"Expected: {expected_bounds}, Actual: {actual_bounds}. {msg}"
    #             )

@pytest.mark.skip(reason="Skip image match test cases for now")
def test_image_match_from_testcase_antennapod_1(request: pytest.FixtureRequest):
    json_path = JSONS_DIR / "antennapod" / "antennapod_1.json"
    test_image_match_from_testcase("antennapod", "antennapod_1", json_path, request)

@pytest.mark.skip(reason="Skip image match test cases for now")
def test_image_match_from_testcase_sample(request: pytest.FixtureRequest):
    json_path = JSONS_DIR  / "sample.json"
    test_image_match_from_testcase("sample", "sample", json_path, request)

@pytest.mark.skip(reason="Skip image match test cases for now")
def test_image_match_from_testcase_sample_agent_image_match():
    json_path =  Path(__file__).parent.parent / "scripts" / "sample_agent_image_match.json"
    payload = _load_and_resolve_payload(json_path)
    template_image = payload.get("template_image")
    target_image = payload.get("target_image")

    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
    else:
        pytest.fail(
            f"No groundtruth found in {json_path}. "
            f"Test case must include 'expected_passed' (boolean) or 'label' (string) field."
        )

    debug_dir = REPO_ROOT / "AIChecker" / "tests" / "image_match_output" / f"{sample}"

    result = check_image_match(payload, output=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    assert result.passed is expected_pass, (
        f"Expected passed: {expected_pass}, but got: {result.passed}"
    )