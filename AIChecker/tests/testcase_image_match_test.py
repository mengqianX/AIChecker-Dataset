"""
基于 testcase/image_match 目录的测试用例，验证 check_image_match 模块。
JSON 中的路径相对于各 JSON 文件所在目录解析。
"""
import json
from math import e
from pathlib import Path
import re

import pytest
from aichecker.checkers import check_image_match
from aichecker.utils import _encode

# 仓库根目录（AIChecker/tests -> AIChecker -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTCASE_DIR = REPO_ROOT / "testcase" / "image_match"
JSONS_DIR = TESTCASE_DIR / "jsons"


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
    tolerance: int | None = 30,
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
        ok = dist <= center_tolerance
        return (ok, f"center_dist={dist:.1f}px, center_tolerance={center_tolerance}px")
    if iou_min is not None:
        iou = _bounds_iou(actual, expected)
        ok = iou >= iou_min
        return (ok, f"IoU={iou:.3f}, iou_min={iou_min}")
    # 逐边容差
    t = tolerance or 30
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
    # 包含 sample.json
    sample_path = JSONS_DIR / "sample.json"
    if sample_path.exists():
        cases.append(("sample", "sample", sample_path))

    # 递归收集其余 JSON：
    # - jsons/<app>/<case>.json -> (app, case, path)
    for j in sorted(JSONS_DIR.rglob("*.json")):
        if j == sample_path:
            continue
        rel = j.relative_to(JSONS_DIR)
        cases.append((rel.parts[0], j.stem, j))
    return cases


@pytest.mark.parametrize("platform,app_name,json_path", _collect_testcase_jsons())
def test_image_match_from_testcase(platform: str, app_name: str, json_path: Path):
    """
    使用 testcase/image_match 中各应用的 JSON 跑图片匹配检测。
    
    重要说明：
    - expected_passed 是人工标注的groundtruth，表示该测试用例期望的检测结果
    - 模型只负责检测是否找到匹配，返回 passed（True/False）
    - 测试脚本会比较模型的检测结果与groundtruth，验证模型准确性
    """
    if not json_path.exists():
        pytest.skip(f"JSON not found: {json_path}")
    
    payload = _load_and_resolve_payload(json_path)
    
    # 检查图片文件是否存在
    template_image = payload.get("template_image")
    target_image = payload.get("target_image")
    
    if not template_image or not target_image:
        pytest.skip(f"Missing images in {json_path}")
    
    for img_path in (template_image, target_image):
        if img_path and not Path(img_path).exists():
            pytest.skip(f"Image not found: {img_path}")
    
    # 读取人工标注的groundtruth（必须在调用检测之前读取）
    # expected_passed 是人工标注的期望结果，不是模型判断的结果
    if "expected_passed" in payload:
        expected_pass = bool(payload["expected_passed"])
        groundtruth_source = f"expected_passed={expected_pass}"
    elif "label" in payload:
        expected_pass = payload.get("label") == "pass"
        groundtruth_source = f"label={payload.get('label')}"
    else:
        pytest.fail(
            f"No groundtruth found in {json_path}. "
            f"Test case must include 'expected_passed' (boolean) or 'label' (string) field "
            f"as manually annotated groundtruth."
        )
    
    # 调用检测器进行匹配（检测器只负责检测，不依赖groundtruth）
    debug_dir = REPO_ROOT / "AIChecker" / "debug" / "image_match" / f"{platform}_{app_name}"
    
    try:
        result = check_image_match(payload, output=debug_dir)
    except ImportError as e:
        # ImportError（如缺少cv2包）应该跳过，这是环境配置问题
        pytest.skip(
            f"{platform}/{app_name}: Missing required dependency (opencv-python). "
            f"Error: {e}"
        )
    except Exception as e:
        # 其他错误也应该记录
        pytest.fail(
            f"{platform}/{app_name}: Error during image matching: {e}"
        )
    
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
    
    # 比较模型的检测结果与人工标注的groundtruth
    # 如果两者不一致，说明模型检测有误
    assert result.passed is expected_pass, (
        f"{platform}/{app_name}: "
        f"Groundtruth (manually annotated) expected_passed={expected_pass}, "
        f"but model detected passed={result.passed}. "
        f"Similarity: {result.details.get('similarity', 'N/A')}, "
        f"Threshold: {result.details.get('similarity_threshold', 'N/A')}. "
        f"Model basis: {result.basis}"
    )
    
    # 如果提供了 expected_bounds，做模糊匹配验证（可选）
    # 支持三种方式，优先级：bounds_center_tolerance > bounds_iou_min > bounds_tolerance
    if "expected_bounds" in payload and result.passed:
        expected_bounds = payload["expected_bounds"]
        actual_bounds = result.details.get("bounds")
        if expected_bounds and actual_bounds:
            tolerance = payload.get("bounds_tolerance", 30)
            center_tol = payload.get("bounds_center_tolerance")
            iou_min = payload.get("bounds_iou_min")
            ok, msg = _bounds_match(
                tuple(actual_bounds),
                tuple(expected_bounds),
                tolerance=tolerance,
                center_tolerance=center_tol,
                iou_min=iou_min,
            )
            if not ok:
                pytest.fail(
                    f"{platform}/{app_name}: Bounds fuzzy match failed. "
                    f"Expected: {expected_bounds}, Actual: {actual_bounds}. "
                    f"{msg}"
                )


def test_sample_image_match():
    """测试示例用例（用于演示）。"""
    sample_path = JSONS_DIR / "sample.json"
    if not sample_path.exists():
        pytest.skip(f"Sample JSON not found: {sample_path}")
    
    payload = _load_and_resolve_payload(sample_path)
    
    # 检查图片文件是否存在
    template_image = payload.get("template_image")
    target_image = payload.get("target_image")
    expected_bounds = payload.get("expected_bounds")
    if expected_bounds:
        expected_bounds = tuple(expected_bounds)
    else:
        expected_bounds = None
    
    if template_image and target_image and Path(template_image).exists() and Path(target_image).exists():
        result = check_image_match(payload, output=REPO_ROOT / "AIChecker" / "debug" / "image_match" / "sample")
        print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))
        assert result.passed is payload.get("expected_passed")
        print(f"\n检测结果: {'✅ 找到匹配' if result.passed else '❌ 未找到匹配'}")
        print(f"相似度: {result.details.get('similarity', 0):.2%}")
        if expected_bounds:
            ok, msg = _bounds_match(result.details.get('bounds'), expected_bounds)
            assert ok is True, f"Bounds fuzzy match failed. {msg}"
        else:
            assert result.details.get('bounds') is not None
        print(f"匹配位置: {result.details.get('bounds', 'N/A')}")
        print(f"判定依据: {result.basis}")
    else:
        pytest.skip(f"Images not found or placeholder: {sample_path}")
