"""
Standalone OCR accuracy tests for image_match OCR fallback.

These tests bypass the main image_match pipeline and call run_ocr_fallback
directly, so we can isolate OCR localization quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
from PIL import Image

from aichecker.checkers.image_match_checker_ocr import run_ocr_fallback


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RILI_JSON_DIR = REPO_ROOT / "testcase" / "image_match" / "jsons" / "rili"
OCR_DEBUG_DIR = REPO_ROOT / "AIChecker" / "tests" / "image_match_output"


def _resolve_path(json_path: Path, rel_path: str) -> Path:
    return (json_path.parent / rel_path.strip()).resolve()


def _collect_rili_json_cases() -> list[Path]:
    return sorted(RILI_JSON_DIR.glob("*.json"))


@pytest.mark.parametrize("json_path", _collect_rili_json_cases())
@pytest.mark.parametrize(
    "min_box_size",
    [80],
)
def test_rili_ocr_localization_accuracy(
    json_path: Path,
    min_box_size: int,
) -> None:
    payload_json = json.loads(json_path.read_text(encoding="utf-8"))
    template_path = _resolve_path(json_path, str(payload_json["template_image"]))
    target_path = _resolve_path(json_path, str(payload_json["target_image"]))
    expected_passed = bool(payload_json.get("expected_passed", False))
    assert template_path.exists(), f"Missing template image: {template_path}"
    assert target_path.exists(), f"Missing target image: {target_path}"

    target_w, target_h = Image.open(target_path).size
    payload: Dict[str, object] = {
        "ocr_fallback_enabled": True,
        "ocr_include_global_roi": True,
        "ocr_allow_full_roi_bounds": False,
        "ocr_auto_generate_rois": True,
        "ocr_auto_max_rois": 8,
        "ocr_max_rois": 8,
        "ocr_pass_threshold": 0.42,
        "ocr_max_input_side": 2048,
        "ocr_token_min_conf": 0.20,
        "ocr_affine_min_inliers_for_bounds": 4,
        "ocr_min_matched_tokens_for_precise_bounds": 3,
        "ocr_precise_bounds_recall_min": 0.20,
        "ocr_precise_bounds_highcov_min": 0.20,
        # OCR for text boxes only; mask text then match structure.
        "ocr_structure_mask_text_enabled": True,
        "ocr_structure_tm_bounds_min_score": 0.45,
    }
    pass_threshold = 0.42

    output_dir = OCR_DEBUG_DIR / f"{json_path.stem}_ocr_accuracy"
    result = run_ocr_fallback(
        payload=payload,
        template_path=str(template_path),
        target_path=str(target_path),
        candidate_bounds=[(0, 0, target_w, target_h)],
        output=output_dir,
    )

    assert bool(result.get("passed", False)) is expected_passed, (
        f"OCR result mismatch for {json_path.name}: expected_passed={expected_passed}, "
        f"actual={result.get('passed')}, result={result}"
    )

    score = float(result.get("ocr_score", 0.0) or 0.0)
    bounds = tuple(result.get("bounds", (0, 0, 0, 0)))
    assert len(bounds) == 4
    assert bounds != (0, 0, 0, 0), f"OCR returned empty bounds. result={result}"
    bw = int(bounds[2]) - int(bounds[0])
    bh = int(bounds[3]) - int(bounds[1])

    best_meta = ((result.get("details") or {}).get("best") or {})
    bounds_source = str(best_meta.get("bounds_source", ""))
    best_scores = (((result.get("details") or {}).get("best") or {}).get("scores") or {})

    if expected_passed:
        assert score >= pass_threshold, (
            f"OCR score too low for positive case {json_path.name}: score={score:.4f}, "
            f"threshold={pass_threshold:.4f}, bounds={bounds}, "
            f"bounds_source={bounds_source}, best_scores={best_scores}"
        )
    else:
        assert score < pass_threshold, (
            f"OCR score too high for negative case {json_path.name}: score={score:.4f}, "
            f"threshold={pass_threshold:.4f}, bounds={bounds}, "
            f"bounds_source={bounds_source}, best_scores={best_scores}"
        )

    assert bw >= min_box_size and bh >= min_box_size, (
        f"OCR localization box too small for {json_path.name}: bounds={bounds}, "
        f"box_size=({bw},{bh}), min_box_size={min_box_size}, score={score:.4f}, "
        f"bounds_source={bounds_source}, best_scores={best_scores}"
    )
    assert bounds != (0, 0, target_w, target_h), (
        f"OCR should not fallback to full image bounds for {json_path.name}. "
        f"bounds={bounds}, score={score:.4f}, bounds_source={bounds_source}, best_scores={best_scores}"
    )
