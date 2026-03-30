from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, cast

import cv2
import numpy as np

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import load_image
from .image_match_checker import _pil_to_cv2, _resolve_target_image, _resolve_template_image

# Feature-based matching defaults.
DEFAULT_FEATURE_SIMILARITY_THRESHOLD = 0.35
DEFAULT_ORB_NFEATURES = 2000
DEFAULT_RATIO_TEST_THRESHOLD = 0.75
DEFAULT_MIN_GOOD_MATCHES = 12
DEFAULT_MIN_INLIERS = 8


def _compute_similarity(
    num_template_kp: int,
    num_good_matches: int,
    num_inliers: int,
) -> float:
    """
    Compute a [0, 1]-like similarity score for feature matching.

    The score combines:
    - match coverage on template keypoints
    - geometric consistency from RANSAC inlier ratio
    """
    if num_template_kp <= 0:
        return 0.0

    coverage = min(1.0, num_good_matches / float(max(num_template_kp, 1)))
    inlier_ratio = (
        float(num_inliers) / float(num_good_matches)
        if num_good_matches > 0
        else 0.0
    )
    return 0.5 * coverage + 0.5 * inlier_ratio


def _clip_bounds(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    left = max(0, min(left, width))
    top = max(0, min(top, height))
    right = max(left, min(right, width))
    bottom = max(top, min(bottom, height))
    return left, top, right, bottom


def _rotate_image_90n(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by right-angle degrees: 0/90/180/270."""
    normalized = int(angle) % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported right-angle rotation: {angle}")


def _robust_bounds_from_points(
    points: np.ndarray,
    target_w: int,
    target_h: int,
    quantile: float = 0.05,
) -> Tuple[int, int, int, int]:
    """
    Build a robust bbox from matched points by trimming outliers.
    """
    if points.size == 0:
        return (0, 0, 0, 0)
    xs = points[:, 0]
    ys = points[:, 1]
    q = float(max(0.0, min(quantile, 0.4)))
    left = int(np.floor(np.quantile(xs, q)))
    top = int(np.floor(np.quantile(ys, q)))
    right = int(np.ceil(np.quantile(xs, 1.0 - q)))
    bottom = int(np.ceil(np.quantile(ys, 1.0 - q)))
    return _clip_bounds(left, top, right, bottom, target_w, target_h)


def _merge_bounds(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> Tuple[int, int, int, int]:
    """Union of two bboxes."""
    left = min(a[0], b[0])
    top = min(a[1], b[1])
    right = max(a[2], b[2])
    bottom = max(a[3], b[3])
    return _clip_bounds(left, top, right, bottom, target_w, target_h)


def _feature_match(
    template_bgr: np.ndarray,
    target_bgr: np.ndarray,
    nfeatures: int,
    ratio_test_threshold: float,
    min_good_matches: int,
    ransac_reproj_threshold: float,
) -> Dict[str, Any]:
    """Run ORB + BFMatcher + RANSAC and return raw match artifacts."""
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    orb_create = cast(Callable[..., cv2.ORB], getattr(cv2, "ORB_create"))
    orb = orb_create(nfeatures=nfeatures)
    kp_template, des_template = orb.detectAndCompute(template_gray, None)
    kp_target, des_target = orb.detectAndCompute(target_gray, None)

    result: Dict[str, Any] = {
        "kp_template": kp_template or [],
        "kp_target": kp_target or [],
        "good_matches": [],
        "homography": None,
        "inlier_mask": None,
        "inlier_count": 0,
    }

    if des_template is None or des_target is None:
        return result
    if len(result["kp_template"]) < 2 or len(result["kp_target"]) < 2:
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des_template, des_target, k=2)

    good_matches: List[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)

    result["good_matches"] = good_matches
    if len(good_matches) < min_good_matches:
        return result

    src_pts = np.float32(
        [result["kp_template"][m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [result["kp_target"][m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_reproj_threshold,
    )
    result["homography"] = homography
    result["inlier_mask"] = inlier_mask
    if inlier_mask is not None:
        result["inlier_count"] = int(inlier_mask.ravel().sum())

    return result


def check_image_match_feature(
    payload: Dict[str, Any],
    output: Path | None = None,
) -> CheckResult:
    """
    Feature-based image matching using ORB + BFMatcher + RANSAC.

    Payload fields:
    - template_image / template / small_image
    - target_image / target / large_image / screenshot
    - similarity_threshold (default 0.35)
    - orb_nfeatures (default 2000)
    - ratio_test_threshold (default 0.75)
    - min_good_matches (default 12)
    - min_inliers (default 8)
    - ransac_reproj_threshold (default 5.0)
    """
    template_path = _resolve_template_image(payload)
    target_path = _resolve_target_image(payload)

    similarity_threshold = float(
        payload.get("similarity_threshold", DEFAULT_FEATURE_SIMILARITY_THRESHOLD)
    )
    nfeatures = int(payload.get("orb_nfeatures", DEFAULT_ORB_NFEATURES))
    ratio_test_threshold = float(
        payload.get("ratio_test_threshold", DEFAULT_RATIO_TEST_THRESHOLD)
    )
    min_good_matches = int(payload.get("min_good_matches", DEFAULT_MIN_GOOD_MATCHES))
    min_inliers = int(payload.get("min_inliers", DEFAULT_MIN_INLIERS))
    ransac_reproj_threshold = float(payload.get("ransac_reproj_threshold", 5.0))

    template_pil = load_image(template_path)
    target_pil = load_image(target_path)

    # Keep the same convention as template matching: search small image in large image.
    tw, th = template_pil.size
    gw, gh = target_pil.size
    swapped_by_size = (tw * th) > (gw * gh)
    if swapped_by_size:
        template_pil, target_pil = target_pil, template_pil
        template_path, target_path = target_path, template_path

    template_cv = _pil_to_cv2(template_pil)
    target_cv = _pil_to_cv2(target_pil)
    target_h, target_w = target_cv.shape[:2]

    # Rotation-aware search for cross-device / orientation changes.
    rotation_search = bool(payload.get("rotation_search", True))
    rotation_angles = payload.get("rotation_angles", [0, 90, 180, 270])
    if not rotation_search:
        rotation_angles = [0]
    angle_candidates: List[int] = []
    for a in rotation_angles:
        try:
            normalized = int(a) % 360
        except (TypeError, ValueError):
            continue
        if normalized in (0, 90, 180, 270):
            angle_candidates.append(normalized)
    if not angle_candidates:
        angle_candidates = [0]

    best_angle = 0
    best_raw: Dict[str, Any] | None = None
    best_similarity = -1.0
    for angle in angle_candidates:
        rotated_template = _rotate_image_90n(template_cv, angle)
        raw = _feature_match(
            rotated_template,
            target_cv,
            nfeatures=nfeatures,
            ratio_test_threshold=ratio_test_threshold,
            min_good_matches=min_good_matches,
            ransac_reproj_threshold=ransac_reproj_threshold,
        )
        kp_template_candidate = raw["kp_template"]
        good_matches_candidate: List[cv2.DMatch] = raw["good_matches"]
        inlier_count_candidate = int(raw["inlier_count"])
        similarity_candidate = _compute_similarity(
            num_template_kp=len(kp_template_candidate),
            num_good_matches=len(good_matches_candidate),
            num_inliers=inlier_count_candidate,
        )
        candidate_key = (
            inlier_count_candidate,
            len(good_matches_candidate),
            similarity_candidate,
        )
        if best_raw is None:
            best_raw = raw
            best_angle = angle
            best_similarity = similarity_candidate
            best_key = candidate_key
            continue
        if candidate_key > best_key:
            best_raw = raw
            best_angle = angle
            best_similarity = similarity_candidate
            best_key = candidate_key

    assert best_raw is not None

    template_cv = _rotate_image_90n(template_cv, best_angle)
    kp_template = cast(List[cv2.KeyPoint], best_raw["kp_template"])
    kp_target = cast(List[cv2.KeyPoint], best_raw["kp_target"])
    good_matches = cast(List[cv2.DMatch], best_raw["good_matches"])
    homography = best_raw["homography"]
    inlier_mask = best_raw["inlier_mask"]
    inlier_count = int(best_raw["inlier_count"])

    bounds_tuple = (0, 0, 0, 0)
    polygon: List[Tuple[int, int]] = []
    matched_template_w = 0.0
    matched_template_h = 0.0

    if homography is not None:
        h_t, w_t = template_cv.shape[:2]
        corners = np.float32(
            [[0, 0], [w_t - 1, 0], [w_t - 1, h_t - 1], [0, h_t - 1]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        xs = projected[:, 0]
        ys = projected[:, 1]
        matched_template_w = float(xs.max() - xs.min())
        matched_template_h = float(ys.max() - ys.min())
        left, top, right, bottom = _clip_bounds(
            int(np.floor(xs.min())),
            int(np.floor(ys.min())),
            int(np.ceil(xs.max())),
            int(np.ceil(ys.max())),
            target_w,
            target_h,
        )
        bounds_tuple = (left, top, right, bottom)
        polygon = [
            (int(round(x)), int(round(y)))
            for x, y in projected
        ]
    # Robust fallback/merge: use inlier points (or good points) trimmed bbox.
    if good_matches:
        dst_pts_all = np.float32([kp_target[m.trainIdx].pt for m in good_matches])
        dst_pts_used = dst_pts_all
        if inlier_mask is not None and len(inlier_mask.ravel()) == len(good_matches):
            flags = inlier_mask.ravel().astype(bool)
            if flags.any():
                dst_pts_used = dst_pts_all[flags]
        robust_bounds = _robust_bounds_from_points(dst_pts_used, target_w, target_h, quantile=0.05)
        if bounds_tuple == (0, 0, 0, 0):
            bounds_tuple = robust_bounds
        else:
            bounds_tuple = _merge_bounds(bounds_tuple, robust_bounds, target_w, target_h)

    # Layout-adaptive horizontal expansion:
    # For wide strip-like UI components, some cross-device layouts stretch horizontally
    # while keeping similar vertical placement. Expand X bounds conservatively in that case.
    adaptive_expand_x = bool(payload.get("adaptive_expand_x", True))
    adaptive_expand_x_margin_ratio = float(payload.get("adaptive_expand_x_margin_ratio", 0.02))
    adaptive_expand_x_applied = False
    if adaptive_expand_x and homography is not None and bounds_tuple != (0, 0, 0, 0):
        template_h, template_w = template_cv.shape[:2]
        template_aspect = float(template_w) / float(max(template_h, 1))
        target_vs_template_w = float(target_w) / float(max(template_w, 1))
        matched_scale_x = matched_template_w / float(max(template_w, 1))
        # Trigger only when: wide strip + target much wider + observed match scale ~1x + enough inliers.
        should_expand_x = (
            template_aspect >= 3.0
            and target_vs_template_w >= 1.8
            and 0.75 <= matched_scale_x <= 1.35
            and inlier_count >= max(20, min_inliers * 2)
        )
        if should_expand_x:
            margin = int(round(target_w * max(0.0, min(adaptive_expand_x_margin_ratio, 0.2))))
            left, top, right, bottom = bounds_tuple
            bounds_tuple = _clip_bounds(margin, top, target_w - margin, bottom, target_w, target_h)
            adaptive_expand_x_applied = True

    similarity = best_similarity

    passed = (
        homography is not None
        and len(good_matches) >= min_good_matches
        and inlier_count >= min_inliers
        and similarity >= similarity_threshold
    )

    bounds = Bounds.from_sequence(bounds_tuple)
    if passed:
        basis = (
            f"image_match_feature: Found template in target image with similarity={similarity:.4f} "
            f"(threshold={similarity_threshold:.4f}), inliers={inlier_count}/{len(good_matches)}, "
            f"bounds={bounds.as_box()}"
        )
    else:
        basis = (
            f"image_match_feature: Template not found. "
            f"similarity={similarity:.4f} (threshold={similarity_threshold:.4f}), "
            f"good_matches={len(good_matches)}, inliers={inlier_count}, bounds={bounds.as_box()}"
        )

    control_info = ControlInfo(
        bounds=bounds,
        source="image_match_feature",
        extras={
            "similarity": similarity,
            "good_matches": len(good_matches),
            "inliers": inlier_count,
            "template_keypoints": len(kp_template),
            "target_keypoints": len(kp_target),
        },
    )

    details: Dict[str, Any] = {
        "method": "feature_orb_bf_ransac",
        "similarity": similarity,
        "similarity_threshold": similarity_threshold,
        "bounds": bounds_tuple,
        "polygon": polygon,
        "template_path": template_path,
        "target_path": target_path,
        "template_size": template_pil.size,
        "target_size": target_pil.size,
        "swapped_by_size": swapped_by_size,
        "feature_params": {
            "orb_nfeatures": nfeatures,
            "ratio_test_threshold": ratio_test_threshold,
            "min_good_matches": min_good_matches,
            "min_inliers": min_inliers,
            "ransac_reproj_threshold": ransac_reproj_threshold,
        },
        "feature_stats": {
            "template_keypoints": len(kp_template),
            "target_keypoints": len(kp_target),
            "good_matches": len(good_matches),
            "inliers": inlier_count,
            "homography_found": homography is not None,
            "rotation_search": rotation_search,
            "rotation_angles": angle_candidates,
            "best_rotation_angle": best_angle,
            "matched_template_width": matched_template_w,
            "matched_template_height": matched_template_h,
        },
        "layout_adaptive": {
            "adaptive_expand_x": adaptive_expand_x,
            "adaptive_expand_x_margin_ratio": adaptive_expand_x_margin_ratio,
            "adaptive_expand_x_applied": adaptive_expand_x_applied,
        },
    }

    if output:
        output.mkdir(parents=True, exist_ok=True)

        target_vis = target_cv.copy()
        if polygon:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(target_vis, [pts], True, (0, 255, 255), 2)
        cv2.rectangle(
            target_vis,
            (bounds.left, bounds.top),
            (bounds.right, bounds.bottom),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            target_vis,
            f"sim={similarity:.3f}, inliers={inlier_count}/{len(good_matches)}",
            (bounds.left, max(0, bounds.top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(output / "feature_match_result.png"), target_vis)

        h1, w1 = template_cv.shape[:2]
        h2, w2 = target_cv.shape[:2]
        out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        matches_vis = cv2.drawMatches(
            template_cv,
            kp_template,
            target_cv,
            kp_target,
            good_matches[:80],
            out_img,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(output / "feature_matches.png"), matches_vis)
        cv2.imwrite(str(output / "template.png"), template_cv)
        cv2.imwrite(str(output / "target.png"), target_cv)

    return CheckResult(
        passed=passed,
        basis=basis,
        control_info=control_info,
        details=details,
    )
