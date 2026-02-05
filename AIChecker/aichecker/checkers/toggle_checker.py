from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from ..models import Bounds, CheckResult, ControlInfo
from ..utils import load_image
from .detectors import check_toggle_cv, check_toggle_uitree


_TOGGLE_KEYWORDS = ("toggle", "switch", "checkbox")


def _find_node_by_bounds(ui_tree: Dict[str, Any], bounds: Bounds) -> Optional[Dict[str, Any]]:
    target_box = bounds.as_box()
    stack = [ui_tree]
    while stack:
        node = stack.pop()
        attrs = node.get("attributes") or {}
        node_bounds = attrs.get("bounds") or attrs.get("origBounds")
        if node_bounds:
            box = _parse_bounds_box(node_bounds)
            if box and box == target_box:
                return node
        stack.extend(node.get("children") or [])
    return None


def _is_official_toggle(control_type: Any) -> bool:
    if control_type is None:
        return False
    normalized = str(control_type).strip().lower()
    return any(keyword in normalized for keyword in _TOGGLE_KEYWORDS)


_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def _coerce_boolish(raw: Any, allow_none: bool = False) -> Optional[bool]:
    """Parse common truthy/falsey strings/ints; optionally allow None/empty."""
    if raw is None or raw == "":
        if allow_none:
            return None
        raise ValueError(f"Cannot coerce None/empty value to bool (allow_none=False)")
    if isinstance(raw, bool):
        return raw
    val = str(raw).strip().lower()
    if val in _TRUE_VALUES:
        return True
    if val in _FALSE_VALUES:
        return False
    if allow_none:
        return None
    raise ValueError(f"Cannot coerce value to bool: {raw!r}")


def _coerce_checked_value(raw: Any) -> Optional[bool]:
    return _coerce_boolish(raw, allow_none=True)


def _coerce_expected_value(raw: Any) -> bool:
    """Coerce expected value to bool, raising error if cannot be parsed."""
    return _coerce_boolish(raw, allow_none=False)


def _extract_checked_from_attrs(attrs: Dict[str, Any]) -> Optional[bool]:
    candidates = [
        attrs.get("checked"),
        attrs.get("isChecked"),
        attrs.get("selected"),
        (attrs.get("state") or {}).get("checked"),
        (attrs.get("accessibility") or {}).get("checked"),
    ]
    for raw in candidates:
        checked = _coerce_checked_value(raw)
        if checked is not None:
            return checked
    return None


def check_toggle(payload: Dict[str, Any], debug_dir: Path | None = None) -> CheckResult:
    bounds = Bounds.from_sequence(payload["bounds"])
    raw_expected = payload.get("expected")
    expected = _coerce_expected_value(raw_expected)
    ui_tree = payload.get("ui_tree")

    node = _find_node_by_bounds(ui_tree, bounds) if ui_tree else None
    attrs = (node.get("attributes") or node) if node else {}
    control_type = attrs.get("type")
    is_official = _is_official_toggle(control_type)
    checked = _extract_checked_from_attrs(attrs) if is_official else None

    control = ControlInfo(
        bounds=bounds,
        text=attrs.get("text"),
        semantics=attrs.get("semantics"),
        checked=checked,
        extras={k: v for k, v in (attrs or {}).items() if k not in {"text", "semantics", "checked"}},
        source="ui_tree" if checked is not None else "cv",
    )

    if checked is not None:
        return check_toggle_uitree(control, expected)

    img_before = load_image(payload["screenshot_a"])
    img_after = load_image(payload["screenshot_b"])
    return check_toggle_cv(
        img_before,
        img_after,
        bounds,
        expected,
        meta=control,
        debug_dir=Path(debug_dir) if debug_dir else None,
    )


def _parse_bounds_box(raw: Any) -> Optional[tuple]:
    """Parse bounds string like '[l,t][r,b]' into tuple box."""
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        l, t, r, b = raw
        return (int(l), int(t), int(r), int(b))
    if not isinstance(raw, str):
        return None
    try:
        # Remove brackets and split on ][ or ,
        cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
        parts = [p for p in cleaned.split() if p]
        if len(parts) != 4:
            return None
        l, t, r, b = map(int, parts)
        return (l, t, r, b)
    except Exception:
        return None
