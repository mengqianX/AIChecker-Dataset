"""VLM JSON 兜底包装器：用确定性规则校验并重装模型输出，不二次调用模型。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*\r?\n?(.*?)\r?\n?```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


@dataclass(frozen=True)
class JsonRecoveryResult:
    """包装器恢复结果。"""

    payload: dict[str, Any]
    repaired: bool
    method: str


def recover_vlm_json(
    raw_text: str,
    required_fields: dict[str, type | tuple[type, ...]] | None = None,
) -> JsonRecoveryResult:
    """
    校验模型输出是否为合法 JSON 对象；不合格则抽取内容并重装。

    顺序：原文 → markdown 围栏内文本 → 第一个完整 {...} → 去尾逗号后再解析。
    解析成功后按 required_fields 做类型兜底（如 "true" → bool），不补造缺失字段。
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("模型输出为空，无法解析 JSON")

    last_error: Exception | None = None
    for method, candidate in _candidate_payloads(text):
        parsed, parse_error = _loads_object(candidate)
        if parsed is None:
            last_error = parse_error
            continue
        coerced = _coerce_required_fields(parsed, required_fields)
        method_name = method
        if method == "raw" and coerced is not parsed:
            method_name = "type_coerce"
        repaired = method_name != "raw"
        return JsonRecoveryResult(payload=coerced, repaired=repaired, method=method_name)

    raise ValueError(f"模型输出不是合法 JSON: {raw_text}") from last_error


def _candidate_payloads(text: str) -> list[tuple[str, str]]:
    ordered: list[tuple[str, str]] = [("raw", text)]
    fenced = _FENCE_RE.search(text)
    if fenced:
        inner = fenced.group(1).strip()
        if inner:
            ordered.append(("fence", inner))
    extracted = _extract_balanced_object(text)
    if extracted:
        ordered.append(("object", extracted))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for method, candidate in ordered:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append((method, candidate))
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if repaired != candidate and repaired not in seen:
            seen.add(repaired)
            unique.append((f"{method}_trailing_comma", repaired))
    return unique


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _loads_object(candidate: str) -> tuple[dict[str, Any] | None, Exception | None]:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, exc
    if not isinstance(parsed, dict):
        return None, ValueError(f"JSON 根节点不是对象: {type(parsed).__name__}")
    return parsed, None


def _coerce_required_fields(
    payload: dict[str, Any],
    required_fields: dict[str, type | tuple[type, ...]] | None,
) -> dict[str, Any]:
    if not required_fields:
        return payload
    coerced = dict(payload)
    changed = False
    for field, expected_type in required_fields.items():
        if field not in coerced:
            raise ValueError(f"模型输出缺少必需字段: {field} | {payload}")
        expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        value = coerced[field]
        if isinstance(value, expected_types):
            continue
        coerced[field] = _coerce_value(value, expected_types, field)
        changed = True
    return coerced if changed else payload


def _coerce_value(value: Any, expected_types: tuple[type, ...], field: str) -> Any:
    if bool in expected_types:
        coerced_bool = _as_bool(value)
        if coerced_bool is not None:
            return coerced_bool
    if type(None) in expected_types and _is_nullish(value):
        return None
    if str in expected_types and value is not None:
        return str(value)
    raise ValueError(
        f"字段类型不匹配且无法兜底: {field} 期望 {expected_types} 实际 {type(value)}"
    )


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return True
    return False
