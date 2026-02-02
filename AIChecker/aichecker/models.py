from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Bounds:
    """Screen bounds for the target control."""

    left: int
    top: int
    right: int
    bottom: int

    def as_box(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_sequence(cls, values: Any) -> "Bounds":
        l, t, r, b = values
        return cls(int(l), int(t), int(r), int(b))


@dataclass
class ControlInfo:
    bounds: Bounds
    text: Optional[str] = None
    semantics: Optional[str] = None
    main_color: Optional[Tuple[int, int, int]] = None
    checked: Optional[bool] = None
    source: str = "cv"  # "hypium" when using structured data
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    passed: bool
    basis: str
    control_info: ControlInfo
    details: Dict[str, Any] = field(default_factory=dict)
