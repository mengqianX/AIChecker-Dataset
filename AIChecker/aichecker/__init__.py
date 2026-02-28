"""AI Checker package with multiple checkers (toggle, button color, count change) under one roof."""

from .models import Bounds, CheckResult, ControlInfo
from .checkers.button_color_checker import check_button_color, DEFAULT_TOLERANCE
from .checkers.toggle_checker import check_toggle
from .checkers.count_change_checker import check_count_change

__all__ = [
    "Bounds",
    "ControlInfo",
    "CheckResult",
    "check_toggle",
    "check_button_color",
    "check_count_change",
    "DEFAULT_TOLERANCE",
]
