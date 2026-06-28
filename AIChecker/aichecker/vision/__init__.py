"""Vision-based UI checking utilities and orchestration."""

from .evaluator import EvaluationResult, VisionEvaluator
from .perception import ExtractedFrame, FrameExtractor

__all__ = [
    "EvaluationResult",
    "VisionEvaluator",
    "ExtractedFrame",
    "FrameExtractor",
]
