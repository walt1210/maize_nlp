"""
Orchestrates the "try Gemini, fall back to static guidance" flow used by
/diagnose. Falls back to offline guidance on explicit offline flag, on
Gemini timeout/API error, or on Gemini response schema validation failure —
all three treated identically so the farmer always gets actionable guidance.

IMPORTANT: the real Gemini failure reason is logged here (not just
swallowed) — the farmer-facing response never needs to show it, but
silently falling back with no trace makes debugging genuine Gemini
failures (bad API key, quota, malformed response) impossible to
distinguish from an intentional offline request.
"""
import logging

from offline.static_guidance import OFFLINE_GUIDANCE
from pipeline.gemini_engine import GeminiCallError, generate_guidance

logger = logging.getLogger(__name__)


def get_static_guidance(classification: str, monitoring_stage: int) -> dict:
    key = (classification, monitoring_stage)
    guidance = OFFLINE_GUIDANCE.get(key)
    if guidance is None:
        # Should never happen if monitoring_stage is computed correctly,
        # but never let a KeyError reach the farmer.
        guidance = OFFLINE_GUIDANCE[("HEALTHY", 0)]
    return guidance


def get_guidance(
    *,
    force_offline: bool,
    classification: str,
    confidence: float,
    severity_pct: float,
    cimmyt_grade: int,
    monitoring_stage: int,
    grade_label: str,
    rag_context: str,
    original_image,
    segmentation_image,
    xai_image,
    # Was True. app.py always passes this explicitly, so production
    # traffic was unaffected, but any other caller (tests, a future
    # script) omitting it would silently get the expensive bilingual
    # path — same issue fixed in gemini_engine.generate_guidance's
    # default. Aligned here for consistency.
    include_tagalog: bool = False,
):
    """Returns (source: 'gemini' | 'offline', guidance_dict)."""
    if force_offline:
        return "offline", get_static_guidance(classification, monitoring_stage)

    try:
        result = generate_guidance(
            classification=classification,
            confidence=confidence,
            severity_pct=severity_pct,
            cimmyt_grade=cimmyt_grade,
            monitoring_stage=monitoring_stage,
            grade_label=grade_label,
            rag_context=rag_context,
            original_image=original_image,
            segmentation_image=segmentation_image,
            xai_image=xai_image,
            include_tagalog=include_tagalog,
        )
        return "gemini", result.model_dump()
    except GeminiCallError as exc:
        logger.warning("Gemini call failed, falling back to offline guidance: %s", exc)
        return "offline", get_static_guidance(classification, monitoring_stage)
