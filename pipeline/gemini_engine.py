"""
Wraps the Gemini multimodal call: builds the prompt, sends all three
images, and validates the JSON response. Raises on any failure so the
caller (offline_fallback.get_guidance) can decide what to do next — this
module never silently falls back on its own.
"""
import concurrent.futures
import logging

from google import genai
from google.genai import types
from PIL import Image

import config
from pipeline.prompt_builder import SYSTEM_PROMPT, build_cot_prompt
from pipeline.response_validator import GeminiGuidanceResponse, validate_gemini_output

# Migrated from google.generativeai (deprecated, all support ended) to
# google.genai — same pattern as tagalog_handler.py and
# conversation_manager.py: a single reusable Client instance instead of a
# module-level configure() call.
_client = genai.Client(api_key=config.GEMINI_API_KEY)

logger = logging.getLogger(__name__)


class GeminiCallError(Exception):
    """Raised on timeout, API error, or response validation failure."""


def _log_if_truncated(response, context: str) -> None:
    """
    Checks finish_reason and logs clearly if the response was cut off by
    hitting max_output_tokens, rather than finishing naturally. This
    doesn't change behavior here — a truncated /diagnose response still
    fails JSON parsing downstream and triggers the normal offline
    fallback either way — it just replaces a generic "Unterminated
    string..." parse error with an unambiguous root cause in the logs.
    """
    try:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason is not None and finish_reason.name == "MAX_TOKENS":
            logger.warning(
                "%s: Gemini response was TRUNCATED (hit max_output_tokens) — "
                "not a natural stop. Consider raising the token budget further.",
                context,
            )
    except (AttributeError, IndexError):
        pass  # response shape unexpected — not worth failing over a diagnostic check


def _resolve_model_name(use_lite_model: bool | None) -> str:
    """
    use_lite_model=True/False is an explicit per-call override (e.g. /chat
    or an eval script choosing deliberately). None defers to
    config.GEMINI_DEV_MODE, so local/dev runs are cheap by default without
    every call site needing to know about that env var.
    """
    if use_lite_model is None:
        use_lite_model = config.GEMINI_DEV_MODE
    return config.GEMINI_MODEL_LITE if use_lite_model else config.GEMINI_MODEL


def _call_gemini_sync(
    prompt_text: str,
    original_image: Image.Image,
    segmentation_image: Image.Image,
    xai_image: Image.Image,
    model_name: str,
) -> str:
    config_kwargs = dict(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        # 2000 was too tight: the response needs justification, several
        # multi-item lists, AND (when include_tagalog=True) a full parallel
        # Tagalog translation of most of that content in the same JSON
        # object — real-world test showed it truncating mid-string before
        # finishing. 4096 still truncated mid-string on some cases. Now
        # centralized in config.GEMINI_MAX_OUTPUT_TOKENS rather than
        # hardcoded, since the English-only default path needs meaningfully
        # less budget than the bilingual path did.
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
    )
    # thinking_config is a confirmed, genuinely-supported field in
    # google.genai's GenerateContentConfig for gemini-3.5-flash (unlike
    # the old google.generativeai SDK, where support was version-dependent
    # and this needed an AttributeError guard). No guard needed here.
    # Caveat worth knowing: several open google-genai GitHub issues report
    # the backend not always strictly honoring thinking_budget even when
    # set — treat this as a best-effort cap, not a hard guarantee. If you
    # need to confirm it's actually taking effect, check
    # response.usage_metadata.thoughts_token_count against the budget.
    if config.GEMINI_THINKING_BUDGET is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=config.GEMINI_THINKING_BUDGET
        )

    response = _client.models.generate_content(
        model=model_name,
        contents=[prompt_text, original_image, segmentation_image, xai_image],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    _log_if_truncated(response, context=f"/diagnose ({model_name})")
    return response.text


def generate_guidance(
    classification: str,
    confidence: float,
    severity_pct: float,
    cimmyt_grade: int,
    monitoring_stage: int,
    grade_label: str,
    rag_context: str,
    original_image: Image.Image,
    segmentation_image: Image.Image,
    xai_image: Image.Image,
    # Was True by default, meaning every /diagnose call generated a full
    # parallel Tagalog translation regardless of whether it was wanted —
    # the exact waste the project notes describe as already fixed
    # elsewhere. It wasn't wired through; now it actually is (see
    # prompt_builder.py). Tagalog stays available on-demand via Chat's
    # language toggle, which should pass True explicitly when it needs it.
    include_tagalog: bool = False,
    # None = defer to config.GEMINI_DEV_MODE. Pass True explicitly for
    # dev/debug calls or non-final eval runs; pass False explicitly for
    # real/demo-quality /diagnose calls regardless of the env default.
    use_lite_model: bool | None = None,
) -> GeminiGuidanceResponse:
    prompt_text = build_cot_prompt(
        classification=classification,
        confidence=confidence,
        severity_pct=severity_pct,
        cimmyt_grade=cimmyt_grade,
        monitoring_stage=monitoring_stage,
        grade_label=grade_label,
        rag_context=rag_context,
        include_tagalog=include_tagalog,
    )
    model_name = _resolve_model_name(use_lite_model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _call_gemini_sync,
            prompt_text,
            original_image,
            segmentation_image,
            xai_image,
            model_name,
        )
        try:
            raw_text = future.result(timeout=config.GEMINI_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError as exc:
            raise GeminiCallError("Gemini call timed out") from exc
        except Exception as exc:  # API errors, network errors, etc.
            raise GeminiCallError(f"Gemini call failed: {exc}") from exc

    try:
        return validate_gemini_output(raw_text)
    except Exception as exc:
        raise GeminiCallError(f"Gemini response failed validation: {exc}") from exc
