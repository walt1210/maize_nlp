"""
Wraps the Gemini 2.5 Flash multimodal call: builds the prompt, sends
all three images, and validates the JSON response. Raises on any failure
so the caller (offline_fallback.get_guidance) can decide what to do next —
this module never silently falls back on its own.
"""
import concurrent.futures

import google.generativeai as genai
from PIL import Image

import config
from pipeline.prompt_builder import SYSTEM_PROMPT, build_cot_prompt
from pipeline.response_validator import GeminiGuidanceResponse, validate_gemini_output

genai.configure(api_key=config.GEMINI_API_KEY)


class GeminiCallError(Exception):
    """Raised on timeout, API error, or response validation failure."""


def _call_gemini_sync(
    prompt_text: str,
    original_image: Image.Image,
    segmentation_image: Image.Image,
    xai_image: Image.Image,
) -> str:
    model = genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        [prompt_text, original_image, segmentation_image, xai_image],
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            # 2000 was too tight: the response needs justification, several
            # multi-item lists, AND a full parallel Tagalog translation of
            # most of that content in the same JSON object — real-world
            # test showed it truncating mid-string before finishing.
            # 4096 still truncated mid-string on some cases — a detailed
            # visually-grounded justification + full bilingual translation
            # of all guidance fields can genuinely need more than that.
            # Going substantially higher this time rather than incrementing
            # again after hitting the same symptom twice.
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
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
) -> GeminiGuidanceResponse:
    prompt_text = build_cot_prompt(
        classification=classification,
        confidence=confidence,
        severity_pct=severity_pct,
        cimmyt_grade=cimmyt_grade,
        monitoring_stage=monitoring_stage,
        grade_label=grade_label,
        rag_context=rag_context,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _call_gemini_sync, prompt_text, original_image, segmentation_image, xai_image
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
