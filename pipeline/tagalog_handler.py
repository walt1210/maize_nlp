"""
On-demand Tagalog translation (POST /translate) and chrF scoring against
an expert-validated reference, used both at eval time and for spot-checks.
"""
from google import genai
from google.genai import types
from sacrebleu.metrics import CHRF

import config

# Migrated from google.generativeai (deprecated, all support ended) to
# google.genai. The old SDK's module-level genai.configure(api_key=...)
# + genai.GenerativeModel(...) pattern is replaced by an explicit Client
# instance, reused across calls rather than constructed per-request.
_client = genai.Client(api_key=config.GEMINI_API_KEY)

_TRANSLATE_PROMPT = """\
Translate the following agricultural guidance text into natural, accurate
Tagalog. Preserve all safety-critical instructions exactly — do not
simplify or omit any warning, distance, dosage, or timing detail.
Context: {context}

TEXT TO TRANSLATE:
{text}

Respond with ONLY the Tagalog translation, no preamble, no quotation marks.
"""


def translate_text(text: str, context: str = "") -> str:
    prompt = _TRANSLATE_PROMPT.format(context=context or "General maize disease guidance", text=text)
    response = _client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1000),
    )
    return response.text.strip()


def chrf_score(hypothesis: str, reference: str) -> float:
    """Returns a chrF score in [0, 1]. Target per config.CHRF_TARGET is 0.60."""
    chrf = CHRF()
    result = chrf.sentence_score(hypothesis, [reference])
    return result.score / 100.0
