"""
Validates Gemini's JSON output against the expected schema before it's
returned to the Android app. response_mime_type="application/json"
guarantees syntactically valid JSON — it does NOT guarantee the right
keys or types are present. Any validation failure here should trigger
the same offline-fallback path as a timeout (see pipeline/offline_fallback.py).
"""
import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional

# Defense-in-depth safety net: prompt_builder.py's SYSTEM_PROMPT and schema
# instructions tell Gemini never to state a specific dosage/application
# rate, but instruction-following isn't 100% reliable (confirmed
# separately -- Gemini has been observed not fully following an explicit
# English-only language instruction in some calls). This regex catches
# the pattern even if the prompt-level instruction is ignored: a number
# followed immediately by a dosage-style unit (g, mL, L, kg, %) in close
# proximity to another such number is a strong signal of a specific
# mixing ratio/application rate, which this system must never surface
# per its documented scope (educational/field-support only, not a
# regulated-intervention prescriber).
_DOSAGE_PATTERN = re.compile(
    r"\d+\s?(g|ml|mL|L|kg)\b.{0,30}\d+\s?(g|ml|mL|L|kg)\b",
    re.IGNORECASE,
)


class ControlMeasures(BaseModel):
    chemical: str
    biological: str
    cultural: str

    @field_validator("chemical")
    @classmethod
    def no_specific_dosage(cls, v: str) -> str:
        if _DOSAGE_PATTERN.search(v):
            raise ValueError(
                "chemical control field appears to contain a specific "
                "dosage/mixing ratio, which this system must not surface "
                "(educational/field-support scope only, not a regulated "
                "intervention prescriber) -- rejecting response"
            )
        return v


class TagalogGuidance(BaseModel):
    justification: str
    key_fact: str
    protocol_steps: list[str] = Field(min_length=1)
    immediate_actions: list[str] = Field(min_length=1)


class GeminiGuidanceResponse(BaseModel):
    justification: str
    xai_explanation: str
    key_fact: str
    immediate_actions: list[str] = Field(min_length=1)
    management: list[str] = Field(min_length=1)
    spread: str
    distances: str
    prevention: list[str] = Field(min_length=1)
    precautions: list[str] = Field(min_length=1)
    detection: str
    control: ControlMeasures
    protocol_title: str
    protocol_steps: list[str] = Field(min_length=1)
    # Optional, not required: gemini_engine.generate_guidance defaults to
    # include_tagalog=False (English-only /diagnose calls, the common
    # case), and prompt_builder now genuinely omits the "tagalog" key from
    # the JSON schema it asks Gemini for in that case — it's not just
    # missing by accident. Requiring this field would fail validation on
    # every English-only call and silently trigger the offline fallback,
    # which would have completely masked the Tagalog-generation cost fix.
    tagalog: Optional[TagalogGuidance] = None


class ResponseValidationError(ValueError):
    """Raised when Gemini's output doesn't match GeminiGuidanceResponse."""


def validate_gemini_output(raw_text: str) -> GeminiGuidanceResponse:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"Gemini output was not valid JSON: {exc}") from exc

    try:
        return GeminiGuidanceResponse.model_validate(data)
    except ValidationError as exc:
        raise ResponseValidationError(f"Gemini output failed schema validation: {exc}") from exc
