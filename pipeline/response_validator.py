"""
Validates Gemini's JSON output against the expected schema before it's
returned to the Android app. response_mime_type="application/json"
guarantees syntactically valid JSON — it does NOT guarantee the right
keys or types are present. Any validation failure here should trigger
the same offline-fallback path as a timeout (see pipeline/offline_fallback.py).
"""
import json

from pydantic import BaseModel, Field, ValidationError


class ControlMeasures(BaseModel):
    chemical: str
    biological: str
    cultural: str


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
    tagalog: TagalogGuidance


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
