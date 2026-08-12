"""
Turns the raw /diagnose request body into everything the rest of the
pipeline needs: decoded original image, monitoring stage, grade label,
and the low-confidence flag. Classification/confidence/severity_pct come
from the on-device TFLite model (Android) and are trusted as-is — they
are NOT recomputed here. The two overlay images (segmentation boundary,
XAI heatmap) are generated server-side by pipeline/xai_engine.py, since
TFLite can't do gradient-based XAI — this module only decodes the ONE
image the client actually sends (the original photo).
"""
import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError

import config


class InputValidationError(ValueError):
    """Raised when the /diagnose request body is malformed or unsafe to process."""


def b64_to_pil(b64_str: str, field_name: str) -> Image.Image:
    if not b64_str:
        raise InputValidationError(f"{field_name} is required")
    if len(b64_str) > config.MAX_IMAGE_B64_BYTES:
        raise InputValidationError(f"{field_name} exceeds max allowed size")
    try:
        img_bytes = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputValidationError(f"{field_name} is not valid base64") from exc
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InputValidationError(f"{field_name} is not a decodable image") from exc


def pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encodes a PIL image (e.g. a server-generated overlay) back to base64
    so it can be included in the /diagnose response for Android to display."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def prepare_original_image(original_image_b64: str) -> Image.Image:
    """
    Decodes the original photo. The two overlay images are NOT decoded
    here — Android no longer sends them (see xai_engine.generate_overlays,
    which produces them server-side from this same original image).
    """
    original_pil = b64_to_pil(original_image_b64, "original_image_b64")
    max_size = config.MAX_IMAGE_DIMENSION
    thumbnail = original_pil.copy()
    thumbnail.thumbnail((max_size, max_size), Image.LANCZOS)
    return thumbnail


def resize_for_gemini(image: Image.Image) -> Image.Image:
    """Thumbnails a server-generated overlay image to the same Gemini-friendly
    size used for the original photo, without mutating the input."""
    resized = image.copy()
    resized.thumbnail((config.MAX_IMAGE_DIMENSION, config.MAX_IMAGE_DIMENSION), Image.LANCZOS)
    return resized


def severity_to_stage(severity_pct: float, classification: str) -> int:
    if classification == "HEALTHY" or severity_pct < 1.0:
        return 0  # No infection — routine monitoring
    elif severity_pct < 25.0:
        return 1  # Mild — monitor weekly, scout for vectors
    elif severity_pct < 60.0:
        return 2  # Moderate — apply management protocol immediately
    else:
        return 3  # Severe — immediate intervention, consider rouging


def _grade_from_brackets(severity_pct: float, brackets: list[tuple[int, int, int]]) -> int:
    for lo, hi, grade in brackets:
        if lo <= severity_pct < hi:
            return grade
    return brackets[-1][2]  # severity_pct >= 100 falls through to the top bracket


def severity_to_cimmyt_grade(classification: str, severity_pct: float) -> int:
    """
    Derives the 1-5 CIMMYT grade from severity_pct, using the EXACT bracket
    tables the training pipeline used to label training data
    (config.CIMMYT_MSV_BRACKETS / CIMMYT_MLN_BRACKETS — lower-inclusive,
    upper-exclusive, e.g. severity==10.0 falls into the (10,25,2) bracket,
    not (0,10,1)). The Student model only ever outputs continuous
    severity_pct, never a grade directly, so this MUST be computed
    server-side — matching the training-time boundary convention exactly
    keeps deployed grades consistent with what the model was trained against.
    """
    if classification == "HEALTHY":
        return 0
    elif classification == "MSV":
        return _grade_from_brackets(severity_pct, config.CIMMYT_MSV_BRACKETS)
    elif classification == "MLN":
        return _grade_from_brackets(severity_pct, config.CIMMYT_MLN_BRACKETS)
    else:
        return 0


def grade_label(classification: str, cimmyt_grade: int) -> str:
    labels = config.GRADE_LABELS.get(classification, {})
    return labels.get(cimmyt_grade, f"Grade {cimmyt_grade}")


def needs_confidence_caveat(confidence: float) -> bool:
    return confidence < config.LOW_CONFIDENCE_THRESHOLD


def validate_diagnostic_fields(classification: str, confidence: float, severity_pct: float):
    if classification not in ("HEALTHY", "MSV", "MLN"):
        raise InputValidationError(f"Unknown classification: {classification}")
    if not (0.0 <= confidence <= 1.0):
        raise InputValidationError("confidence must be between 0.0 and 1.0")
    if not (0.0 <= severity_pct <= 100.0):
        raise InputValidationError("severity_pct must be between 0.0 and 100.0")
