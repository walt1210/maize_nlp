"""
Turns the raw /diagnose request body into everything the rest of the
pipeline needs: decoded images, monitoring stage, grade label, and the
low-confidence flag. The Student model itself never runs here — it runs
on-device (Android, TFLite); this module only processes its outputs.
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


def prepare_images(original_image_b64: str, segmentation_overlay_b64: str, xai_overlay_b64: str):
    """
    Decode all three images and resize to Gemini-friendly dimensions.
    Three DISTINCT images per the Student model's actual outputs
    (see evaluate_xai.py): the original photo, the crisp green
    segmentation-boundary contour (UNet head — pixel-level localization),
    and the XAI attention heatmap (whichever method config.XAI_METHOD
    names — class-discriminative explanation, not the same thing as the
    segmentation boundary).
    """
    original_pil = b64_to_pil(original_image_b64, "original_image_b64")
    segmentation_pil = b64_to_pil(segmentation_overlay_b64, "segmentation_overlay_b64")
    xai_pil = b64_to_pil(xai_overlay_b64, "xai_overlay_b64")

    max_size = config.MAX_IMAGE_DIMENSION
    original_pil.thumbnail((max_size, max_size), Image.LANCZOS)
    segmentation_pil.thumbnail((max_size, max_size), Image.LANCZOS)
    xai_pil.thumbnail((max_size, max_size), Image.LANCZOS)

    return original_pil, segmentation_pil, xai_pil


def severity_to_stage(severity_pct: float, classification: str) -> int:
    if classification == "HEALTHY" or severity_pct < 1.0:
        return 0  # No infection — routine monitoring
    elif severity_pct < 25.0:
        return 1  # Mild — monitor weekly, scout for vectors
    elif severity_pct < 60.0:
        return 2  # Moderate — apply management protocol immediately
    else:
        return 3  # Severe — immediate intervention, consider rouging


def severity_to_cimmyt_grade(classification: str, severity_pct: float) -> int:
    """
    Derives the 1-5 CIMMYT grade from severity_pct. The Student model
    (see train_student.py / export_tflite.py) only ever outputs a
    continuous severity_pct — it never produces a CIMMYT grade directly,
    and no other part of the training/export pipeline computes one either.
    So this MUST be computed server-side rather than trusted from the
    client; there's no upstream source of truth for it otherwise.
    Boundaries match config.GRADE_LABELS' documented ranges.
    """
    if classification == "HEALTHY":
        return 0
    if severity_pct <= 10.0:
        return 1
    elif severity_pct <= 25.0:
        return 2
    elif severity_pct <= 50.0:
        return 3
    elif severity_pct <= 75.0:
        return 4
    else:
        return 5


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
