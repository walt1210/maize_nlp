"""
Realistic mock Student model outputs for testing the NLP pipeline before
the trained vision model / Android app is ready. No xai_description field —
the pipeline only consumes classification, confidence, severity, grade,
and the two images (see pipeline/input_processor.py and Design Decision #2).
"""
import base64
import io

from PIL import Image as PILImage

MOCK_OUTPUTS = {
    "msv_moderate": {
        "classification": "MSV",
        "confidence": 0.923,
        "severity_pct": 31.4,
        "cimmyt_grade": 3,
        "original_image_b64": None,   # filled in by make_placeholder_images()
        "gradcam_overlay_b64": None,
    },
    "mln_severe": {
        "classification": "MLN",
        "confidence": 0.887,
        "severity_pct": 58.2,
        "cimmyt_grade": 4,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
    "healthy": {
        "classification": "HEALTHY",
        "confidence": 0.961,
        "severity_pct": 0.0,
        "cimmyt_grade": 0,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
    "msv_early": {
        "classification": "MSV",
        "confidence": 0.812,
        "severity_pct": 8.3,
        "cimmyt_grade": 1,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
    "msv_low_confidence": {
        "classification": "MSV",
        "confidence": 0.48,  # below LOW_CONFIDENCE_THRESHOLD (0.6) — exercises the caveat path
        "severity_pct": 22.0,
        "cimmyt_grade": 2,
        "original_image_b64": None,
        "gradcam_overlay_b64": None,
    },
}


def _solid_b64(color, size=(224, 224)) -> str:
    img = PILImage.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_placeholder_images() -> dict:
    """Solid-colour placeholder base64 images, standing in for real photos."""
    return {
        "original_image_b64": _solid_b64((34, 139, 34)),   # forest green leaf
        "gradcam_overlay_b64": _solid_b64((255, 140, 0)),  # amber heatmap
    }


def get_mock_request_body(key: str, include_tagalog: bool = True, language: str = "english") -> dict:
    """Returns a ready-to-POST /diagnose request body for the given mock case."""
    case = dict(MOCK_OUTPUTS[key])
    case.update(make_placeholder_images())
    case["language"] = language
    case["include_tagalog"] = include_tagalog
    return case
