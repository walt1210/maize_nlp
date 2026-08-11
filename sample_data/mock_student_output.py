"""
Realistic mock Student model outputs for testing the NLP pipeline before
the trained vision model / Android app is ready. No xai_description field —
the pipeline only consumes classification, confidence, severity, and the
three images (see pipeline/input_processor.py and Design Decision #2).

No cimmyt_grade field either — the real Student model (train_student.py /
export_tflite.py) only ever outputs a continuous severity_pct, never a
1-5 CIMMYT grade. The backend derives the grade itself from severity_pct
(see input_processor.severity_to_cimmyt_grade), so these mocks match what
a real Android request actually contains.
"""
import base64
import io

from PIL import Image as PILImage

MOCK_OUTPUTS = {
    "msv_moderate": {
        "classification": "MSV",
        "confidence": 0.923,
        "severity_pct": 31.4,
        "original_image_b64": None,   # filled in by make_placeholder_images()
    },
    "mln_severe": {
        "classification": "MLN",
        "confidence": 0.887,
        "severity_pct": 58.2,
        "original_image_b64": None,
    },
    "healthy": {
        "classification": "HEALTHY",
        "confidence": 0.961,
        "severity_pct": 0.0,
        "original_image_b64": None,
    },
    "msv_early": {
        "classification": "MSV",
        "confidence": 0.812,
        "severity_pct": 8.3,
        "original_image_b64": None,
    },
    "msv_low_confidence": {
        "classification": "MSV",
        "confidence": 0.48,  # below LOW_CONFIDENCE_THRESHOLD (0.6) — exercises the caveat path
        "severity_pct": 22.0,
        "original_image_b64": None,
    },
}


def _solid_b64(color, size=(224, 224)) -> str:
    img = PILImage.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_placeholder_images() -> dict:
    """
    Solid-colour placeholder base64 images, standing in for real photos.
    Three DISTINCT images, matching what the real Student model produces
    (see evaluate_xai.py): original photo, segmentation-boundary contour
    (green, per CLASS_COLORS in validate_student.py), and the XAI
    attention heatmap (amber/JET colormap, per heatmap_overlay()).
    """
    return {
        "original_image_b64": _solid_b64((34, 139, 34)),        # forest green leaf
        "segmentation_overlay_b64": _solid_b64((0, 200, 0)),    # green boundary contour
        "xai_overlay_b64": _solid_b64((255, 140, 0)),           # amber heatmap
    }


def get_mock_request_body(key: str, include_tagalog: bool = True, language: str = "english") -> dict:
    """Returns a ready-to-POST /diagnose request body for the given mock case."""
    case = dict(MOCK_OUTPUTS[key])
    case.update(make_placeholder_images())
    case["language"] = language
    case["include_tagalog"] = include_tagalog
    return case
