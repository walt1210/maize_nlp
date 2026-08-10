import pytest

from pipeline.input_processor import (
    InputValidationError,
    b64_to_pil,
    grade_label,
    needs_confidence_caveat,
    prepare_images,
    severity_to_stage,
    validate_diagnostic_fields,
)
from pipeline.response_validator import ResponseValidationError, validate_gemini_output
from sample_data.mock_student_output import make_placeholder_images


def test_severity_to_stage_boundaries():
    assert severity_to_stage(0.0, "HEALTHY") == 0
    assert severity_to_stage(0.5, "MSV") == 0   # below 1.0% counts as stage 0
    assert severity_to_stage(24.9, "MSV") == 1
    assert severity_to_stage(59.9, "MSV") == 2
    assert severity_to_stage(60.0, "MSV") == 3
    assert severity_to_stage(99.0, "MLN") == 3


def test_needs_confidence_caveat():
    assert needs_confidence_caveat(0.59) is True
    assert needs_confidence_caveat(0.60) is False
    assert needs_confidence_caveat(0.95) is False


def test_grade_label_known_and_unknown():
    assert "chlorotic" in grade_label("MSV", 3).lower()
    assert grade_label("MSV", 99) == "Grade 99"  # unknown grade doesn't crash


def test_validate_diagnostic_fields_rejects_bad_classification():
    with pytest.raises(InputValidationError):
        validate_diagnostic_fields("CORN_RUST", 0.9, 20.0)


def test_validate_diagnostic_fields_rejects_bad_confidence():
    with pytest.raises(InputValidationError):
        validate_diagnostic_fields("MSV", 1.5, 20.0)


def test_prepare_images_from_placeholders():
    images = make_placeholder_images()
    original, gradcam = prepare_images(images["original_image_b64"], images["gradcam_overlay_b64"])
    assert original.size[0] <= 512 and original.size[1] <= 512
    assert gradcam.mode == "RGB"


def test_b64_to_pil_rejects_garbage():
    with pytest.raises(InputValidationError):
        b64_to_pil("not-valid-base64!!!", "test_field")


def test_b64_to_pil_rejects_empty():
    with pytest.raises(InputValidationError):
        b64_to_pil("", "test_field")


def test_validate_gemini_output_rejects_malformed_json():
    with pytest.raises(ResponseValidationError):
        validate_gemini_output("not json at all")


def test_validate_gemini_output_rejects_missing_keys():
    with pytest.raises(ResponseValidationError):
        validate_gemini_output('{"justification": "only this key present"}')


def test_validate_gemini_output_accepts_well_formed_response():
    valid_json = """
    {
      "justification": "j", "xai_explanation": "x", "key_fact": "k",
      "immediate_actions": ["a"], "management": ["m"], "spread": "s",
      "distances": "d", "prevention": ["p"], "precautions": ["pr"],
      "detection": "det",
      "control": {"chemical": "c", "biological": "b", "cultural": "cu"},
      "protocol_title": "t", "protocol_steps": ["step1"],
      "tagalog": {"justification": "j", "key_fact": "k",
                  "protocol_steps": ["s1"], "immediate_actions": ["a1"]}
    }
    """
    result = validate_gemini_output(valid_json)
    assert result.justification == "j"
