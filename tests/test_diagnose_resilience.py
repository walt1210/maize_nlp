"""
Tests that /diagnose degrades to offline guidance instead of hard-failing
when XAI overlay generation or RAG retrieval fails. Both used to bypass
the offline fallback entirely — XAI failure returned a hard 502, RAG
failure was completely unhandled (an unhandled 500) — even though
offline guidance never needed images or RAG context in the first place.
See app.py's inline comments on the /diagnose route for the full
reasoning behind the fix these tests verify.

Patches are applied at "app.xai_engine.generate_overlays" /
"app.rag_engine.retrieve_context" (where app.py imported the names into
its own namespace), NOT "pipeline.xai_engine.generate_overlays" — patching
the original module wouldn't affect the reference app.py already holds.
"""
from unittest.mock import patch

import pytest
from PIL import Image

import config
from app import app
from pipeline.xai_engine import XaiEngineError
from sample_data.mock_student_output import make_placeholder_images


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def _diagnose_body() -> dict:
    images = make_placeholder_images()
    return {
        "classification": "MSV",
        "confidence": 0.85,
        "severity_pct": 22.0,
        "original_image_b64": images["original_image_b64"],
    }


def test_xai_failure_degrades_to_offline(client):
    # xai_engine.generate_overlays is mocked to fail — never reaches the
    # real Student checkpoint, so this test has no external dependency
    # on that file being present, same as the rest of the pytest suite.
    with patch(
        "app.xai_engine.generate_overlays",
        side_effect=XaiEngineError("mocked XAI failure"),
    ):
        response = client.post(
            "/diagnose",
            json=_diagnose_body(),
            headers={"X-API-Key": config.MAIZE_API_KEY},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["source"] == "offline"
    # Confirms this is genuinely the offline path completing successfully,
    # not some other error response that happens to also return 200.
    assert "diagnosis" in data
    assert "protocol" in data


def test_rag_failure_degrades_to_offline(client):
    # XAI generation is mocked to SUCCEED here (not the thing under test
    # in this case) — deliberately using a fake placeholder image rather
    # than letting the real XAI engine run, so this test also has no
    # dependency on the real Student checkpoint being present.
    fake_overlay = Image.new("RGB", (10, 10))

    with patch(
        "app.xai_engine.generate_overlays",
        return_value=(fake_overlay, fake_overlay),
    ), patch(
        "app.rag_engine.retrieve_context",
        side_effect=RuntimeError("mocked RAG failure — e.g. ChromaDB/embedding API down"),
    ):
        response = client.post(
            "/diagnose",
            json=_diagnose_body(),
            headers={"X-API-Key": config.MAIZE_API_KEY},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["source"] == "offline"
    assert "diagnosis" in data
    assert "protocol" in data
