"""
Tests the REAL /diagnose path — actual Gemini call + server-side XAI
generation using the checkpoint at pipeline/student_model/checkpoints/.
This will spend a small amount of Gemini quota (one call per case) and
take longer than the offline test, since it's running a full PyTorch
forward pass + CAM computation + Gemini vision call.

Run with Flask already running in another terminal (flask run --port 5000).

Usage:
    python test_real_diagnose.py
"""
import base64
import json

import requests

from sample_data.mock_student_output import get_mock_request_body

# TODO: replace with the MAIZE_API_KEY value from your .env
API_KEY = "11338b6aa6d9751bd9d54d69b78a14472ade0dad0051cfac443743945f357f27"

BASE_URL = "http://localhost:5000"


def run_case(key: str):
    body = get_mock_request_body(key, offline=False)  # real path this time
    print(f"\n{'=' * 60}")
    print(f"Case: {key}  (real Gemini + XAI path — this may take a few seconds)")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/diagnose",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        return

    data = response.json()
    print(f"source: {data['source']}")  # 'gemini' if it worked, 'offline' if it fell back
    print(f"classification: {data['classification']}")
    print(f"cimmyt_grade: {data['cimmyt_grade']}")
    print(f"low_confidence: {data['low_confidence']}")
    print(f"\njustification: {data['diagnosis']['justification']}")
    print(f"\nxai_explanation: {data['diagnosis']['xai_explanation']}")
    print(f"\nrag_sources: {data.get('rag_sources', [])}")

    if "overlays" in data:
        seg_bytes = base64.b64decode(data["overlays"]["segmentation_overlay_b64"])
        xai_bytes = base64.b64decode(data["overlays"]["xai_overlay_b64"])
        with open(f"debug_{key}_segmentation.png", "wb") as f:
            f.write(seg_bytes)
        with open(f"debug_{key}_xai.png", "wb") as f:
            f.write(xai_bytes)
        print(f"\noverlays saved: debug_{key}_segmentation.png, debug_{key}_xai.png")
        print("(open these to actually see what the model produced)")
    else:
        print("\nNo 'overlays' key in response — check if it silently fell back to offline mode")


if __name__ == "__main__":
    # Placeholder images are solid colors, so don't expect meaningful
    # symptom detection here — this run is checking that the PIPELINE
    # works end-to-end (checkpoint loads, forward pass runs, CAM
    # generates, Gemini responds with valid JSON), not diagnostic quality.
    run_case("msv_moderate")
