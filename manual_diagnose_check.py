"""
Manual smoke test for /diagnose — NOT part of the pytest suite, run this
by hand against a locally running `python app.py` to sanity-check the
three response paths (default English-only, explicit bilingual, offline
fallback) after the cost-optimization changes.

Usage:
    python test_diagnose_manual.py

Uses sample_data.mock_student_output.make_placeholder_images() — the same
placeholder your pytest suite and ragas_eval.py already rely on — so this
runs without needing a real leaf photo on disk. It verifies the PIPELINE
MECHANICS (model routing, tagalog toggle, offline fallback all wired
correctly), not output QUALITY — the placeholder image won't produce a
meaningful diagnosis, since it's not a real leaf.
"""
import json

import requests

import config
from sample_data.mock_student_output import make_placeholder_images

BASE_URL = "http://localhost:5000"
HEADERS = {"X-API-Key": config.MAIZE_API_KEY, "Content-Type": "application/json"}


def _post_diagnose(label: str, extra_fields: dict) -> None:
    images = make_placeholder_images()
    body = {
        "classification": "MSV",
        "confidence": 0.85,
        "severity_pct": 22.0,
        "original_image_b64": images["original_image_b64"],
        **extra_fields,
    }

    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    resp = requests.post(f"{BASE_URL}/diagnose", headers=HEADERS, json=body, timeout=90)
    print(f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        print("Response was not valid JSON:")
        print(resp.text[:500])
        return

    if resp.status_code != 200:
        print("Error response:", json.dumps(data, indent=2))
        return

    print(f"source: {data.get('source')}")
    print(f"has 'tagalog' key: {'tagalog' in data}")
    print(f"justification: {data.get('diagnosis', {}).get('justification', '')[:120]}...")
    print(f"session_id: {data.get('session_id')}")


def main():
    if not config.MAIZE_API_KEY:
        print("MAIZE_API_KEY is not set — check your .env before running this.")
        return

    _post_diagnose(
        "TEST 1: Default path (English-only, dev model if GEMINI_DEV_MODE=1)",
        extra_fields={},
    )
    _post_diagnose(
        "TEST 2: Explicit bilingual path",
        extra_fields={"include_tagalog": True},
    )
    _post_diagnose(
        "TEST 3: Offline fallback path",
        extra_fields={"offline": True},
    )

    print(f"\n{'=' * 60}")
    print("Expected results:")
    print("  Test 1 -> source: gemini, has 'tagalog' key: False")
    print("  Test 2 -> source: gemini, has 'tagalog' key: True")
    print("  Test 3 -> source: offline, has 'tagalog' key: True")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
