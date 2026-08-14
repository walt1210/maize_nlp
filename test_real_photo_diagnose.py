"""
Tests /diagnose with REAL leaf photos instead of solid-color placeholders —
the actual test of whether Gemini's justification is genuinely grounded in
what it sees, versus just generating plausible text from the classification
label alone.

Run with Flask already running in another terminal (flask run --port 5000).

Usage:
    python test_real_photo_diagnose.py
"""
import base64
import json
import os
import time

import requests
from dotenv import load_dotenv

# Reads MAIZE_API_KEY from .env (already gitignored) instead of hardcoding
# it here — this file can now be safely committed to git without ever
# containing the real key in its source text.
load_dotenv()
API_KEY = os.environ.get("MAIZE_API_KEY")
if not API_KEY:
    raise SystemExit("MAIZE_API_KEY not found in .env — set it there before running this script.")

BASE_URL = "http://localhost:5000"

# Point these at real photos on your machine. classification/confidence/
# severity_pct are stand-ins for what the on-device Student model would
# normally send — we're testing the Gemini+XAI half of the pipeline here,
# not the Student model's own classification accuracy on these specific
# images, so these values are reasonable estimates, not measured output.
CASES = [
    {
        "label": "MLN photo",
        "image_path": "MLN_1726655409020.jpg",
        "classification": "MLN",
        "confidence": 0.85,
        "severity_pct": 45.0,
    },
    {
        "label": "MSV photo",
        "image_path": "MSV_Image_6096.jpg",
        "classification": "MSV",
        "confidence": 0.85,
        "severity_pct": 22.0,
    },
]


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_case(case: dict):
    print(f"\n{'=' * 60}")
    print(f"Case: {case['label']}  ({case['image_path']})")
    print("=" * 60)

    try:
        original_b64 = image_to_b64(case["image_path"])
    except FileNotFoundError:
        print(f"ERROR: {case['image_path']} not found. Save the photo in this "
              f"same folder (or update image_path to the correct location) "
              f"before running.")
        return

    body = {
        "classification": case["classification"],
        "confidence": case["confidence"],
        "severity_pct": case["severity_pct"],
        "original_image_b64": original_b64,
        "language": "english",
        "include_tagalog": True,
        "offline": False,  # real path — this is the point of this test
    }

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
    print(f"source: {data['source']}")
    if data["source"] != "gemini":
        print("^ fell back to offline — check the Flask terminal for the real error")

    print(f"\njustification:\n{data['diagnosis']['justification']}")
    print(f"\nxai_explanation:\n{data['diagnosis']['xai_explanation']}")
    print(f"\nrag_sources: {data.get('rag_sources', [])}")

    if "overlays" in data:
        seg_bytes = base64.b64decode(data["overlays"]["segmentation_overlay_b64"])
        xai_bytes = base64.b64decode(data["overlays"]["xai_overlay_b64"])
        stem = case["image_path"].rsplit(".", 1)[0]
        with open(f"debug_{stem}_segmentation.png", "wb") as f:
            f.write(seg_bytes)
        with open(f"debug_{stem}_xai.png", "wb") as f:
            f.write(xai_bytes)
        print(f"\noverlays saved: debug_{stem}_segmentation.png, debug_{stem}_xai.png")


if __name__ == "__main__":
    for i, case in enumerate(CASES):
        if i > 0:
            print("\nWaiting 20s between requests (free tier: 5 requests/minute)...")
            time.sleep(20)
        run_case(case)
