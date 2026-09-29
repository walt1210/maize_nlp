"""
Tests the DEPLOYED /diagnose (Cloud Run) with real leaf photos — verifying
the new prompt guardrails (FPA terminology, suspected-diagnosis framing,
fertilizer-as-support, destructive-action gating) took effect live.
"""
import base64
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("MAIZE_API_KEY")
if not API_KEY:
    raise SystemExit("MAIZE_API_KEY not found in .env")

# <-- PUT YOUR DEPLOYED CLOUD RUN URL HERE, no trailing slash
BASE_URL = "https://maize-nlp-437030900334.asia-southeast1.run.app"

CASES = [
    {
        "label": "Leaf photo 1",
        "image_path": "MLN_1726655409020.jpg",   # save the first uploaded photo here
        "classification": "MLN",       # adjust if this one's actually MSV
        "confidence": 0.75,
        "severity_pct": 40.0,
    },
    {
        "label": "Leaf photo 2",
        "image_path": "MSV_Image_6096.jpg",   # save the second uploaded photo here
        "classification": "MSV",
        "confidence": 0.80,
        "severity_pct": 15.0,
    },
]


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_case(case: dict):
    print(f"\n{'=' * 60}\nCase: {case['label']} ({case['image_path']})\n{'=' * 60}")
    try:
        original_b64 = image_to_b64(case["image_path"])
    except FileNotFoundError:
        print(f"ERROR: {case['image_path']} not found — save it in this folder first.")
        return

    body = {
        "classification": case["classification"],
        "confidence": case["confidence"],
        "severity_pct": case["severity_pct"],
        "original_image_b64": original_b64,
        "language": "english",
        "include_tagalog": False,
        "offline": False,
    }

    response = requests.post(f"{BASE_URL}/diagnose", json=body, headers={"X-API-Key": API_KEY}, timeout=90)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return

    data = response.json()
    print(f"source: {data['source']}")
    print(f"\njustification:\n{data['diagnosis']['justification']}")
    print(f"\nmanagement:\n{data['guidance']['management']}")
    print(f"\ncontrol.chemical:\n{data['guidance']['control']['chemical']}")
    print(f"\nprecautions:\n{data['guidance']['precautions']}")
    print(f"\nrag_sources: {data.get('rag_sources', [])}")


if __name__ == "__main__":
    for i, case in enumerate(CASES):
        if i > 0:
            print("\nWaiting 20s between requests...")
            time.sleep(20)
        run_case(case)