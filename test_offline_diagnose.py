"""
Quick manual test of /diagnose in offline mode. Run with Flask already
running in another terminal (flask run --port 5000).

Usage:
    python test_offline_diagnose.py
"""
import json
import requests

from sample_data.mock_student_output import get_mock_request_body

# TODO: replace with the MAIZE_API_KEY value from your .env
API_KEY = "11338b6aa6d9751bd9d54d69b78a14472ade0dad0051cfac443743945f357f27"

BASE_URL = "http://localhost:5000"


def run_case(key: str):
    body = get_mock_request_body(key, offline=True)
    response = requests.post(
        f"{BASE_URL}/diagnose",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    print(f"\n{'=' * 60}")
    print(f"Case: {key}  |  Status: {response.status_code}")
    print("=" * 60)
    if response.status_code == 200:
        data = response.json()
        print(f"source: {data['source']}")
        print(f"classification: {data['classification']}")
        print(f"severity_pct: {data['severity_pct']}")
        print(f"cimmyt_grade: {data['cimmyt_grade']}")
        print(f"monitoring_stage: {data['monitoring_stage']}")
        print(f"low_confidence: {data['low_confidence']}")
        print(f"has 'overlays' key: {'overlays' in data}")  # should be False in offline mode
        print(f"\njustification: {data['diagnosis']['justification']}")
        print(f"\nimmediate_actions: {data['guidance']['immediate_actions']}")
    else:
        print(response.text)


if __name__ == "__main__":
    run_case("msv_moderate")       # expect cimmyt_grade == 3
    run_case("msv_low_confidence")  # expect low_confidence == True
