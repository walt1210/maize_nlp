"""
Manual smoke test for /chat and /translate — NOT part of the pytest suite.
Run against a locally running `python app.py` to confirm
conversation_manager.py and tagalog_handler.py's google.genai migration
actually works end to end, since nothing else exercises either of them.

Usage:
    python test_chat_translate_manual.py
"""
import requests

import config
from sample_data.mock_student_output import make_placeholder_images

BASE_URL = "http://localhost:5000"
HEADERS = {"X-API-Key": config.MAIZE_API_KEY, "Content-Type": "application/json"}


def _get_session_id() -> str:
    """Runs one /diagnose call just to get a real session_id to chat against."""
    images = make_placeholder_images()
    body = {
        "classification": "MSV",
        "confidence": 0.85,
        "severity_pct": 22.0,
        "original_image_b64": images["original_image_b64"],
    }
    resp = requests.post(f"{BASE_URL}/diagnose", headers=HEADERS, json=body, timeout=90)
    resp.raise_for_status()
    return resp.json()["session_id"]


def test_chat(session_id: str) -> None:
    print(f"\n{'=' * 60}\nTEST: /chat\n{'=' * 60}")
    body = {
        "session_id": session_id,
        "message": "How far apart should I plant next season to avoid this?",
        "language": "english",
    }
    resp = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=body, timeout=90)
    print(f"HTTP {resp.status_code}")
    data = resp.json()
    print(f"source: {data.get('source')}")
    print(f"reply: {data.get('reply', '')[:200]}...")
    print(f"error: {data.get('error')}")


def test_chat_tagalog(session_id: str) -> None:
    print(f"\n{'=' * 60}\nTEST: /chat (Tagalog toggle)\n{'=' * 60}")
    body = {
        "session_id": session_id,
        "message": "Ano ang dapat kong gawin ngayon?",
        "language": "tagalog",
    }
    resp = requests.post(f"{BASE_URL}/chat", headers=HEADERS, json=body, timeout=90)
    print(f"HTTP {resp.status_code}")
    data = resp.json()
    print(f"source: {data.get('source')}")
    print(f"reply: {data.get('reply', '')[:200]}...")


def test_translate() -> None:
    print(f"\n{'=' * 60}\nTEST: /translate\n{'=' * 60}")
    body = {
        "text": "Isolate the affected plant and monitor daily for spread.",
        "context": "MSV management guidance",
    }
    resp = requests.post(f"{BASE_URL}/translate", headers=HEADERS, json=body, timeout=90)
    print(f"HTTP {resp.status_code}")
    data = resp.json()
    print(f"status: {data.get('status')}")
    print(f"tagalog: {data.get('tagalog')}")


def main():
    if not config.MAIZE_API_KEY:
        print("MAIZE_API_KEY is not set — check your .env before running this.")
        return

    print("Getting a session_id from /diagnose first...")
    session_id = _get_session_id()
    print(f"session_id: {session_id}")

    test_chat(session_id)
    test_chat_tagalog(session_id)
    test_translate()

    print(f"\n{'=' * 60}")
    print("Expected results:")
    print("  /chat            -> HTTP 200, source: gemini, reply is a real English sentence")
    print("  /chat (tagalog)  -> HTTP 200, source: gemini, reply is in Tagalog")
    print("  /translate       -> HTTP 200, status: success, tagalog is a real translation")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
