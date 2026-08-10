"""
In-memory chatbot session store for POST /chat follow-up Q&A.

KNOWN LIMITATION: sessions live in process memory, not a shared store
(e.g. Redis). Cloud Run gives no guarantee that two requests for the
same session_id land on the same container instance, and a container
restart drops all sessions. This deployment pins --max-instances=1 in
cloudbuild.yaml specifically to work around the first issue for a
thesis-scale demo — it does NOT protect against session loss on
restart/redeploy. Fine for a thesis demo; not appropriate at real scale.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone

import google.generativeai as genai

import config
from pipeline.prompt_builder import SYSTEM_PROMPT

genai.configure(api_key=config.GEMINI_API_KEY)

_sessions: dict[str, dict] = {}


def create_session(diagnosis_context: dict, language: str = "english") -> str:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _sessions[session_id] = {
        "session_id": session_id,
        "diagnosis_context": diagnosis_context,
        "history": [],
        "language": language,
        "created_at": now,
        "last_active": now,
    }
    return session_id


def _prune_idle_sessions():
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=config.SESSION_IDLE_TIMEOUT_MINUTES)
    idle = [sid for sid, s in _sessions.items() if s["last_active"] < cutoff]
    for sid in idle:
        del _sessions[sid]


def get_session(session_id: str) -> dict | None:
    _prune_idle_sessions()
    return _sessions.get(session_id)


def is_out_of_scope(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in config.OUT_OF_SCOPE_KEYWORDS)


def _build_chat_prompt(session: dict, user_message: str, language: str) -> str:
    context_json = json.dumps(session["diagnosis_context"], ensure_ascii=False)
    history_lines = []
    for turn in session["history"]:
        role = "Farmer" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {turn['content']}")
    history_text = "\n".join(history_lines) if history_lines else "(no prior messages)"

    language_instruction = (
        "Respond in Tagalog." if language == "tagalog" else "Respond in English."
    )

    return f"""\
DIAGNOSIS CONTEXT (from the original /diagnose call, do not contradict this):
{context_json}

CONVERSATION HISTORY:
{history_text}

FARMER'S NEW MESSAGE:
{user_message}

{language_instruction} Stay strictly within the scope of this diagnosis and
maize streak diseases (MSV, MLN). Respond with plain text only, no JSON.
"""


def handle_chat_message(session_id: str, message: str, language: str = "english") -> dict:
    session = get_session(session_id)
    if session is None:
        return {"status": "error", "error": "session_not_found_or_expired"}

    if is_out_of_scope(message):
        reply = config.OUT_OF_SCOPE_REPLY_TL if language == "tagalog" else config.OUT_OF_SCOPE_REPLY_EN
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": reply})
        session["last_active"] = datetime.now(timezone.utc)
        return {"status": "success", "source": "offline", "reply": reply, "session_id": session_id}

    prompt = _build_chat_prompt(session, message, language)
    model = genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=800),
        )
        reply = response.text.strip()
        source = "gemini"
    except Exception:
        reply = (
            "Sorry, I couldn't reach the guidance service right now. "
            "Please try again shortly, or consult your local DA extension officer."
        )
        source = "offline"

    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": reply})
    session["last_active"] = datetime.now(timezone.utc)

    return {"status": "success", "source": source, "reply": reply, "session_id": session_id}
