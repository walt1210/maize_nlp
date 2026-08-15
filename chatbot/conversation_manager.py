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
import concurrent.futures
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import google.generativeai as genai

import config
from pipeline.prompt_builder import CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

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


class ChatTruncatedError(Exception):
    """Raised when Gemini's chat reply was cut off by max_output_tokens
    instead of finishing naturally — never serve a truncated mid-sentence
    reply to the farmer as if it were complete."""


def _call_gemini_chat_sync(model, prompt: str) -> str:
    response = model.generate_content(
        prompt,
        # 800, then 1200, both still truncated mid-sentence on real
        # responses (confirmed on both English and Tagalog replies).
        # Going more generous this time rather than incrementing again —
        # cost is no longer a real constraint now that billing is enabled.
        # 2048 fixed English replies but Tagalog still truncated mid-word —
        # most tokenizers (including Gemini's) are trained predominantly
        # on English, so non-English output typically costs more tokens
        # per unit of actual content. Same budget, uneven headroom by
        # language — going higher to cover both comfortably.
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=3500),
    )

    # Unlike /diagnose, this response is never JSON-parsed, so a truncated
    # reply wouldn't fail on its own — it would just silently ship a
    # cut-off sentence to the farmer as if it were a complete, successful
    # answer. Checking finish_reason explicitly closes that gap.
    try:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason is not None and finish_reason.name == "MAX_TOKENS":
            raise ChatTruncatedError("Gemini chat reply was truncated (hit max_output_tokens)")
    except (AttributeError, IndexError):
        pass  # response shape unexpected — don't fail a diagnostic check itself

    return response.text.strip()


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
    model = genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=CHAT_SYSTEM_PROMPT)
    try:
        # Same enforced-timeout pattern as gemini_engine.py's
        # generate_guidance() — without this, a slow/hung Gemini call had
        # NO server-side deadline at all, so Flask would wait indefinitely
        # and the only thing that ever gave up was the Android client's
        # own socket timeout, surfacing as a raw exception instead of the
        # graceful canned-reply fallback below.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini_chat_sync, model, prompt)
            reply = future.result(timeout=config.GEMINI_TIMEOUT_SECONDS)
        source = "gemini"
    except Exception as exc:
        # Same fix as offline_fallback.py's get_guidance() — this except
        # was previously silent, which is exactly what caused a real
        # Gemini quota failure here to look identical (in the Flask
        # terminal) to a fully successful call, since the fallback still
        # returns 200 with a valid reply. Logging it stops that ambiguity.
        logger.warning("Gemini chat call failed, falling back to canned reply: %s", exc)
        reply = (
            "Sorry, I couldn't reach the guidance service right now. "
            "Please try again shortly, or consult your local DA extension officer."
        )
        source = "offline"

    session["history"].append({"role": "user", "content": message})
    session["history"].append({"role": "assistant", "content": reply})
    session["last_active"] = datetime.now(timezone.utc)

    return {"status": "success", "source": source, "reply": reply, "session_id": session_id}
