"""
MAIze NLP Pipeline — Flask entry point.
Endpoints: POST /diagnose, POST /chat, POST /translate, GET /health,
GET /offline-check.
"""
import logging
from functools import wraps

from flask import Flask, jsonify, request

import config
from chatbot import conversation_manager
from offline.static_guidance import OFFLINE_GUIDANCE
from pipeline import input_processor, rag_engine, xai_engine
from pipeline.input_processor import InputValidationError
from pipeline.offline_fallback import get_guidance
from pipeline.tagalog_handler import translate_text
from pipeline.xai_engine import XaiEngineError

app = Flask(__name__)
logger = logging.getLogger(__name__)


def require_api_key(view_func):
    """Shared API key auth. The Cloud Run endpoint is otherwise public
    (--allow-unauthenticated), so this is the only thing standing
    between the deployed service and unlimited free Gemini usage by
    anyone with the URL."""

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not config.MAIZE_API_KEY:
            # Misconfiguration — fail closed, not open.
            return jsonify({"status": "error", "error": "server_misconfigured"}), 500
        provided = request.headers.get("X-API-Key")
        if provided != config.MAIZE_API_KEY:
            return jsonify({"status": "error", "error": "unauthorized"}), 401
        return view_func(*args, **kwargs)

    return wrapped


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": config.GEMINI_MODEL})


@app.route("/offline-check", methods=["GET"])
@require_api_key
def offline_check():
    classification = request.args.get("classification", "").upper()
    try:
        monitoring_stage = int(request.args.get("monitoring_stage", -1))
    except ValueError:
        return jsonify({"status": "error", "error": "invalid monitoring_stage"}), 400

    available = (classification, monitoring_stage) in OFFLINE_GUIDANCE
    return jsonify({"status": "success", "available": available})


@app.route("/diagnose", methods=["POST"])
@require_api_key
def diagnose():
    body = request.get_json(silent=True) or {}

    try:
        classification = body["classification"]
        confidence = float(body["confidence"])
        severity_pct = float(body["severity_pct"])
        original_image_b64 = body["original_image_b64"]
        # NOT segmentation_overlay_b64 / xai_overlay_b64 — those are
        # generated server-side now (see pipeline/xai_engine.py), since
        # TFLite (on-device) can't run gradient-based XAI methods.
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"status": "error", "error": f"missing or invalid field: {exc}"}), 400

    # Was default=True — meant any caller that omitted this field (a
    # curl test, a future integration, anything other than the current
    # Android client) silently got the expensive full bilingual
    # generation path. Android already always sends this field explicitly
    # (defaulting to false itself — see MaizeApiClient.kt), so this only
    # changes behavior for callers that don't specify it.
    include_tagalog = bool(body.get("include_tagalog", False))
    force_offline = bool(body.get("offline", False))

    try:
        input_processor.validate_diagnostic_fields(classification, confidence, severity_pct)
        original_image = input_processor.prepare_original_image(original_image_b64)
    except InputValidationError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400

    # Only generate overlays when actually calling Gemini — static offline
    # guidance is plain text lookup and never touches images (see
    # offline/static_guidance.py), so skip this entirely when force_offline
    # is set. Also means local testing with {"offline": true} works without
    # needing the real Student checkpoint bundled yet.
    segmentation_image = None
    xai_image = None
    if not force_offline:
        try:
            segmentation_image, xai_image = xai_engine.generate_overlays(original_image, classification)
        except XaiEngineError as exc:
            # Was: return 502 immediately, bypassing BOTH Gemini AND the
            # offline fallback entirely — a farmer with a perfectly valid
            # on-device classification got nothing at all if the XAI
            # checkpoint had any hiccup, even though offline guidance
            # never needed images in the first place (see
            # static_guidance.py). Degrades to offline guidance instead,
            # matching this project's stated design goal: "Never
            # crashes, never leaves the farmer with nothing."
            logger.warning("XAI generation failed, forcing offline fallback: %s", exc)
            force_offline = True

    # cimmyt_grade is NOT sent by the client — the Student model only
    # outputs a continuous severity_pct (see train_student.py /
    # export_tflite.py), never a 1-5 CIMMYT grade. Computing it here
    # server-side, from the same severity_pct the model actually
    # produces, avoids trusting the Android app to derive it correctly.
    cimmyt_grade = input_processor.severity_to_cimmyt_grade(classification, severity_pct)
    monitoring_stage = input_processor.severity_to_stage(severity_pct, classification)
    label = input_processor.grade_label(classification, cimmyt_grade)
    low_confidence = input_processor.needs_confidence_caveat(confidence)

    rag_context, rag_sources = "", []
    if not force_offline:
        try:
            rag_context, rag_sources = rag_engine.retrieve_context(classification, severity_pct, cimmyt_grade)
        except Exception as exc:
            # Was completely unhandled — a ChromaDB or embedding-API
            # failure here (rag_engine.py's own docstring already notes
            # this "can fail independently of generation") propagated as
            # a raw, unhandled 500, the same bypass-the-fallback problem
            # as the XAI case above. Degrades to offline guidance rather
            # than attempting Gemini generation with no retrieved
            # context — ungrounded output is strictly worse here (the
            # project's own RAGAS evaluation showed retrieval-grounded
            # content scores meaningfully better on faithfulness), so
            # well-tested static guidance is the safer degrade, not a
            # "try Gemini anyway" middle ground.
            logger.warning("RAG retrieval failed, forcing offline fallback: %s", exc)
            force_offline = True

    source, guidance = get_guidance(
        force_offline=force_offline,
        classification=classification,
        confidence=confidence,
        severity_pct=severity_pct,
        cimmyt_grade=cimmyt_grade,
        monitoring_stage=monitoring_stage,
        grade_label=label,
        rag_context=rag_context,
        original_image=original_image,
        segmentation_image=input_processor.resize_for_gemini(segmentation_image) if segmentation_image else None,
        xai_image=input_processor.resize_for_gemini(xai_image) if xai_image else None,
        include_tagalog=include_tagalog,
    )

    response = {
        "status": "success",
        "source": source,
        "classification": classification,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "severity_pct": severity_pct,
        "cimmyt_grade": cimmyt_grade,
        "monitoring_stage": monitoring_stage,
        "diagnosis": {
            "justification": guidance["justification"],
            # .get() with a fallback, not guidance["xai_explanation"] — the
            # 12 static offline entries never have this key, since offline
            # mode skips overlay generation entirely (nothing to explain).
            # Only Gemini's response schema guarantees this field.
            "xai_explanation": guidance.get(
                "xai_explanation",
                "Visual analysis unavailable in offline mode — showing general "
                "guidance based on classification and severity only.",
            ),
            "key_fact": guidance["key_fact"],
        },
        "guidance": {
            "immediate_actions": guidance["immediate_actions"],
            "management": guidance["management"],
            "spread": guidance["spread"],
            "distances": guidance["distances"],
            "prevention": guidance["prevention"],
            "precautions": guidance["precautions"],
            "detection": guidance["detection"],
            "control": guidance["control"],
        },
        "protocol": {
            "title": guidance["protocol_title"],
            "steps": guidance["protocol_steps"],
        },
        "rag_sources": rag_sources if source == "gemini" else [],
    }

    # Only present when overlays were actually generated (source == "gemini"
    # attempted the full path) — offline responses have no overlay images,
    # since static guidance never needed them in the first place.
    if segmentation_image is not None and xai_image is not None:
        response["overlays"] = {
            "segmentation_overlay_b64": input_processor.pil_to_b64(segmentation_image),
            "xai_overlay_b64": input_processor.pil_to_b64(xai_image),
        }

    if include_tagalog:
        # .get() rather than guidance["tagalog"]: when source == "gemini",
        # tagalog is populated whenever include_tagalog=True reached
        # gemini_engine (see prompt_builder.py). When source == "offline"
        # (Gemini failed and get_guidance fell back mid-request), this
        # assumes static_guidance.py's 12 entries always carry a "tagalog"
        # key too — true if they're built bilingual as the project notes
        # describe, but not re-verified here. .get() means a mismatch
        # degrades to tagalog: null instead of a 500 on an otherwise-
        # successful offline-fallback response.
        response["tagalog"] = guidance.get("tagalog")

    # Start a chatbot session seeded with this diagnosis for follow-up Q&A.
    # Deliberately NOT passing the raw `response` dict — it can contain
    # two full base64-encoded PNG overlay images (when source == "gemini"),
    # and conversation_manager stores this and RE-SENDS it as text on
    # EVERY single chat turn. The chat model can't interpret raw base64
    # as an image that way anyway — it's pure wasted input tokens,
    # multiplied by every message in a conversation. Only the fields the
    # chat model actually needs to continue the conversation intelligently
    # go into chat memory; images and tagalog (re-derivable on request via
    # the language toggle) are excluded.
    chat_context = {
        "classification": response["classification"],
        "confidence": response["confidence"],
        "low_confidence": response["low_confidence"],
        "severity_pct": response["severity_pct"],
        "cimmyt_grade": response["cimmyt_grade"],
        "monitoring_stage": response["monitoring_stage"],
        "diagnosis": response["diagnosis"],
        "guidance": response["guidance"],
        "protocol": response["protocol"],
    }
    session_id = conversation_manager.create_session(chat_context, language=body.get("language", "english"))
    response["session_id"] = session_id

    return jsonify(response)


@app.route("/chat", methods=["POST"])
@require_api_key
def chat():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    message = body.get("message")
    language = body.get("language", "english")

    if not session_id or not message:
        return jsonify({"status": "error", "error": "session_id and message are required"}), 400

    result = conversation_manager.handle_chat_message(session_id, message, language)
    status_code = 404 if result.get("error") == "session_not_found_or_expired" else 200
    return jsonify(result), status_code


@app.route("/translate", methods=["POST"])
@require_api_key
def translate():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    context_hint = body.get("context", "")

    if not text:
        return jsonify({"status": "error", "error": "text is required"}), 400

    try:
        tagalog = translate_text(text, context_hint)
    except Exception as exc:
        return jsonify({"status": "error", "error": f"translation failed: {exc}"}), 502

    # No chrF score here: chrF requires an expert-validated Tagalog
    # reference to compare against, which doesn't exist for arbitrary
    # on-demand text. chrF scoring only happens offline, against the
    # 30-case test set with expert references (see evaluation/chrf_eval.py).
    return jsonify({"status": "success", "tagalog": tagalog})


if __name__ == "__main__":
    # threaded=True: without this, Flask's dev server processes requests
    # ONE AT A TIME — if a second request comes in while a slow /chat or
    # /diagnose call is still in flight, it queues behind it rather than
    # running concurrently, which can make total wait time exceed even a
    # generous client-side timeout. debug=True also gives auto-reload on
    # file changes, which `flask run` (the command used previously) does
    # NOT do by default — use `python app.py` instead of `flask run` to
    # actually get both of these.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
