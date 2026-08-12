"""
MAIze NLP Pipeline — Flask entry point.
Endpoints: POST /diagnose, POST /chat, POST /translate, GET /health,
GET /offline-check.
"""
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

    include_tagalog = bool(body.get("include_tagalog", True))
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
            return jsonify({"status": "error", "error": f"XAI generation failed: {exc}"}), 502

    # cimmyt_grade is NOT sent by the client — the Student model only
    # outputs a continuous severity_pct (see train_student.py /
    # export_tflite.py), never a 1-5 CIMMYT grade. Computing it here
    # server-side, from the same severity_pct the model actually
    # produces, avoids trusting the Android app to derive it correctly.
    cimmyt_grade = input_processor.severity_to_cimmyt_grade(classification, severity_pct)
    monitoring_stage = input_processor.severity_to_stage(severity_pct, classification)
    label = input_processor.grade_label(classification, cimmyt_grade)
    low_confidence = input_processor.needs_confidence_caveat(confidence)

    rag_context, rag_sources = rag_engine.retrieve_context(classification, severity_pct, cimmyt_grade)

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
            "xai_explanation": guidance["xai_explanation"],
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
        response["tagalog"] = guidance["tagalog"]

    # Start a chatbot session seeded with this diagnosis for follow-up Q&A.
    session_id = conversation_manager.create_session(response, language=body.get("language", "english"))
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
    app.run(host="0.0.0.0", port=5000, debug=True)
