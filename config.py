"""
Central configuration for the MAIze NLP pipeline.
All tunable values live here so nothing is hardcoded in the pipeline modules.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Loads .env into the process environment automatically, cross-platform.
# In production (Cloud Run), real env vars/secrets are already set and
# there's no .env file — load_dotenv() silently no-ops in that case.
load_dotenv(BASE_DIR / ".env")

# --- API keys / auth -------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # dev/eval only, never required in prod
MAIZE_API_KEY = os.environ.get("MAIZE_API_KEY")  # shared key the Android app sends

# --- Models ------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GPT_COMPARISON_MODEL = "gpt-4o-mini"  # comparison-only, never called in the deployed app
EMBEDDING_MODEL = "models/gemini-embedding-001"  # text-embedding-004 was shut down Jan 14, 2026

# --- RAG ---------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(BASE_DIR / "rag" / "chroma_db"))
KNOWLEDGE_BASE_DIR = BASE_DIR / "rag" / "knowledge_base"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3

# --- Diagnostic thresholds ----------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.6  # below this, guidance opens with an extension-officer caveat
GEMINI_TIMEOUT_SECONDS = 10  # falls back to offline guidance if exceeded

# --- XAI method (deployed heatmap) ----------------------------------------------
# evaluate_xai.py (training pipeline, separate codebase) benchmarks Grad-CAM,
# Grad-CAM++, and Score-CAM, then auto-writes the winner to XAI_DEPLOYED_METHOD
# in ITS OWN config.py. That doesn't propagate here automatically — the two
# config.py files are in separate projects and don't sync. Whoever deploys a
# new Student/XAI model must manually update XAI_METHOD below to match
# whatever evaluate_xai.py selected, or the prompt will describe the wrong
# method to Gemini.
XAI_METHOD = "gradcamplusplus"  # one of: gradcam, gradcamplusplus, scorecam
XAI_METHOD_DISPLAY_NAMES = {
    "gradcam": "Grad-CAM",
    "gradcamplusplus": "Grad-CAM++",
    "scorecam": "Score-CAM",
}
XAI_METHOD_DISPLAY_NAME = XAI_METHOD_DISPLAY_NAMES.get(XAI_METHOD, XAI_METHOD)

# --- Grading / staging ---------------------------------------------------------
GRADE_LABELS = {
    "MSV": {
        1: "Trace (\u226410% chlorotic area)",
        2: "Light (11-25% chlorotic area)",
        3: "Moderate (26-50% chlorotic area)",
        4: "Severe (51-75% chlorotic area)",
        5: "Highly susceptible (\u226575% chlorotic area)",
    },
    "MLN": {
        1: "No/trace symptoms (<10%)",
        2: "Fine mottling (10-25%)",
        3: "Mosaic throughout plant (25-50%)",
        4: "Necrosis, dead heart (50-75%)",
        5: "Dead plant (>75%)",
    },
    "HEALTHY": {0: "No symptoms"},
}

# --- Image handling ------------------------------------------------------------
MAX_IMAGE_DIMENSION = 512  # px, thumbnail max side for Gemini vision calls
MAX_IMAGE_B64_BYTES = 8 * 1024 * 1024  # 8 MB raw base64 payload cap per image

# --- Tagalog evaluation ----------------------------------------------------------
CHRF_TARGET = 0.60

# --- Flask / sessions --------------------------------------------------------------
SESSION_IDLE_TIMEOUT_MINUTES = 30

# --- Out-of-scope detection ------------------------------------------------------
OUT_OF_SCOPE_KEYWORDS = [
    "rice", "tomato", "banana", "human", "person", "price",
    "market", "weather", "soil", "fertilizer for other crops",
]

OUT_OF_SCOPE_REPLY_EN = (
    "I can only provide guidance specific to maize streak diseases (MSV and MLN). "
    "For other concerns, please consult your local DA agricultural extension officer."
)
OUT_OF_SCOPE_REPLY_TL = (
    "Makakapagbigay lamang ako ng gabay na partikular sa mga sakit ng mais na MSV at MLN. "
    "Para sa ibang alalahanin, mangyaring kumonsulta sa inyong lokal na DA agricultural "
    "extension officer."
)
