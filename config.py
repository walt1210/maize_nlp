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
# gemini-2.5-flash is blocked for newly-created API keys as of ~Aug 2026,
# ahead of its official Oct 16, 2026 shutdown — Google's recommended
# replacement for new projects is gemini-3.5-flash.
GEMINI_MODEL = "gemini-3.5-flash"
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
GEMINI_TIMEOUT_SECONDS = 30  # 10s was too tight for 3 images + RAG context + long JSON output; falls back to offline guidance if exceeded

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

# --- Student model (server-side XAI/segmentation inference) --------------------
# TFLite (used on-device for classification/severity) is inference-only and
# can't do gradient-based XAI (Grad-CAM/Grad-CAM++ need backprop). So the
# server loads the full PyTorch Student checkpoint to generate the two
# overlay images. Classification/confidence/severity_pct are NOT recomputed
# here — the on-device values stay the source of truth (see app.py); this
# is only used for seg_logits + CAM activations.
#
# These MUST match the training pipeline's config.py values for whichever
# checkpoint is actually bundled into the Docker image — update both
# together, they're separate codebases that don't auto-sync (same caveat
# as XAI_METHOD above).
STUDENT_IMG_SIZE = 224
STUDENT_BEST_VARIANT = "mobilenet_v3_small"  # matches the checkpoint currently in use — update if you swap in a different trained variant
STUDENT_FACTORY_MODE = "mode_b"
STUDENT_DROPOUT = 0.3
CBAM_SPATIAL_KERNEL = 7
CLASSES = ["HEALTHY", "MSV", "MLN"]
CLASS_TO_IDX = {"HEALTHY": 0, "MSV": 1, "MLN": 2}

# Checkpoint must be present in the deployed image at this path — bundled
# the same way rag/chroma_db/ is (see Dockerfile), NOT downloaded at
# runtime or baked in via a build ARG (same leak risk as the Gemini key).
STUDENT_CKPT_PATH = os.environ.get(
    "STUDENT_CKPT_PATH",
    str(BASE_DIR / "pipeline" / "student_model" / "checkpoints" /
        f"student_{STUDENT_BEST_VARIANT}_{STUDENT_FACTORY_MODE}_best.pth"),
)

# Target layer for Grad-CAM/Grad-CAM++/Score-CAM, resolved against
# model.unet (NOT the top-level StudentModel — see pipeline/xai_engine.py
# for why get_target_layer() must be called on model.unet.encoder, not
# model.encoder, which doesn't exist on StudentModel directly).
XAI_TARGET_LAYERS = {
    "mobilenet_v2": "encoder.features[-1][0]",
    "mobilenet_v2_cbam": "encoder.features[-1][0]",
    # mobilenet_v3_small maps to smp's "tu-mobilenetv3_small_100" — a
    # TIMM-based encoder (via smp's "tu-" prefix), not torchvision. It has
    # no .features attribute; confirmed via direct inspection
    # (inspect_encoder.py) that the real path is encoder.model.blocks[5][0]
    # — the final ConvBnAct block (containing the last Conv2d) before
    # pooling, analogous to what "features[-1][0]" targets on torchvision's
    # mobilenet_v2.
    "mobilenet_v3_small": "encoder.model.blocks[5][0]",
    # NOT verified the same way — these were copied from the training
    # pipeline's config.py and may have the same timm-attribute-path issue
    # mobilenet_v3_small had. Run inspect_encoder.py with
    # STUDENT_BEST_VARIANT set to one of these before trusting it.
    "efficientnet_b0": "encoder.blocks[-1][-1]",
    "efficientnet_b0_cbam": "encoder.blocks[-1][-1]",
}

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

# Authoritative bracket tables, copied from the training pipeline's own
# config.py (CIMMYT_MSV_BRACKETS / CIMMYT_MLN_BRACKETS) rather than
# re-derived — these are what factory_master.py used to LABEL the
# training data, so matching them exactly (including the lower-inclusive,
# upper-exclusive boundary convention) keeps the deployed grade
# computation consistent with what the model was actually trained against.
# MSV source: Soto et al. (1982), validated by Sime et al. (2021),
#   Agriculture 11(2):130. https://doi.org/10.3390/agriculture11020130
# MLN source: Beyene et al. (2017), Euphytica 213:224.
#   https://doi.org/10.1007/s10681-017-2012-3
CIMMYT_MSV_BRACKETS = [
    (0, 10, 1),
    (10, 25, 2),
    (25, 50, 3),
    (50, 75, 4),
    (75, 101, 5),
]
CIMMYT_MLN_BRACKETS = [
    (0, 10, 1),
    (10, 25, 2),
    (25, 50, 3),
    (50, 75, 4),
    (75, 101, 5),
]

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
