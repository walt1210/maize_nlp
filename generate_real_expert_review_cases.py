"""
Generates 30 REAL end-to-end case packets (10 per class) for the MAIze
Pathological Guidance Expert Evaluation Form, using real leaf photographs
run through the actual trained Student model, then the actual deployed
/diagnose endpoint -- no synthetic/placeholder data anywhere in this
script.

LANGUAGE CONDITIONS: within each class (MSV/MLN/HEALTHY), the first half
of cases are generated English-only (include_tagalog=False) and the
second half bilingual (include_tagalog=True, English + Filipino shown
together). "Filipino-only" is NOT offered as a separate condition: this
system's actual design always generates the base response in English,
with Filipino as an ADDITIONAL parallel block when requested (confirmed
via prompt_builder.py) -- there is no mode where only Filipino text is
produced. This split reflects the system's real, documented behavior.

WHERE TO RUN THIS:
    Run from the maize_nlp repo root (same folder as app.py, config.py).
    This script uses maize_nlp's OWN vendored StudentModel
    (pipeline/student_model/model.py) and severity-to-grade logic
    (pipeline/input_processor.py) -- no training-pipeline codebase
    required, since you don't have that installed. This is also more
    correct anyway: it guarantees the exact same model-loading code path
    as the deployed backend itself, not a parallel reimplementation.

WHAT IT DOES:
    1. Loads 10 real images per class (MSV/MLN/HEALTHY) from your image
       folder.
    2. Runs each through the REAL trained Student model to get genuine
       classification, confidence, severity_pct, and CIMMYT grade --
       not randomly-generated synthetic values.
    3. Sends each real result + the real photo to your LIVE DEPLOYED
       /diagnose endpoint (Cloud Run), exercising the exact same
       RAG + Gemini pipeline a real farmer's request would use -- half
       the cases per class as English-only, half as bilingual.
    4. Saves everything (including the image itself, base64-encoded) so
       it can be embedded directly into the expert review form.

COST NOTE: this makes 30 real, billed Gemini calls through the live
deployed service. 15 cases use the standard (single-language) generation
cost, 15 use the bilingual generation cost (roughly double output tokens
per call, per this project's own documented cost notes), so total cost
is somewhat higher than a uniform English-only batch -- expect roughly
$0.75-0.90 total based on prior full-batch cost figures.

USAGE (run from inside maize_nlp/, same folder as app.py):
    python generate_real_expert_review_cases.py \\
        --image_dir "C:\\Users\\Walter\\Downloads\\severity_rating_images\\severity_rating_images" \\
        --api_url "https://maize-nlp-437030900334.asia-southeast1.run.app" \\
        --api_key "YOUR_MAIZE_API_KEY" \\
        --n_per_class 10
"""
import argparse
import base64
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

import config
from pipeline.student_model.model import StudentModel
from pipeline.input_processor import severity_to_cimmyt_grade

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=config.STUDENT_IMG_SIZE),
    A.PadIfNeeded(config.STUDENT_IMG_SIZE, config.STUDENT_IMG_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Matches filenames like HEALTHY_HEALTHY_107.JPEG, MLN_MLN_225.JPEG,
# MSV_MSV_222.JPEG -- class name appears doubled, separated by underscore.
# Also allows an optional " (N)" suffix before the extension, seen in
# some real filenames (e.g. "HEALTHY_HEALTHY_12456 (1201).JPEG").
_FILENAME_RE = re.compile(r"^(HEALTHY|MSV|MLN)_\1_\d+(\s\(\d+\))?\.jpe?g$", re.IGNORECASE)


def load_student_model(maize_nlp_ckpt_path: str = None, encoder_name: str = None):
    # Defaults to maize_nlp's OWN resolved checkpoint path and encoder
    # variant (config.STUDENT_CKPT_PATH / config.STUDENT_BEST_VARIANT) --
    # the exact same values the deployed backend itself uses to load this
    # checkpoint. Overridable via CLI args if you want to point at a
    # different file for any reason.
    ckpt_path = Path(maize_nlp_ckpt_path or config.STUDENT_CKPT_PATH)
    encoder_name = encoder_name or config.STUDENT_BEST_VARIANT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run this script from inside the "
            f"maize_nlp repo root (same folder as app.py), or pass "
            f"--student_ckpt explicitly."
        )
    use_cbam = "cbam" in encoder_name
    model = StudentModel(encoder_name, use_cbam=use_cbam).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded Student model from: {ckpt_path}")
    return model


@torch.no_grad()
def predict(model, img_rgb: np.ndarray):
    tensor = _INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    _seg_logits, cls_out, sev_out = model(tensor)
    pred_class = config.CLASSES[cls_out.argmax(dim=1).item()]
    confidence = torch.softmax(cls_out, dim=1).max().item()
    severity = float(sev_out.item()) * 100.0
    grade = severity_to_cimmyt_grade(pred_class, severity)
    return pred_class, confidence, severity, grade


def find_images_by_class(image_dir: Path, n_per_class: int) -> dict:
    all_files = [f for f in image_dir.iterdir() if _FILENAME_RE.match(f.name)]
    by_class = {"MSV": [], "MLN": [], "HEALTHY": []}
    for f in all_files:
        cls = f.name.split("_")[0].upper()
        if cls in by_class:
            by_class[cls].append(f)
    random.seed(7)
    selected = {}
    for cls, files in by_class.items():
        if len(files) < n_per_class:
            raise ValueError(
                f"Only found {len(files)} images for class {cls}, need {n_per_class}."
            )
        selected[cls] = random.sample(files, n_per_class)
    return selected


def call_diagnose_api(api_url: str, api_key: str, image_b64: str,
                       classification: str, confidence: float, severity_pct: float,
                       include_tagalog: bool) -> dict:
    resp = requests.post(
        f"{api_url}/diagnose",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        json={
            "classification": classification,
            "confidence": confidence,
            "severity_pct": severity_pct,
            "original_image_b64": image_b64,
            "language": "english",
            "include_tagalog": include_tagalog,
            "offline": False,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--api_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument(
        "--student_ckpt", default=None,
        help="Optional override. Defaults to config.STUDENT_CKPT_PATH "
             "(maize_nlp's own resolved checkpoint path) -- run this "
             "script from inside maize_nlp/ and you shouldn't need to "
             "set this at all.",
    )
    parser.add_argument(
        "--encoder_name", default=None,
        help="Optional override. Defaults to config.STUDENT_BEST_VARIANT.",
    )
    parser.add_argument("--n_per_class", type=int, default=10)
    parser.add_argument("--out", default="real_expert_review_cases.json")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    model = load_student_model(args.student_ckpt, args.encoder_name)
    selected = find_images_by_class(image_dir, args.n_per_class)

    records = []
    case_num = 1
    for cls in ("MSV", "MLN", "HEALTHY"):
        # Balanced 2-way language split per class: first half English-only,
        # second half bilingual (include_tagalog=True). "Tagalog-only" is
        # NOT a real mode this system supports (confirmed via
        # prompt_builder.py -- the base response is always English;
        # include_tagalog only ADDS a parallel "tagalog" block, it never
        # replaces the English content), so this 2-way split reflects the
        # system's actual real behavior rather than testing a
        # non-existent third mode.
        images = selected[cls]
        half = len(images) // 2
        language_plan = (
            [False] * half + [True] * (len(images) - half)
        )

        for img_path, include_tagalog in zip(images, language_plan):
            lang_label = "Bilingual (English + Filipino)" if include_tagalog else "English only"
            print(f"  [{case_num:02d}/{args.n_per_class*3}] {img_path.name} ({lang_label}) ...")
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            pred_class, confidence, severity, grade = predict(model, img_rgb)

            with open(img_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            tagalog_guidance = None
            try:
                diagnose_result = call_diagnose_api(
                    args.api_url, args.api_key, image_b64,
                    pred_class, confidence, severity, include_tagalog
                )
                guidance = diagnose_result.get("guidance", {})
                immediate_actions = guidance.get("immediate_actions", [])
                management = guidance.get("management", [])
                rag_sources = diagnose_result.get("rag_sources", [])
                answer = " ".join(immediate_actions) + " " + " ".join(management)
                source = diagnose_result.get("source", "unknown")

                if include_tagalog:
                    tagalog = diagnose_result.get("tagalog", {})
                    tl_actions = tagalog.get("immediate_actions", [])
                    tagalog_guidance = " ".join(tl_actions) if tl_actions else None
            except Exception as exc:
                answer = f"[API CALL FAILED: {exc}]"
                rag_sources = []
                source = "error"

            records.append({
                "case_number": f"{case_num:02d}",
                "image_filename": img_path.name,
                "image_b64": image_b64,
                "classification": pred_class,
                "cimmyt_grade": grade,
                "confidence": round(confidence, 3),
                "severity_pct": round(severity, 1),
                "rag_sources": rag_sources,
                "language_condition": lang_label,
                "generated_guidance_english": answer.strip(),
                "generated_guidance_tagalog": tagalog_guidance,
                "response_source": source,
            })
            case_num += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Wrote {len(records)} real case packets to {args.out}")
    print("Upload this file back to finish formatting the expert review form.")


if __name__ == "__main__":
    main()
