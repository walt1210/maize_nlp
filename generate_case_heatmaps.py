"""
Generates original + heatmap image PAIRS for the 30 real cases already
produced by generate_real_expert_review_cases.py, using the EXACT SAME
colorization algorithm as MaizeModelRunner.kt's createSymptomHeatmap()
(sigmoid on segmentation channel 1, threshold 0.35, red-yellow blend by
symptom strength) -- so the server-generated heatmap visually matches
what the app itself would show on-device.

This does NOT make any Gemini/API calls (free, local only) -- it only
re-runs the Student model's forward pass on the same 30 images to get
the segmentation output, then colorizes it identically to the Kotlin
implementation.

WHERE TO RUN THIS: same place as generate_real_expert_review_cases.py
(maize_nlp repo root).

USAGE:
    python generate_case_heatmaps.py \\
        --image_dir "C:\\Users\\Walter\\Downloads\\severity_rating_images\\severity_rating_images" \\
        --cases_json real_expert_review_cases_no_images.json \\
        --out_dir case_images
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from pipeline.student_model.model import StudentModel

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
HEATMAP_VISIBLE_THRESHOLD = 0.35

_INFER_TF = A.Compose([
    A.LongestMaxSize(max_size=INPUT_SIZE),
    A.PadIfNeeded(INPUT_SIZE, INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Same letterbox transform WITHOUT normalization, to get the actual
# 224x224 RGB pixels the heatmap gets blended onto (matching what the
# on-device version blends onto -- the resized/padded input, not the
# original full-resolution photo).
_DISPLAY_TF = A.Compose([
    A.LongestMaxSize(max_size=INPUT_SIZE),
    A.PadIfNeeded(INPUT_SIZE, INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=0),
])


def load_student_model():
    ckpt_path = Path(config.STUDENT_CKPT_PATH)
    encoder_name = config.STUDENT_BEST_VARIANT
    use_cbam = "cbam" in encoder_name
    model = StudentModel(encoder_name, use_cbam=use_cbam).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded Student model from: {ckpt_path}")
    return model


@torch.no_grad()
def get_segmentation(model, img_rgb: np.ndarray) -> np.ndarray:
    """Returns sigmoid(channel 1) as a (224, 224) float array in [0, 1]."""
    tensor = _INFER_TF(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    seg_logits, _cls_out, _sev_out = model(tensor)
    # seg_logits shape: (1, 2, 224, 224) -- channel 1 is the symptom mask
    channel1 = seg_logits[0, 1].cpu().numpy()
    probability = 1.0 / (1.0 + np.exp(-channel1))  # sigmoid
    return probability


def create_symptom_heatmap(display_rgb: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """
    Exact port of MaizeModelRunner.kt's createSymptomHeatmap(): for each
    pixel where probability >= 0.35, blend red-yellow onto the original
    pixel by strength-derived alpha. Below threshold, pixel is untouched.
    """
    out = display_rgb.copy().astype(np.float64)
    mask = probability >= HEATMAP_VISIBLE_THRESHOLD
    strength = np.clip(
        (probability - HEATMAP_VISIBLE_THRESHOLD) / (1.0 - HEATMAP_VISIBLE_THRESHOLD),
        0.0, 1.0
    )
    heat_red = 255.0
    heat_green = 210.0 * (1.0 - strength)
    alpha = 65.0 + 150.0 * strength
    inverse = 255.0 - alpha

    r = out[:, :, 0]
    g = out[:, :, 1]
    b = out[:, :, 2]

    r_new = (r * inverse + heat_red * alpha) / 255.0
    g_new = (g * inverse + heat_green * alpha) / 255.0
    b_new = (b * inverse) / 255.0

    out[:, :, 0] = np.where(mask, r_new, r)
    out[:, :, 1] = np.where(mask, g_new, g)
    out[:, :, 2] = np.where(mask, b_new, b)

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--cases_json", required=True,
                         help="The real_expert_review_cases_no_images.json "
                              "from the earlier run, used to know which "
                              "filename maps to which case number.")
    parser.add_argument("--out_dir", default="case_images")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    with open(args.cases_json, "r", encoding="utf-8") as f:
        cases = json.load(f)

    model = load_student_model()

    for c in cases:
        case_num = c["case_number"]
        filename = c["image_filename"]
        img_path = image_dir / filename
        if not img_path.exists():
            print(f"  [{case_num}] WARNING: {filename} not found, skipping")
            continue

        print(f"  [{case_num}] {filename} ...")
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        probability = get_segmentation(model, img_rgb)
        display_rgb = _DISPLAY_TF(image=img_rgb)["image"]
        heatmap_rgb = create_symptom_heatmap(display_rgb, probability)

        orig_path = out_dir / f"case_{case_num}_original.jpg"
        heat_path = out_dir / f"case_{case_num}_heatmap.jpg"
        cv2.imwrite(str(orig_path), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heat_path), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))

    print(f"\nDone. Wrote original+heatmap pairs to {out_dir}/")


if __name__ == "__main__":
    main()
