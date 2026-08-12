"""
Server-side generation of the two overlay images (segmentation boundary +
XAI attention heatmap). Exists because TFLite, used on-device for fast
classification/severity, is inference-only and can't run gradient-based
XAI methods like Grad-CAM++ (see README for the full rationale and the
architecture decision this followed).

The full PyTorch Student checkpoint is loaded ONCE at module import and
reused across requests — reloading per-request would add real latency.

IMPORTANT: this module does NOT recompute classification/confidence/
severity_pct. Those stay the on-device TFLite values (the source of
truth the farmer already saw on their phone). This module only re-runs
the model to get seg_logits and CAM activations for the two images —
its own cls_out/sev_out from this forward pass are discarded.
"""
import re

import cv2
import numpy as np
import torch
from PIL import Image

import config
from pipeline.student_model.model import StudentModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERLAY_ALPHA = 0.50  # heatmap overlay transparency, matches evaluate_xai.py

_model = None  # lazy singleton, loaded once per process
_cam_wrapper = None


class XaiEngineError(Exception):
    """Raised when the model/checkpoint can't be loaded or inference fails."""


def get_target_layer(model: StudentModel, encoder_variant: str):
    """
    Resolves the XAI target layer module from config.XAI_TARGET_LAYERS.

    IMPORTANT: walks from model.unet, NOT model directly. The training
    pipeline's evaluate_xai.py calls get_target_layer(model, variant) with
    the full StudentModel instance, but XAI_TARGET_LAYERS strings like
    "encoder.features[-1][0]" only resolve correctly if the walk starts
    at model.unet — StudentModel has no top-level .encoder attribute,
    only .unet.encoder (see pipeline/student_model/model.py). Starting
    from model.unet here avoids that AttributeError.
    """
    layer_str = config.XAI_TARGET_LAYERS.get(encoder_variant)
    if layer_str is None:
        raise XaiEngineError(f"No XAI target layer defined for: {encoder_variant}")

    obj = model.unet
    for part in layer_str.split("."):
        if "[" in part:
            attr = part[: part.index("[")]
            obj = getattr(obj, attr)
            indices = part[part.index("[") :]
            for idx_str in re.findall(r"\[(-?\d+)\]", indices):
                obj = obj[int(idx_str)]
        else:
            obj = getattr(obj, part)
    return obj


def _load_model() -> StudentModel:
    use_cbam = "cbam" in config.STUDENT_BEST_VARIANT
    model = StudentModel(config.STUDENT_BEST_VARIANT, use_cbam=use_cbam).to(DEVICE)
    try:
        ckpt = torch.load(config.STUDENT_CKPT_PATH, map_location=DEVICE, weights_only=False)
    except FileNotFoundError as exc:
        raise XaiEngineError(
            f"Student checkpoint not found at {config.STUDENT_CKPT_PATH}. "
            f"It must be bundled into the deployed image — see Dockerfile "
            f"and README for how chroma_db/ handles the same requirement."
        ) from exc
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _get_model_and_cam():
    global _model, _cam_wrapper
    if _model is None:
        _model = _load_model()
        target_layer = get_target_layer(_model, config.STUDENT_BEST_VARIANT)

        try:
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
        except ImportError as exc:
            raise XaiEngineError(
                "pytorch-grad-cam not installed. Add 'grad-cam' to requirements.txt."
            ) from exc

        cam_classes = {
            "gradcam": GradCAM,
            "gradcamplusplus": GradCAMPlusPlus,
            "scorecam": ScoreCAM,
        }
        cam_cls = cam_classes.get(config.XAI_METHOD)
        if cam_cls is None:
            raise XaiEngineError(f"Unknown XAI_METHOD in config.py: {config.XAI_METHOD}")
        _cam_wrapper = cam_cls(model=_model, target_layers=[target_layer])
    return _model, _cam_wrapper


def _preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """
    Letterbox to STUDENT_IMG_SIZE + ImageNet normalize, matching the
    training pipeline's inference transform (validate_student.py's
    _INFER_TF: LongestMaxSize + PadIfNeeded + Normalize + ToTensorV2).
    """
    size = config.STUDENT_IMG_SIZE
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (padded.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(DEVICE)


def _heatmap_overlay(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Amber/JET colormap overlay — identical to evaluate_xai.py's heatmap_overlay()."""
    h, w = img_rgb.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (img_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)
    return blended


def _seg_boundary_overlay(img_rgb: np.ndarray, seg_logits: torch.Tensor, channel: int = 1) -> np.ndarray:
    """
    Crisp green contour from the UNet segmentation head. channel=1 is the
    symptom mask (per the confirmed design: "Symptom boundary — UNet Ch1"),
    NOT channel=0 (leaf silhouette) — identical logic to
    evaluate_xai.py's seg_boundary_overlay(), just with channel=1 as the
    default instead of that script's channel=0 example call.
    """
    h, w = img_rgb.shape[:2]
    prob = torch.sigmoid(seg_logits[0, channel]).detach().cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob >= 0.5).astype(np.uint8)

    overlay = img_rgb.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 0), 2)
    return overlay


def generate_overlays(original_image: Image.Image, classification: str) -> tuple[Image.Image, Image.Image]:
    """
    Runs the full Student model on the original photo and returns
    (segmentation_overlay, xai_overlay) as PIL Images at the original
    resolution. `classification` is the on-device TFLite result — the CAM
    is targeted at that class, so the heatmap explains "why does this
    look like {classification}" using the same label the farmer already
    saw, even though this forward pass's own classification output is
    discarded.
    """
    model, cam = _get_model_and_cam()
    img_rgb = np.array(original_image.convert("RGB"))

    input_tensor = _preprocess(img_rgb)
    class_idx = config.CLASS_TO_IDX.get(classification, 0)

    with torch.no_grad():
        seg_logits, _cls_out, _sev_out = model(input_tensor)

    try:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]
    except Exception as exc:
        raise XaiEngineError(f"CAM computation failed: {exc}") from exc

    seg_overlay_np = _seg_boundary_overlay(img_rgb, seg_logits, channel=1)
    xai_overlay_np = _heatmap_overlay(img_rgb, heatmap)

    return Image.fromarray(seg_overlay_np), Image.fromarray(xai_overlay_np)
