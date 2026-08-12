"""
Vendored copy of the StudentModel architecture (CBAM modules + StudentModel
class) from the training pipeline's train_student.py, trimmed to
INFERENCE ONLY — no datasets, losses, or training loop, since the Flask
backend only ever runs a forward pass to get seg_logits + CAM activations
for the two overlay images. Classification/severity from THIS model's own
forward pass are discarded; the on-device TFLite values remain the source
of truth (see pipeline/xai_engine.py).

Keep this in sync with the training pipeline's train_student.py — these
are separate codebases that don't auto-sync. If the architecture changes
there (new encoder variant, different head structure), this file needs
the same change or checkpoint loading will fail with a state_dict
mismatch.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config

# smp's classic encoder dict doesn't include "mobilenet_v3_small" and
# uses a hyphen for efficientnet, not the underscore used in our variant
# names — same mapping as the training pipeline's train_student.py.
_SMP_ENCODER_NAMES = {
    "mobilenet_v2": "mobilenet_v2",
    "mobilenet_v2_cbam": "mobilenet_v2",
    "mobilenet_v3_small": "tu-mobilenetv3_small_100",
    "efficientnet_b0": "efficientnet-b0",
    "efficientnet_b0_cbam": "efficientnet-b0",
}


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap_out = self.mlp(self.gap(x))
        gmp_out = self.mlp(self.gmp(x))
        scale = self.sigmoid(gap_out + gmp_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = None):
        super().__init__()
        kernel_size = kernel_size or config.CBAM_SPATIAL_KERNEL
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(cat))
        return x * scale


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class StudentModel(nn.Module):
    """
    Multi-task Student model:
      - Shared encoder (configurable) via self.unet.encoder
      - UNet decoder → leaf silhouette + symptom mask (2-channel seg output)
      - Classification head → HEALTHY/MSV/MLN
      - Severity head → continuous 0-1 (x100 at inference)

    NOTE: the encoder is at self.unet.encoder, NOT self.encoder directly —
    XAI target-layer resolution must walk from model.unet, not model
    itself (see pipeline/xai_engine.py get_target_layer()).
    """

    def __init__(self, encoder_name: str, use_cbam: bool = False):
        super().__init__()
        self.use_cbam = use_cbam
        self.encoder_name = encoder_name

        smp_encoder = _SMP_ENCODER_NAMES.get(
            encoder_name, encoder_name.replace("_cbam", "")
        )

        self.unet = smp.Unet(
            encoder_name=smp_encoder,
            encoder_weights=None,  # loading a trained checkpoint, not ImageNet init
            in_channels=3,
            classes=2,  # Ch0: leaf silhouette | Ch1: symptom mask
            activation=None,
        )

        if use_cbam:
            enc_channels = self.unet.encoder.out_channels
            skip_channels = [c for c in enc_channels[1:-1] if c > 0]
            self.cbam_blocks = nn.ModuleList([
                CBAMBlock(c) for c in skip_channels
            ])

        enc_out_ch = self.unet.encoder.out_channels[-1]

        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.STUDENT_DROPOUT),
            nn.Linear(enc_out_ch, len(config.CLASSES)),
        )

        # Severity head — ReLU + clamp, NOT Sigmoid (avoids ceiling issue)
        self.sev_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.STUDENT_DROPOUT),
            nn.Linear(enc_out_ch, 1),
            nn.ReLU(),
        )

    def _apply_cbam_to_features(self, features: list) -> list:
        if not self.use_cbam:
            return features
        result = [features[0]]
        for i, feat in enumerate(features[1:]):
            if feat is not None and i < len(self.cbam_blocks):
                result.append(self.cbam_blocks[i](feat))
            else:
                result.append(feat)
        return result

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.unet.encoder(x)
        deep_feat = features[-1]

        features_cbam = self._apply_cbam_to_features(features)

        decoder_out = self.unet.decoder(features_cbam)
        seg_logits = self.unet.segmentation_head(decoder_out)

        pooled = self.cls_gap(deep_feat)
        cls_out = self.cls_head(pooled)

        sev_out = self.sev_head(pooled).squeeze(1).clamp(0.0, 1.0)

        return seg_logits, cls_out, sev_out
