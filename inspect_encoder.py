"""
Diagnostic: prints the actual module structure of the loaded encoder so
we can find the correct XAI target layer path for mobilenet_v3_small
(a timm-based encoder via smp's "tu-" prefix, which doesn't expose
.features the way torchvision's mobilenet_v2 does).

Run directly (no Flask needed):
    python inspect_encoder.py
"""
import sys

sys.path.insert(0, ".")

import config
from pipeline.student_model.model import StudentModel

use_cbam = "cbam" in config.STUDENT_BEST_VARIANT
model = StudentModel(config.STUDENT_BEST_VARIANT, use_cbam=use_cbam)

encoder = model.unet.encoder
print(f"Encoder class: {type(encoder).__name__}")
print(f"\nTop-level attributes on encoder:")
for name, _ in encoder.named_children():
    print(f"  .{name}")

print(f"\n{'=' * 60}")
print("Full named_modules tree (last 30 entries — usually where the")
print("final conv block before pooling lives):")
print("=" * 60)
all_modules = list(encoder.named_modules())
for name, module in all_modules[-30:]:
    print(f"  {name:50s} {type(module).__name__}")

print(f"\n{'=' * 60}")
print(f"Total modules in encoder: {len(all_modules)}")
print("=" * 60)
print(
    "\nLook for the LAST Conv2d layer (or the last block containing one) "
    "in the list above — that's usually the right CAM target for "
    "classification tasks. Copy its dotted name (the first column) and "
    "send it back — I'll turn it into the correct config.XAI_TARGET_LAYERS "
    "entry."
)
