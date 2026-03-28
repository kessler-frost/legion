"""One-time export of YOLOE-26x prompt-free model from .pt to .onnx format.

Run: uv run python scripts/export-yoloe-onnx.py
"""

# /// script
# dependencies = ["ultralytics"]
# ///

import json
from pathlib import Path

from ultralytics import YOLOE

MODEL_NAME = "yoloe-26x-seg-pf"
ROOT = Path(__file__).resolve().parent.parent

print(f"Loading {MODEL_NAME}.pt...")
model = YOLOE(f"{MODEL_NAME}.pt")

# Save class names
names_path = ROOT / f"{MODEL_NAME}-names.json"
with names_path.open("w") as f:
    json.dump(model.names, f)
print(f"Saved {len(model.names)} class names to {names_path}")

# Workaround: YOLOE-PF export bug (is_fused not set after fuse(None))
head = model.model.model[-1]
head.is_fused = True

print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, simplify=True)
print(f"Done: {MODEL_NAME}.onnx")
