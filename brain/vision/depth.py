import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

_model = None


def _get_model():
    global _model
    if _model is None:
        from depth_anything_3.api import DepthAnything3
        _model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        _model = _model.to(device=torch.device("mps"))
        # Warmup
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        tmp = Path(tempfile.mktemp(suffix=".jpg"))
        cv2.imwrite(str(tmp), dummy)
        _model.inference([str(tmp)])
        tmp.unlink()
        print("Depth model ready (DA3METRIC-LARGE)")
    return _model


def get_depth_at_points(frame, points: list[tuple[int, int]]) -> dict:
    """Run DA3 on a frame, return depth at specified pixel coordinates.

    Args:
        frame: BGR numpy array from camera
        points: list of (x, y) pixel coordinates to sample depth at

    Returns:
        dict with depth_map stats and per-point depths
    """
    tmp = Path(tempfile.mktemp(suffix=".jpg"))
    cv2.imwrite(str(tmp), frame)

    model = _get_model()
    pred = model.inference([str(tmp)])
    tmp.unlink()

    depth = pred.depth[0]  # [H, W] in meters
    orig_h, orig_w = frame.shape[:2]
    dep_h, dep_w = depth.shape

    # Scale points from original frame to depth map resolution
    results = {
        "depth_range_m": [round(float(depth.min()), 3), round(float(depth.max()), 3)],
        "points": [],
    }

    for px, py in points:
        dx = int(px * dep_w / orig_w)
        dy = int(py * dep_h / orig_h)
        dx = min(max(dx, 0), dep_w - 1)
        dy = min(max(dy, 0), dep_h - 1)
        d = float(depth[dy, dx])
        results["points"].append({
            "pixel": [px, py],
            "depth_m": round(d, 3),
        })

    return results
