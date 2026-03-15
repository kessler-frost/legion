import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import time
from pathlib import Path

import cv2

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / ".legion-snapshots"

_lock = threading.Lock()
_latest_raw_frame = None
_latest_stream_bytes = b""
_latest_state = {}


def set_frames(raw_frame, annotated_frame):
    global _latest_raw_frame, _latest_stream_bytes
    _, jpg = cv2.imencode(".jpg", annotated_frame)
    with _lock:
        _latest_raw_frame = raw_frame
        _latest_stream_bytes = jpg.tobytes()


def get_frame_bytes(annotated: bool = True) -> bytes:
    with _lock:
        if annotated:
            return _latest_stream_bytes
        if _latest_raw_frame is not None:
            _, jpg = cv2.imencode(".jpg", _latest_raw_frame)
            return jpg.tobytes()
        return b""


def set_state(state: dict):
    global _latest_state
    state["timestamp"] = time.time()
    with _lock:
        _latest_state = state


def get_state() -> dict:
    with _lock:
        return _latest_state.copy()


def save_snapshot() -> Path:
    """Save RAW frame (no ArUco overlay) for CC to analyze."""
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
    with _lock:
        if _latest_raw_frame is not None:
            cv2.imwrite(str(path), _latest_raw_frame)
    return path


def get_snapshot_with_depth() -> dict:
    """Save snapshot + run DA3 depth on it. Returns path + depth info."""
    path = save_snapshot()

    with _lock:
        frame = _latest_raw_frame.copy() if _latest_raw_frame is not None else None
        state = _latest_state.copy()

    if frame is None:
        return {"path": str(path), "depth": None}

    from brain.vision.depth import get_depth_at_points

    # Collect points to sample: all bot positions + frame center
    points = []
    h, w = frame.shape[:2]
    points.append((w // 2, h // 2))  # center

    for bot in state.get("bots", []):
        px, py = bot["position_px"]
        points.append((px, py))

    depth_info = get_depth_at_points(frame, points)

    return {
        "path": str(path),
        "depth": depth_info,
    }
