import json
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


def get_frame_bytes() -> bytes:
    with _lock:
        return _latest_stream_bytes


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
