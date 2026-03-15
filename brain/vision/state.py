import json
import threading
import time
from pathlib import Path

import cv2

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / ".legion-snapshots"

_lock = threading.Lock()
_latest_frame = None
_latest_frame_bytes = b""
_latest_state = {}


def set_raw_frame(frame):
    global _latest_frame, _latest_frame_bytes
    _, jpg = cv2.imencode(".jpg", frame)
    with _lock:
        _latest_frame = frame
        _latest_frame_bytes = jpg.tobytes()


def get_frame_bytes() -> bytes:
    with _lock:
        return _latest_frame_bytes


def set_state(state: dict):
    global _latest_state
    state["timestamp"] = time.time()
    with _lock:
        _latest_state = state


def get_state() -> dict:
    with _lock:
        return _latest_state.copy()


def save_snapshot() -> Path:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
    with _lock:
        if _latest_frame is not None:
            cv2.imwrite(str(path), _latest_frame)
    return path
