import json
import threading
import time
from pathlib import Path

import cv2

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / ".legion-snapshots"

_lock = threading.Lock()
_latest_raw_bytes = b""
_latest_annotated_bytes = b""
_latest_state = {}


def set_state(state: dict):
    global _latest_state
    state["timestamp"] = time.time()
    with _lock:
        _latest_state = state


def read_state() -> dict:
    with _lock:
        return _latest_state.copy()


def set_raw_frame(frame):
    global _latest_raw_bytes
    _, raw_jpg = cv2.imencode(".jpg", frame)
    with _lock:
        _latest_raw_bytes = raw_jpg.tobytes()


def set_frame(raw_frame, annotated_frame):
    global _latest_raw_bytes, _latest_annotated_bytes
    _, raw_jpg = cv2.imencode(".jpg", raw_frame)
    with _lock:
        _latest_raw_bytes = raw_jpg.tobytes()
        if annotated_frame is not None:
            _, ann_jpg = cv2.imencode(".jpg", annotated_frame)
            _latest_annotated_bytes = ann_jpg.tobytes()


def get_frame(annotated: bool = False) -> bytes:
    with _lock:
        return _latest_annotated_bytes if annotated else _latest_raw_bytes


def save_snapshot(frame) -> Path:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
    cv2.imwrite(str(path), frame)
    return path
