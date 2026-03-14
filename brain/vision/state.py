import threading
import time
from pathlib import Path

import cv2

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / ".legion-snapshots"

_lock = threading.Lock()
_latest_frame = None
_latest_frame_bytes = b""


def set_raw_frame(frame):
    global _latest_frame, _latest_frame_bytes
    _, jpg = cv2.imencode(".jpg", frame)
    with _lock:
        _latest_frame = frame
        _latest_frame_bytes = jpg.tobytes()


def get_frame_bytes() -> bytes:
    with _lock:
        return _latest_frame_bytes


def save_snapshot() -> Path:
    """Save current frame as JPEG, return path."""
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
    with _lock:
        if _latest_frame is not None:
            cv2.imwrite(str(path), _latest_frame)
    return path
