import json
import time
from pathlib import Path

import cv2

STATE_FILE = Path(__file__).parent.parent.parent / ".legion-scene.json"
SNAPSHOT_DIR = Path(__file__).parent.parent.parent / ".legion-snapshots"


def write_state(state: dict):
    state["timestamp"] = time.time()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def read_state() -> dict:
    return json.loads(STATE_FILE.read_text())


def save_snapshot(frame) -> Path:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"snap_{int(time.time())}.jpg"
    cv2.imwrite(str(path), frame)
    return path
