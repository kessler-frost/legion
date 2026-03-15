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


def get_state(with_objects: bool = False, with_depth: bool = False) -> dict:
    with _lock:
        state = _latest_state.copy()
        frame = _latest_raw_frame.copy() if _latest_raw_frame is not None else None

    if frame is None:
        return state

    # YOLO object detection — on demand
    if with_objects:
        try:
            from brain.vision.detector import detect_objects_on_demand
            objects = detect_objects_on_demand(frame)
            state["objects"] = objects

            # Compute distances
            import math
            distances = []
            for bot in state.get("bots", []):
                bx, by = bot["position_px"]
                for obj in objects:
                    ox, oy = obj["position_px"]
                    px_dist = math.sqrt((bx - ox) ** 2 + (by - oy) ** 2)
                    distances.append({
                        "from_bot": bot["id"],
                        "to": obj["label"],
                        "to_position": obj["position_px"],
                        "pixel_dist": round(px_dist),
                    })
            state["distances"] = distances
        except Exception as e:
            print(f"YOLO failed: {e}")

    # Depth — on demand
    if with_depth:
        try:
            from brain.vision.depth import get_depth_at_points

            points = []
            for bot in state.get("bots", []):
                points.append(tuple(bot["position_px"]))
            for obj in state.get("objects", []):
                points.append(tuple(obj["position_px"]))

            if points:
                depth_info = get_depth_at_points(frame, points)

                idx = 0
                for bot in state.get("bots", []):
                    bot["depth_m"] = depth_info["points"][idx]["depth_m"]
                    idx += 1
                for obj in state.get("objects", []):
                    obj["depth_m"] = depth_info["points"][idx]["depth_m"]
                    idx += 1

                state["depth_range_m"] = depth_info["depth_range_m"]
        except Exception as e:
            print(f"Depth failed: {e}")

    return state


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

    depth_info = None
    try:
        from brain.vision.depth import get_depth_at_points

        points = []
        h, w = frame.shape[:2]
        points.append((w // 2, h // 2))

        for bot in state.get("bots", []):
            px, py = bot["position_px"]
            points.append((px, py))

        depth_info = get_depth_at_points(frame, points)
    except Exception as e:
        print(f"Depth failed: {e}")

    return {
        "path": str(path),
        "depth": depth_info,
    }
