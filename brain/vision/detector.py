import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
from onnxruntime import YOLO

from brain.vision.state import set_state, set_frame, set_raw_frame

DEFAULT_SOURCE = 0
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DEPTH_MODEL_PATH = MODELS_DIR / "DepthAnythingV2SmallF16.mlpackage"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

CAMERA_MATRIX = np.array([
    [1400, 0, 960],
    [0, 1400, 540],
    [0, 0, 1],
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)
MARKER_SIZE = 0.04

# Labels we track
BOT_LABELS = {"car", "truck", "motorcycle", "bus"}
BALL_LABELS = {"sports ball", "frisbee", "clock"}
TRACKED_LABELS = BOT_LABELS | BALL_LABELS

# Persistent object registry
OBJECT_TIMEOUT = 5.0  # seconds before untracked object expires

_stop_event = threading.Event()
_object_registry: dict[int, dict] = {}  # reg_id → object info
_next_reg_id = 1


def request_stop():
    _stop_event.set()


def _match_to_registry(detected_objects):
    """Match detected objects to registry by position proximity. Register new ones."""
    global _next_reg_id
    now = time.time()
    matched_reg_ids = set()

    for obj in detected_objects:
        ox, oy = obj["position_px"]
        best_id = None
        best_dist = 100  # max pixel distance to match

        for reg_id, reg in _object_registry.items():
            if reg_id in matched_reg_ids:
                continue
            rx, ry = reg["position_px"]
            dist = math.sqrt((ox - rx) ** 2 + (oy - ry) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_id = reg_id

        if best_id is not None:
            obj["reg_id"] = best_id
            _object_registry[best_id].update(obj)
            _object_registry[best_id]["last_seen"] = now
            matched_reg_ids.add(best_id)
        else:
            obj["reg_id"] = _next_reg_id
            _object_registry[_next_reg_id] = {**obj, "last_seen": now}
            _next_reg_id += 1

    # Keep objects that weren't detected this frame but haven't expired
    for reg_id, reg in list(_object_registry.items()):
        if reg_id not in matched_reg_ids and now - reg.get("last_seen", 0) > OBJECT_TIMEOUT:
            del _object_registry[reg_id]

    # Return all registered objects (detected + persisted)
    return [
        {k: v for k, v in reg.items() if k != "last_seen"}
        for reg in _object_registry.values()
    ]


def _detect_aruco(frame):
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
    bots = []
    if ids is None:
        return bots

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS,
    )

    for i, marker_id in enumerate(ids.flatten()):
        c = corners[i][0]
        cx, cy = c.mean(axis=0)
        dx = c[1][0] - c[0][0]
        dy = c[1][1] - c[0][1]
        heading = math.degrees(math.atan2(-dy, dx)) % 360

        bot = {
            "id": int(marker_id),
            "marker_id": int(marker_id),
            "position_px": [round(cx), round(cy)],
            "heading_deg": round(heading, 1),
            "confidence": 1.0,
        }

        if tvecs is not None:
            t = tvecs[i][0]
            bot["position_3d"] = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]

        bots.append(bot)

    return bots


def _detect_yolo(model, frame):
    results = model(frame, verbose=False)[0]
    objects = []

    for i, box in enumerate(results.boxes):
        label = results.names[int(box.cls)]
        if label not in TRACKED_LABELS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        conf = float(box.conf)

        obj = {
            "label": label,
            "type": "bot" if label in BOT_LABELS else "ball",
            "position_px": [round(cx), round(cy)],
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "confidence": round(conf, 3),
        }

        if results.masks is not None and i < len(results.masks):
            mask = results.masks[i].xy[0]
            if len(mask) >= 4:
                pts = mask.astype(np.float32)
                mean = pts.mean(axis=0)
                centered = pts - mean
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                major_axis = eigenvectors[:, -1]
                angle = math.degrees(math.atan2(-major_axis[1], major_axis[0])) % 360
                obj["orientation_deg"] = round(angle, 1)
                obj["elongation"] = round(float(eigenvalues[-1] / max(eigenvalues[0], 1e-6)), 2)

        objects.append(obj)

    return objects, results


def _estimate_depth(depth_model, frame):
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = img.resize((518, 396))
    result = depth_model.predict({"image": img_resized})
    depth_map = result["depth"]
    depth_array = np.array(depth_map)
    if len(depth_array.shape) > 2:
        depth_array = depth_array.squeeze()
    return cv2.resize(depth_array, (frame.shape[1], frame.shape[0]))


def _compute_distances(bots, objects, depth_map):
    distances = []
    h, w = depth_map.shape[:2]

    for bot in bots:
        bx, by = bot["position_px"]
        bot_depth = float(depth_map[min(by, h - 1), min(bx, w - 1)])

        for obj in objects:
            ox, oy = obj["position_px"]
            obj_depth = float(depth_map[min(oy, h - 1), min(ox, w - 1)])
            pixel_dist = math.sqrt((bx - ox) ** 2 + (by - oy) ** 2)
            depth_diff = abs(bot_depth - obj_depth)

            distances.append({
                "from_bot": bot["id"],
                "to_object": obj.get("label", "unknown"),
                "to_reg_id": obj.get("reg_id"),
                "to_position": obj["position_px"],
                "pixel_dist": round(pixel_dist),
                "depth_diff": round(depth_diff, 3),
            })

    return distances


def _annotate_frame(frame, bots, results):
    annotated = results.plot() if results else frame.copy()

    for bot in bots:
        px, py = bot["position_px"]
        heading = bot["heading_deg"]
        rad = math.radians(heading)
        ax = int(px + 40 * math.cos(rad))
        ay = int(py - 40 * math.sin(rad))
        cv2.arrowedLine(annotated, (px, py), (ax, ay), (0, 255, 0), 2, tipLength=0.3)
        label = f"Bot {bot['id']} ({heading:.0f})"
        cv2.putText(annotated, label, (px - 30, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated


def _vision_thread(source, yolo_model, depth_model):
    """Runs in a dedicated thread — captures frames and processes them."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Vision: camera not available")
        return

    print("Vision started")

    # Depth runs in its own thread to avoid blocking YOLO
    depth_executor = ThreadPoolExecutor(max_workers=1) if depth_model else None
    latest_depth_map = [None]  # mutable container for sharing between threads

    def _run_depth(frame):
        latest_depth_map[0] = _estimate_depth(depth_model, frame)

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Always update raw frame for smooth streaming
        set_raw_frame(frame)

        # Run ArUco + YOLO (fast)
        bots = _detect_aruco(frame)
        detected_objects, yolo_results = _detect_yolo(yolo_model, frame)

        # Kick off depth in background (non-blocking)
        if depth_executor:
            depth_executor.submit(_run_depth, frame)

        # Match to persistent registry
        all_objects = _match_to_registry(detected_objects)

        # Use latest available depth map (may be from previous frame)
        distances = []
        depth_map = latest_depth_map[0]
        if depth_map is not None and (bots or all_objects):
            if bots and all_objects:
                distances = _compute_distances(bots, all_objects, depth_map)

            h, w = depth_map.shape[:2]
            for obj in all_objects:
                ox, oy = obj["position_px"]
                obj["depth_rel"] = round(float(depth_map[min(oy, h - 1), min(ox, w - 1)]), 3)

        state = {
            "bots": bots,
            "objects": all_objects,
            "distances": distances,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        }
        set_state(state)

        annotated = _annotate_frame(frame, bots, yolo_results)
        set_frame(frame, annotated)

    cap.release()
    print("Vision stopped")


def start_vision(source=DEFAULT_SOURCE):
    """Start vision in a dedicated thread. Returns the thread."""
    _stop_event.clear()

    print("Vision: loading models...")
    yolo_model = YOLO("yoloe-26s-seg-pf.pt")

    depth_model = None
    if DEPTH_MODEL_PATH.exists():
        depth_model = ct.models.MLModel(str(DEPTH_MODEL_PATH))
        print("Vision: depth model loaded")

    thread = threading.Thread(
        target=_vision_thread,
        args=(source, yolo_model, depth_model),
        daemon=True,
    )
    thread.start()
    return thread
