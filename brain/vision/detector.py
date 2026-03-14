import asyncio
import math
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
from onnxruntime import YOLO

from brain.vision.state import set_state, set_frame

DEFAULT_SOURCE = 0
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DEPTH_MODEL_PATH = MODELS_DIR / "DepthAnythingV2SmallF16.mlpackage"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Camera calibration (placeholder — replace after calibration)
# These are rough defaults for a 1080p iPhone camera
CAMERA_MATRIX = np.array([
    [1400, 0, 960],
    [0, 1400, 540],
    [0, 0, 1],
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)
MARKER_SIZE = 0.04  # 4cm marker side length in meters

_stop = False


def request_stop():
    global _stop
    _stop = True


def _detect_aruco(frame):
    """Detect ArUco markers → bot ID, 2D position, heading, optional 3D."""
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
    bots = []

    if ids is None:
        return bots

    # Estimate 3D pose
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS,
    )

    for i, marker_id in enumerate(ids.flatten()):
        c = corners[i][0]
        cx, cy = c.mean(axis=0)

        # Heading from marker orientation (top-right to top-left vector)
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

        # 3D position from pose estimation
        if tvecs is not None:
            t = tvecs[i][0]
            bot["position_3d"] = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]

        bots.append(bot)

    return bots


# COCO labels we care about — everything else is filtered out
BOT_LABELS = {"car", "truck", "motorcycle", "bus"}
BALL_LABELS = {"sports ball", "frisbee", "clock"}
TRACKED_LABELS = BOT_LABELS | BALL_LABELS


def _detect_yolo(model, frame):
    """Run YOLOE-26 detection → scene objects (filtered to bots + balls only)."""
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

        # Add segmentation mask info for orientation estimation
        if results.masks is not None and i < len(results.masks):
            mask = results.masks[i].xy[0]  # polygon points
            if len(mask) >= 4:
                # Compute orientation from mask's major axis (PCA-like)
                pts = mask.astype(np.float32)
                mean = pts.mean(axis=0)
                centered = pts - mean
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Major axis direction = eigenvector with largest eigenvalue
                major_axis = eigenvectors[:, -1]
                angle = math.degrees(math.atan2(-major_axis[1], major_axis[0])) % 360
                obj["orientation_deg"] = round(angle, 1)
                obj["elongation"] = round(float(eigenvalues[-1] / max(eigenvalues[0], 1e-6)), 2)

        objects.append(obj)

    return objects, results


def _estimate_depth(depth_model, frame):
    """Run CoreML Depth Anything V2 → depth map."""
    from PIL import Image

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = img.resize((518, 396))
    result = depth_model.predict({"image": img_resized})
    depth_map = result["depth"]
    # Resize depth map back to frame size
    depth_array = np.array(depth_map)
    if len(depth_array.shape) > 2:
        depth_array = depth_array.squeeze()
    return cv2.resize(depth_array, (frame.shape[1], frame.shape[0]))


def _compute_distances(bots, objects, depth_map):
    """Compute distances between bots and objects using depth."""
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
                "to_object": obj["label"],
                "to_position": obj["position_px"],
                "pixel_dist": round(pixel_dist),
                "depth_diff": round(depth_diff, 3),
            })

    return distances


def _annotate_frame(frame, bots, results):
    """Draw ArUco annotations + YOLO detections on frame."""
    annotated = results.plot() if results else frame.copy()

    for bot in bots:
        px, py = bot["position_px"]
        heading = bot["heading_deg"]

        # Draw heading arrow
        rad = math.radians(heading)
        ax = int(px + 40 * math.cos(rad))
        ay = int(py - 40 * math.sin(rad))
        cv2.arrowedLine(annotated, (px, py), (ax, ay), (0, 255, 0), 2, tipLength=0.3)

        # Draw bot label
        label = f"Bot {bot['id']} ({heading:.0f} deg)"
        cv2.putText(annotated, label, (px - 30, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated


def _pipeline(yolo_model, depth_model, frame):
    """Full pipeline: ArUco + YOLO + Depth. Blocking, called via asyncio.to_thread."""
    bots = _detect_aruco(frame)
    objects, yolo_results = _detect_yolo(yolo_model, frame)

    depth_map = None
    distances = []
    if depth_model and bots and objects:
        depth_map = _estimate_depth(depth_model, frame)
        distances = _compute_distances(bots, objects, depth_map)

        # Add depth to each object
        h, w = depth_map.shape[:2]
        for obj in objects:
            ox, oy = obj["position_px"]
            obj["depth_rel"] = round(float(depth_map[min(oy, h - 1), min(ox, w - 1)]), 3)

    state = {
        "bots": bots,
        "objects": objects,
        "distances": distances,
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
    }

    annotated = _annotate_frame(frame, bots, yolo_results)

    return frame, annotated, state


async def run_vision(source=DEFAULT_SOURCE):
    """Run the vision loop as an asyncio task."""
    global _stop
    _stop = False

    yolo_model = await asyncio.to_thread(YOLO, "yolo26x-seg.pt")
    cap = await asyncio.to_thread(cv2.VideoCapture, source)

    if not cap.isOpened():
        print("Vision: camera not available")
        return

    # Load CoreML depth model
    depth_model = None
    if DEPTH_MODEL_PATH.exists():
        depth_model = await asyncio.to_thread(ct.models.MLModel, str(DEPTH_MODEL_PATH))
        print("Vision: depth model loaded")

    print("Vision started")

    pipeline_task = None

    while not _stop:
        ret, frame = await asyncio.to_thread(cap.read)
        if not ret:
            await asyncio.sleep(0.1)
            continue

        # Always update raw frame for smooth streaming
        from brain.vision.state import set_raw_frame
        set_raw_frame(frame)

        # Only start pipeline if previous one is done
        if pipeline_task is None or pipeline_task.done():
            async def process(f):
                _, annotated, state = await asyncio.to_thread(_pipeline, yolo_model, depth_model, f)
                set_state(state)
                set_frame(f, annotated)

            pipeline_task = asyncio.create_task(process(frame))

    cap.release()
    print("Vision stopped")
