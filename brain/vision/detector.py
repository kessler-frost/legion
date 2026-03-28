import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

from brain.vision.state import set_frames, set_state

DEFAULT_SOURCE = 0

# ArUco setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.minMarkerPerimeterRate = 0.08
ARUCO_PARAMS.maxErroneousBitsInBorderRate = 0.2
ARUCO_PARAMS.errorCorrectionRate = 0.0
ARUCO_PARAMS.minOtsuStdDev = 10.0
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

CAMERA_MATRIX = np.array([
    [930, 0, 640],
    [0, 930, 360],
    [0, 0, 1],
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)
MARKER_SIZE = 0.04

MARKER_TO_BOT = {2: 1, 3: 2}
MARKER_HEADING_OFFSET = {2: 2.1, 3: 0.0}  # keyed by ArUco marker ID

_stop_event = threading.Event()
_onnx_session = None
_class_names = None
_tracker = None

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent
_ONNX_PATH = _MODEL_DIR / "yoloe-26x-seg-pf.onnx"
_NAMES_PATH = _MODEL_DIR / "yoloe-26x-seg-pf-names.json"
_INPUT_SIZE = 640


def request_stop():
    _stop_event.set()


def _detect_aruco(frame):
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(frame)
    bots = []
    if ids is None:
        return bots

    obj_points = np.array([
        [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
        [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
        [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
        [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    ], dtype=np.float32)

    for i, marker_id in enumerate(ids.flatten()):
        c = corners[i][0]
        cx, cy = c.mean(axis=0)

        dx = c[1][0] - c[0][0]
        dy = c[1][1] - c[0][1]
        raw_heading = math.degrees(math.atan2(-dy, dx))
        offset = MARKER_HEADING_OFFSET.get(int(marker_id), 0.0)
        heading = (360 - raw_heading + offset) % 360

        bot_id = MARKER_TO_BOT.get(int(marker_id), int(marker_id))

        bot = {
            "id": bot_id,
            "position_px": [round(float(cx)), round(float(cy))],
            "heading_deg": round(heading, 1),
        }

        ret, rvec, tvec = cv2.solvePnP(obj_points, c, CAMERA_MATRIX, DIST_COEFFS)
        if ret:
            t = tvec.flatten()
            bot["position_3d_m"] = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]

        bots.append(bot)

    return bots


def _letterbox(frame):
    """Resize frame to _INPUT_SIZE with letterbox padding, return padded image and transform params."""
    h, w = frame.shape[:2]
    scale = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 114, dtype=np.uint8)
    dx, dy = (_INPUT_SIZE - nw) // 2, (_INPUT_SIZE - nh) // 2
    padded[dy:dy + nh, dx:dx + nw] = resized
    return padded, scale, dx, dy


def detect_objects_on_demand(frame):
    """Run YOLOE-26x (ONNX) on a frame. Called on-demand, not every frame."""
    global _onnx_session, _class_names, _tracker
    if _onnx_session is None:
        print("Vision: loading YOLOE-26x ONNX model...")
        _onnx_session = ort.InferenceSession(str(_ONNX_PATH))
        with _NAMES_PATH.open() as f:
            _class_names = json.load(f)
        _tracker = sv.ByteTrack(minimum_consecutive_frames=1)
        print("Vision: model loaded")

    h, w = frame.shape[:2]

    # Preprocess: letterbox, BGR->RGB, normalize, HWC->NCHW
    padded, scale, dx, dy = _letterbox(frame)
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    # Inference
    outputs = _onnx_session.run(None, {"images": blob})
    dets = outputs[0][0]  # (300, 38): [x1, y1, x2, y2, conf, class_id, 32 mask coeffs]

    # Filter by confidence
    mask = dets[:, 4] > 0.5
    dets = dets[mask]

    if len(dets) == 0:
        return []

    # Scale bboxes from letterbox space back to original frame coordinates
    bboxes = dets[:, :4].copy()
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dx) / scale
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dy) / scale
    bboxes = np.clip(bboxes, 0, [w, h, w, h])

    confidences = dets[:, 4]
    class_ids = dets[:, 5].astype(int)

    # Track with ByteTrack
    sv_dets = sv.Detections(
        xyxy=bboxes,
        confidence=confidences,
        class_id=class_ids,
    )
    sv_dets = _tracker.update_with_detections(sv_dets)

    # Build output (same format as before)
    objects = []
    for i in range(len(sv_dets)):
        x1, y1, x2, y2 = sv_dets.xyxy[i]
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.5 * w * h:
            continue

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        label = _class_names[str(sv_dets.class_id[i])]

        obj = {
            "label": label,
            "position_px": [round(cx), round(cy)],
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "confidence": round(float(sv_dets.confidence[i]), 3),
        }

        if sv_dets.tracker_id is not None and sv_dets.tracker_id[i] is not None:
            obj["track_id"] = int(sv_dets.tracker_id[i])

        objects.append(obj)

    return objects


def _annotate(frame, bots):
    annotated = frame.copy()
    for bot in bots:
        px, py = bot["position_px"]
        heading = bot["heading_deg"]
        rad = math.radians(heading)
        ax = int(px + 50 * math.sin(rad))
        ay = int(py - 50 * math.cos(rad))
        cv2.arrowedLine(annotated, (px, py), (ax, ay), (0, 0, 0), 4, tipLength=0.3)
        cv2.arrowedLine(annotated, (px, py), (ax, ay), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(annotated, f"Bot {bot['id']}", (px - 20, py - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(annotated, f"Bot {bot['id']}", (px - 20, py - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def _vision_loop(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Vision: camera not available")
        return

    print("Vision started")

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, -1)

        # ArUco only — fast, every frame
        bots = _detect_aruco(frame)

        state = {
            "bots": bots,
            "frame_size": [frame.shape[1], frame.shape[0]],
        }
        set_state(state)

        annotated = _annotate(frame, bots)
        set_frames(frame, annotated)

    cap.release()
    print("Vision stopped")


def start_vision(source=DEFAULT_SOURCE):
    _stop_event.clear()
    thread = threading.Thread(target=_vision_loop, args=(source,), daemon=True)
    thread.start()
    return thread
