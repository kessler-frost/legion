import math
import threading
import time

import cv2
import numpy as np

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
_yolo_model = None


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


def detect_objects_on_demand(frame):
    """Run YOLOE-26x on a frame. Called on-demand, not every frame."""
    global _yolo_model
    if _yolo_model is None:
        from onnxruntime import YOLO
        print("Vision: loading YOLOE-26x model...")
        _yolo_model = YOLO("yoloe-26x-seg-pf.pt")
        print("Vision: model loaded")

    results = _yolo_model.track(frame, verbose=False, conf=0.5, persist=True)[0]
    objects = []
    h, w = frame.shape[:2]

    for box in results.boxes:
        label = results.names[int(box.cls)]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.5 * w * h:
            continue

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        obj = {
            "label": label,
            "position_px": [round(cx), round(cy)],
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "confidence": round(float(box.conf), 3),
        }

        if box.id is not None:
            obj["track_id"] = int(box.id)

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
