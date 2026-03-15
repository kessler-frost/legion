import math
import threading
import time

import cv2
import numpy as np

from brain.vision.state import set_raw_frame, set_state

DEFAULT_SOURCE = 0

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

CAMERA_MATRIX = np.array([
    [930, 0, 640],
    [0, 930, 360],
    [0, 0, 1],
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)
MARKER_SIZE = 0.04

MARKER_TO_BOT = {2: 1}
HEADING_OFFSET = 90

_stop_event = threading.Event()


def request_stop():
    _stop_event.set()


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
        heading = (math.degrees(math.atan2(-dy, dx)) + HEADING_OFFSET) % 360

        bot_id = MARKER_TO_BOT.get(int(marker_id), int(marker_id))

        bot = {
            "id": bot_id,
            "position_px": [round(float(cx)), round(float(cy))],
            "heading_deg": round(heading, 1),
        }

        if tvecs is not None:
            t = tvecs[i][0]
            bot["position_3d_m"] = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]

        bots.append(bot)

    return bots


def _annotate(frame, bots):
    annotated = frame.copy()
    for bot in bots:
        px, py = bot["position_px"]
        heading = bot["heading_deg"]
        rad = math.radians(heading)
        ax = int(px + 50 * math.cos(rad))
        ay = int(py - 50 * math.sin(rad))

        cv2.arrowedLine(annotated, (px, py), (ax, ay), (0, 255, 0), 2, tipLength=0.3)
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

        bots = _detect_aruco(frame)

        state = {
            "bots": bots,
            "frame_size": [frame.shape[1], frame.shape[0]],
        }
        set_state(state)

        annotated = _annotate(frame, bots)
        set_raw_frame(annotated)

    cap.release()
    print("Vision stopped")


def start_vision(source=DEFAULT_SOURCE):
    _stop_event.clear()
    thread = threading.Thread(target=_vision_loop, args=(source,), daemon=True)
    thread.start()
    return thread
