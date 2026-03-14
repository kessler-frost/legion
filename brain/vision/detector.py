import time

import cv2
from onnxruntime import YOLO

from brain.vision.state import write_state, save_snapshot

DEFAULT_SOURCE = 0  # Continuity Camera device index (or RTSP URL)
DETECTION_INTERVAL = 1.0


def run(source=DEFAULT_SOURCE):
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(source)

    print(f"Vision started — reading from {source}")
    print(f"Detection interval: {DETECTION_INTERVAL}s")

    last_detection = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        now = time.time()
        if now - last_detection < DETECTION_INTERVAL:
            continue
        last_detection = now

        results = model(frame, verbose=False)[0]

        bots = []
        objects = []

        for box in results.boxes:
            label = results.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            conf = float(box.conf)

            entry = {
                "label": label,
                "position": [round(cx), round(cy)],
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(conf, 3),
            }

            objects.append(entry)

        state = {
            "bots": bots,
            "objects": objects,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        }
        write_state(state)

        annotated = results.plot()
        save_snapshot(annotated)

    cap.release()
    print("Vision stopped")
