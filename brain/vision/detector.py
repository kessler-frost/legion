import threading
import time

import cv2
from onnxruntime import YOLO

from brain.vision.state import set_state, set_frame

DEFAULT_SOURCE = 0

_stop_event = threading.Event()


def request_stop():
    _stop_event.set()


def run(source=DEFAULT_SOURCE):
    _stop_event.clear()
    model = YOLO("yoloe-26s-seg-pf.pt")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Vision: camera not available")
        return

    print("Vision started")

    while cap.isOpened() and not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        results = model(frame, verbose=False, conf=0.5)[0]

        objects = []
        for box in results.boxes:
            label = results.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            conf = float(box.conf)

            objects.append({
                "label": label,
                "position": [round(cx), round(cy)],
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(conf, 3),
            })

        set_state({
            "bots": [],
            "objects": objects,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        })
        set_frame(frame, results.plot())

    cap.release()
    print("Vision stopped")
