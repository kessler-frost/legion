import asyncio

import cv2
from onnxruntime import YOLO

from brain.vision.state import set_state, set_frame

DEFAULT_SOURCE = 0

_stop = False


def request_stop():
    global _stop
    _stop = True


def _capture_and_detect(model, cap):
    """Blocking: read frame + run YOLO. Called via asyncio.to_thread."""
    ret, frame = cap.read()
    if not ret:
        return None, None, None

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

    state = {
        "bots": [],
        "objects": objects,
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
    }

    return frame, results.plot(), state


async def run_vision(source=DEFAULT_SOURCE):
    """Run the vision loop. Call as an asyncio task."""
    global _stop
    _stop = False

    model = await asyncio.to_thread(YOLO, "yoloe-26s-seg-pf.pt")
    cap = await asyncio.to_thread(cv2.VideoCapture, source)

    if not cap.isOpened():
        print("Vision: camera not available")
        return

    print("Vision started")

    while not _stop:
        result = await asyncio.to_thread(_capture_and_detect, model, cap)
        frame, annotated, state = result

        if frame is None:
            await asyncio.sleep(0.1)
            continue

        set_state(state)
        set_frame(frame, annotated)

    cap.release()
    print("Vision stopped")
