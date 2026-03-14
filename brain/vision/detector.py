import threading
import time

import cv2

from brain.vision.state import set_raw_frame

DEFAULT_SOURCE = 0
_stop_event = threading.Event()


def request_stop():
    _stop_event.set()


def _vision_thread(source):
    """Captures frames from camera. That's it."""
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

        set_raw_frame(frame)

    cap.release()
    print("Vision stopped")


def start_vision(source=DEFAULT_SOURCE):
    """Start camera capture in a dedicated thread."""
    _stop_event.clear()
    thread = threading.Thread(target=_vision_thread, args=(source,), daemon=True)
    thread.start()
    return thread
