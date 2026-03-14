import threading
from pathlib import Path
from typing import Literal

import aiomqtt
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

BROKER_HOST = "localhost"
BROKER_PORT = 1883

Action = Literal["drive", "stop", "kick", "kick_stop"]

_vision_thread = None


class BotCommand(BaseModel):
    action: Action
    params: dict = {}


app = FastAPI(title="Legion Bot Control")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "home.html").read_text()


@app.get("/control", response_class=HTMLResponse)
async def control_page():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/stream", response_class=HTMLResponse)
async def stream_page():
    return (STATIC_DIR / "stream.html").read_text()


@app.get("/bot/{bot_id}/status")
async def bot_status(bot_id: int):
    return {"bot_id": bot_id, "connected": False}


@app.post("/vision/start")
async def vision_start():
    global _vision_thread
    if _vision_thread and _vision_thread.is_alive():
        return {"status": "already running"}
    from brain.vision.detector import run as run_detector
    import traceback

    def _run_with_logging():
        try:
            run_detector()
        except Exception as e:
            traceback.print_exc()
            print(f"Vision thread crashed: {e}")

    _vision_thread = threading.Thread(target=_run_with_logging, daemon=True)
    _vision_thread.start()
    return {"status": "started"}


@app.post("/vision/stop")
async def vision_stop():
    global _vision_thread
    from brain.vision.detector import request_stop
    request_stop()
    _vision_thread = None
    return {"status": "stopped"}


@app.get("/vision/status")
async def vision_status():
    running = _vision_thread is not None and _vision_thread.is_alive()
    return {"running": running}


@app.get("/scene/state")
async def scene_state():
    from brain.vision.state import read_state
    return read_state()


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket, annotated: bool = False):
    import asyncio
    from brain.vision.state import get_frame

    await websocket.accept()
    last_data = b""
    while True:
        data = get_frame(annotated=annotated)
        if data and data != last_data:
            last_data = data
            await websocket.send_bytes(data)
        await asyncio.sleep(0.03)


@app.post("/bot/{bot_id}/command")
async def send_command(bot_id: int, command: BotCommand):
    topic = f"legion/bot/{bot_id}/command"
    payload = command.model_dump_json()
    async with aiomqtt.Client(BROKER_HOST, BROKER_PORT) as client:
        await client.publish(topic, payload)
    return {"status": "sent", "topic": topic, "command": command.model_dump()}
