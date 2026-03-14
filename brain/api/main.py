import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiomqtt
from fastapi import FastAPI, WebSocket, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

BROKER_HOST = "localhost"
BROKER_PORT = 1883

Action = Literal["drive", "stop", "kick", "kick_stop"]

_vision_task = None


class BotCommand(BaseModel):
    action: Action
    params: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start brain on server startup
    from brain.reasoning.brain import run_brain
    asyncio.create_task(run_brain())
    print("Brain started")
    yield


app = FastAPI(title="Legion Bot Control", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "home.html").read_text()


@app.get("/control", response_class=HTMLResponse)
async def control_page():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/command", response_class=HTMLResponse)
async def command_page():
    return (STATIC_DIR / "command.html").read_text()


# --- Vision ---

@app.post("/vision/start")
async def vision_start():
    global _vision_task
    if _vision_task and not _vision_task.done():
        return {"status": "already running"}
    from brain.vision.detector import run_vision
    _vision_task = asyncio.create_task(run_vision())
    return {"status": "started"}


@app.post("/vision/stop")
async def vision_stop():
    global _vision_task
    from brain.vision.detector import request_stop
    request_stop()
    _vision_task = None
    return {"status": "stopped"}


@app.get("/vision/status")
async def vision_status():
    running = _vision_task is not None and not _vision_task.done()
    return {"running": running}


@app.get("/scene/state")
async def scene_state():
    from brain.vision.state import read_state
    return read_state()


# --- Video stream ---

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket, annotated: bool = False):
    from brain.vision.state import get_frame

    await websocket.accept()
    last_data = b""
    while True:
        data = get_frame(annotated=annotated)
        if data and data != last_data:
            last_data = data
            await websocket.send_bytes(data)
        await asyncio.sleep(0.03)


# --- Brain ---

@app.post("/brain/send")
async def brain_send(body: dict):
    from brain.reasoning.brain import send_command
    text = body.get("text", "").strip()
    if text:
        await send_command(text)
        return {"status": "sent", "text": text}
    return {"status": "empty"}


@app.post("/brain/listen")
async def brain_listen(audio: UploadFile):
    from brain.reasoning.brain import send_audio
    audio_bytes = await audio.read()
    text = await send_audio(audio_bytes)
    return {"status": "transcribed", "text": text}


@app.websocket("/ws/brain")
async def ws_brain(websocket: WebSocket):
    from brain.reasoning.brain import subscribe, unsubscribe

    await websocket.accept()
    q = subscribe()
    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except Exception:
        pass
    finally:
        unsubscribe(q)


# --- Bot control ---

@app.get("/bot/{bot_id}/status")
async def bot_status(bot_id: int):
    return {"bot_id": bot_id, "connected": False}


@app.post("/bot/{bot_id}/command")
async def send_bot_command(bot_id: int, command: BotCommand):
    topic = f"legion/bot/{bot_id}/command"
    payload = command.model_dump_json()
    async with aiomqtt.Client(BROKER_HOST, BROKER_PORT) as client:
        await client.publish(topic, payload)
    return {"status": "sent", "topic": topic, "command": command.model_dump()}
