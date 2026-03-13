from pathlib import Path
from typing import Literal

import aiomqtt
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Legion Bot Control")

STATIC_DIR = Path(__file__).parent / "static"

BROKER_HOST = "localhost"
BROKER_PORT = 1883

Action = Literal["forward", "backward", "left", "right", "stop", "kick"]


class BotCommand(BaseModel):
    action: Action
    params: dict = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/bot/{bot_id}/status")
async def bot_status(bot_id: int):
    return {"bot_id": bot_id, "connected": False}


@app.post("/bot/{bot_id}/command")
async def send_command(bot_id: int, command: BotCommand):
    topic = f"legion/bot/{bot_id}/command"
    payload = command.model_dump_json()
    async with aiomqtt.Client(BROKER_HOST, BROKER_PORT) as client:
        await client.publish(topic, payload)
    return {"status": "sent", "topic": topic, "command": command.model_dump()}
