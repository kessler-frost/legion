from typing import Literal

import aiomqtt
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Legion Bot Control")

BROKER_HOST = "localhost"
BROKER_PORT = 1883

Action = Literal["forward", "backward", "left", "right", "stop", "kick"]


class BotCommand(BaseModel):
    action: Action
    params: dict = {}


@app.post("/bot/{bot_id}/command")
async def send_command(bot_id: int, command: BotCommand):
    topic = f"legion/bot/{bot_id}/command"
    payload = command.model_dump_json()
    async with aiomqtt.Client(BROKER_HOST, BROKER_PORT) as client:
        await client.publish(topic, payload)
    return {"status": "sent", "topic": topic, "command": command.model_dump()}
