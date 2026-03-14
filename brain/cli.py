import asyncio
import json
import math
import time

import typer
import uvicorn
import aiomqtt

app = typer.Typer(name="legion", no_args_is_help=True)

BROKER_HOST = "localhost"
BROKER_PORT = 1883
MAX_SPEED = 2048


def publish(topic: str, payload: dict):
    asyncio.run(_publish(topic, payload))


async def _publish(topic: str, payload: dict):
    async with aiomqtt.Client(BROKER_HOST, BROKER_PORT) as client:
        await client.publish(topic, json.dumps(payload))


def angle_speed_to_motors(angle: float, speed: int) -> tuple[int, int]:
    """Convert angle (degrees) + speed to left/right motor speeds.

    0°=forward, 90°=right, 180°=backward, 270°=left.
    Uses differential drive: dx steers, dy throttles.
    """
    rad = math.radians(angle)
    dx = math.sin(rad)
    dy = -math.cos(rad)

    left = int(-dy * speed + dx * speed)
    right = int(-dy * speed - dx * speed)

    left = max(-MAX_SPEED, min(MAX_SPEED, left))
    right = max(-MAX_SPEED, min(MAX_SPEED, right))

    return left, right


@app.command()
def serve(port: int = typer.Option(8000, help="Port to run the server on")):
    """Start the Legion web server."""
    uvicorn.run("brain.api.main:app", host="0.0.0.0", port=port)


@app.command()
def move(
    bot_id: int = typer.Argument(help="Bot ID"),
    angle: float = typer.Argument(help="Direction in degrees (0=forward, 90=right, 180=backward, 270=left)"),
    speed: int = typer.Argument(help="Speed (0-2048)"),
    duration: float = typer.Argument(help="Duration in seconds"),
):
    """Move a bot in a direction for a duration."""
    left, right = angle_speed_to_motors(angle, speed)
    topic = f"legion/bot/{bot_id}/command"

    typer.echo(f"bot {bot_id}: L={left} R={right} for {duration}s")
    publish(topic, {"action": "drive", "params": {"left": left, "right": right}})
    time.sleep(duration)
    publish(topic, {"action": "stop", "params": {}})
    typer.echo(f"bot {bot_id}: stopped")


@app.command()
def kick(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds"),
):
    """Kick for a duration."""
    topic = f"legion/bot/{bot_id}/command"

    typer.echo(f"bot {bot_id}: kick for {duration}s")
    publish(topic, {"action": "kick", "params": {}})
    time.sleep(duration)
    publish(topic, {"action": "kick_stop", "params": {}})
    typer.echo(f"bot {bot_id}: kick stopped")


@app.command()
def stop(
    bot_id: int = typer.Argument(help="Bot ID"),
):
    """Emergency stop a bot."""
    topic = f"legion/bot/{bot_id}/command"
    publish(topic, {"action": "stop", "params": {}})
    typer.echo(f"bot {bot_id}: stopped")
