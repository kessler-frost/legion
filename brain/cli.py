import asyncio
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import typer
import uvicorn
import aiomqtt

app = typer.Typer(name="legion", no_args_is_help=True)

PID_FILE = Path(__file__).parent.parent / ".legion-server.pid"

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


serve_app = typer.Typer(no_args_is_help=True, help="Manage the web server (start/stop).")
app.add_typer(serve_app, name="serve")


def _kill_all_legion_processes():
    """Kill all stale legion/uvicorn processes."""
    subprocess.run(["pkill", "-9", "-f", "uvicorn brain.api.main"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "legion serve"], capture_output=True)
    # Clean up any stale PID files
    for pid_file in Path(__file__).parent.parent.glob(".legion-*.pid"):
        pid_file.unlink(missing_ok=True)
    time.sleep(0.5)


@serve_app.command("start")
def serve_start(
    port: int = typer.Option(8000, help="Port to run the server on"),
    background: bool = typer.Option(False, "--bg", help="Run in background"),
):
    """Start the Legion web server. Kills any stale processes first."""
    _kill_all_legion_processes()
    if background:
        proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "brain.api.main:app", "--host", "0.0.0.0", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        PID_FILE.write_text(str(proc.pid))
        typer.echo(f"server started (pid {proc.pid}, port {port})")
    else:
        uvicorn.run("brain.api.main:app", host="0.0.0.0", port=port)


@serve_app.command("stop")
def serve_stop():
    """Stop ALL legion processes (server, vision, listeners)."""
    _kill_all_legion_processes()
    typer.echo("all legion processes stopped")


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


# ---------------------------------------------------------------------------
# Vision start / stop
# ---------------------------------------------------------------------------

vision_app = typer.Typer(no_args_is_help=True, help="Manage the YOLO vision pipeline (start/stop).")
app.add_typer(vision_app, name="vision")

VISION_PID_FILE = Path(__file__).parent.parent / ".legion-vision.pid"


@vision_app.command("start")
def vision_start(
    background: bool = typer.Option(False, "--bg", help="Run in background"),
    source: str = typer.Option("0", help="Camera device index or RTSP URL"),
):
    """Start the vision detection pipeline."""
    parsed_source = int(source) if source.isdigit() else source
    if background:
        proc = subprocess.Popen(
            ["uv", "run", "python", "-c", f"from brain.vision.detector import run; run({repr(parsed_source)})"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        VISION_PID_FILE.write_text(str(proc.pid))
        typer.echo(f"vision started (pid {proc.pid})")
    else:
        from brain.vision.detector import run

        run(parsed_source)


@vision_app.command("stop")
def vision_stop():
    """Stop the vision detection pipeline."""
    pid = int(VISION_PID_FILE.read_text())
    VISION_PID_FILE.unlink()
    os.kill(pid, signal.SIGTERM)
    typer.echo(f"vision stopped (pid {pid})")


# ---------------------------------------------------------------------------
# Scene queries
# ---------------------------------------------------------------------------

scene_app = typer.Typer(no_args_is_help=True, help="Query the current scene from the vision pipeline.")
app.add_typer(scene_app, name="scene")

API_BASE = "http://localhost:8000"


def _get_scene() -> dict:
    import urllib.request
    resp = urllib.request.urlopen(f"{API_BASE}/scene/state")
    return json.loads(resp.read())


@scene_app.command("state")
def scene_state():
    """Print full scene state as JSON."""
    typer.echo(json.dumps(_get_scene(), indent=2))


@scene_app.command("bots")
def scene_bots():
    """Print bot positions."""
    state = _get_scene()
    typer.echo(json.dumps(state.get("bots", []), indent=2))


@scene_app.command("objects")
def scene_objects():
    """Print detected objects."""
    state = _get_scene()
    typer.echo(json.dumps(state.get("objects", []), indent=2))


@scene_app.command("describe")
def scene_describe():
    """Human-readable scene summary."""
    state = _get_scene()
    lines = []
    lines.append(f"Scene at {state.get('timestamp', 'unknown')}:")
    lines.append(f"  Frame: {state.get('frame_width', '?')}x{state.get('frame_height', '?')}")

    bots = state.get("bots", [])
    lines.append(f"  Bots: {len(bots)}")
    for b in bots:
        lines.append(f"    Bot {b.get('id', '?')}: pos={b['position']}, conf={b['confidence']}")

    objects = state.get("objects", [])
    lines.append(f"  Objects: {len(objects)}")
    for o in objects:
        lines.append(f"    {o['label']}: pos={o['position']}, conf={o['confidence']}")

    typer.echo("\n".join(lines))


# ---------------------------------------------------------------------------
# Listen start / stop
# ---------------------------------------------------------------------------

listen_app = typer.Typer(no_args_is_help=True, help="Manage the voice command listener (start/stop).")
app.add_typer(listen_app, name="listen")

LISTEN_PID_FILE = Path(__file__).parent.parent / ".legion-listen.pid"


@listen_app.command("start")
def listen_start(
    background: bool = typer.Option(False, "--bg", help="Run in background"),
):
    """Start the voice command listener."""
    if background:
        proc = subprocess.Popen(
            ["uv", "run", "python", "-c", "from brain.voice.listener import run; run()"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        LISTEN_PID_FILE.write_text(str(proc.pid))
        typer.echo(f"listener started (pid {proc.pid})")
    else:
        from brain.voice.listener import run

        run()


@listen_app.command("stop")
def listen_stop():
    """Stop the voice command listener."""
    pid = int(LISTEN_PID_FILE.read_text())
    LISTEN_PID_FILE.unlink()
    os.kill(pid, signal.SIGTERM)
    typer.echo(f"listener stopped (pid {pid})")


# ---------------------------------------------------------------------------
# Brain (CC reasoning)
# ---------------------------------------------------------------------------

brain_app = typer.Typer(no_args_is_help=True, help="Manage the CC reasoning engine (start/stop).")
app.add_typer(brain_app, name="brain")


@brain_app.command("start")
def brain_start(
    voice: bool = typer.Option(False, "--voice", help="Enable voice commands from iPhone mic"),
    model: str = typer.Option("claude-sonnet-4-6", help="Model to use for reasoning"),
):
    """Start the CC brain session (interactive)."""
    from brain.reasoning.brain import run
    asyncio.run(run(voice=voice, model=model))
