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
    # Unset CLAUDECODE so the brain's claude-agent-sdk doesn't think it's nested
    env = {**os.environ}
    env.pop("CLAUDECODE", None)

    if background:
        proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "brain.api.main:app", "--host", "0.0.0.0", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        PID_FILE.write_text(str(proc.pid))
        typer.echo(f"server started (pid {proc.pid}, port {port})")
    else:
        os.environ.pop("CLAUDECODE", None)
        uvicorn.run("brain.api.main:app", host="0.0.0.0", port=port)


@serve_app.command("stop")
def serve_stop():
    """Stop ALL legion processes (server, vision, listeners)."""
    _kill_all_legion_processes()
    typer.echo("all legion processes stopped")


# Calibrated motor speeds per bot
BOT_CALIBRATION = {
    1: {"forward": (600, 1000), "backward": (-600, -1000)},
    2: {"forward": (-1000, -1000), "backward": (1000, 1000)},
}
DEFAULT_TURN_SPEED = 1000


def _drive(bot_id: int, left: int, right: int, duration: float):
    topic = f"legion/bot/{bot_id}/command"
    typer.echo(f"bot {bot_id}: L={left} R={right} for {duration}s")
    publish(topic, {"action": "drive", "params": {"left": left, "right": right}})
    time.sleep(duration)
    publish(topic, {"action": "stop", "params": {}})


@app.command()
def forward(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds"),
):
    """Drive forward (calibrated straight line)."""
    cal = BOT_CALIBRATION.get(bot_id, {"forward": (1000, 1000)})
    l, r = cal["forward"]
    _drive(bot_id, l, r, duration)


@app.command()
def backward(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds"),
):
    """Drive backward (calibrated straight line)."""
    cal = BOT_CALIBRATION.get(bot_id, {"backward": (-1000, -1000)})
    l, r = cal["backward"]
    _drive(bot_id, l, r, duration)


@app.command()
def left(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds (0.40s = 90°)"),
):
    """Spin left in place."""
    _drive(bot_id, -DEFAULT_TURN_SPEED, DEFAULT_TURN_SPEED, duration)


@app.command()
def right(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds (0.35s = 90°)"),
):
    """Spin right in place."""
    _drive(bot_id, DEFAULT_TURN_SPEED, -DEFAULT_TURN_SPEED, duration)


@app.command()
def kick(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds"),
):
    """Activate front kicker servo."""
    topic = f"legion/bot/{bot_id}/command"
    typer.echo(f"bot {bot_id}: kick for {duration}s")
    publish(topic, {"action": "kick", "params": {}})
    time.sleep(duration)
    publish(topic, {"action": "kick_stop", "params": {}})


@app.command()
def shoot(
    bot_id: int = typer.Argument(help="Bot ID"),
    duration: float = typer.Argument(help="Duration in seconds"),
):
    """Activate flywheel shooter servo."""
    topic = f"legion/bot/{bot_id}/command"
    typer.echo(f"bot {bot_id}: shoot for {duration}s")
    publish(topic, {"action": "shoot", "params": {}})
    time.sleep(duration)
    publish(topic, {"action": "shoot_stop", "params": {}})


@app.command()
def stop(
    bot_id: int = typer.Argument(help="Bot ID"),
):
    """Emergency stop."""
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
# Scene
# ---------------------------------------------------------------------------

scene_app = typer.Typer(no_args_is_help=True, help="Query the current scene — bot positions, objects, distances.")
app.add_typer(scene_app, name="scene")

API_BASE = "http://localhost:8000"


@scene_app.command("state")
def scene_state(
    objects: bool = typer.Option(False, "--objects", help="Detect objects with YOLOE-26x (~80ms extra)"),
    depth: bool = typer.Option(False, "--depth", help="Add DA-V3 metric depth (~0.25s extra)"),
    full: bool = typer.Option(False, "--full", help="Both objects + depth"),
):
    """Scene state JSON. Default: bots only (instant). Add --objects, --depth, or --full."""
    import urllib.request
    params = []
    if objects or full:
        params.append("objects=true")
    if depth or full:
        params.append("depth=true")
    url = f"{API_BASE}/scene/state"
    if params:
        url += "?" + "&".join(params)
    resp = urllib.request.urlopen(url)
    typer.echo(json.dumps(json.loads(resp.read()), indent=2))


@app.command()
def snapshot():
    """Capture a camera frame with depth analysis. Returns file path + depth at key points."""
    import urllib.request
    resp = urllib.request.urlopen(f"{API_BASE}/scene/snapshot")
    data = json.loads(resp.read())
    typer.echo(json.dumps(data, indent=2))


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
