# brain/reasoning/tools.py
import subprocess
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server

PROJECT_DIR = "/Users/fimbulwinter/dev/legion"


def _run_legion(*args: str) -> str:
    result = subprocess.run(
        ["uv", "run", "legion", *args],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )
    return result.stdout or result.stderr or "No output"


@tool(
    "scene_state",
    "Get full scene state JSON: all detected objects with labels, pixel positions, bounding boxes, confidence scores.",
    {},
)
async def scene_state(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": _run_legion("scene", "state")}]}


@tool(
    "scene_describe",
    "Get a human-readable summary of the current scene.",
    {},
)
async def scene_describe(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": _run_legion("scene", "describe")}]}


@tool(
    "move_bot",
    "Move a bot. Angle: 0=forward, 90=right, 180=backward, 270=left. Speed: 0-2048. Duration: seconds. Bot 1 calibration: use ~350 degrees for straight forward. 90-degree turn = 0.42s at speed 1000.",
    {"bot_id": int, "angle": float, "speed": int, "duration": float},
)
async def move_bot(args: dict[str, Any]) -> dict[str, Any]:
    output = _run_legion("move", str(args["bot_id"]), str(args["angle"]), str(args["speed"]), str(args["duration"]))
    return {"content": [{"type": "text", "text": output}]}


@tool(
    "stop_bot",
    "Emergency stop a bot immediately.",
    {"bot_id": int},
)
async def stop_bot(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": _run_legion("stop", str(args["bot_id"]))}]}


@tool(
    "kick_bot",
    "Activate the kick servo on a bot for a duration. Currently broken — servo needs replacement.",
    {"bot_id": int, "duration": float},
)
async def kick_bot(args: dict[str, Any]) -> dict[str, Any]:
    output = _run_legion("kick", str(args["bot_id"]), str(args["duration"]))
    return {"content": [{"type": "text", "text": output}]}


def create_legion_mcp_server():
    return create_sdk_mcp_server(
        name="legion",
        version="1.0.0",
        tools=[scene_state, scene_describe, move_bot, stop_bot, kick_bot],
    )
