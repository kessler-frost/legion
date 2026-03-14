# brain/reasoning/brain.py
import asyncio
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from brain.reasoning.tools import create_legion_mcp_server

SYSTEM_PROMPT = """\
You are Legion — a swarm robotics controller. You control CyberBrick robots in an arena observed by a handheld iPhone camera.

## Your Tools

- `scene_state`: Get JSON of all detected objects (labels, pixel positions, bounding boxes, confidence)
- `scene_describe`: Get a human-readable scene summary
- `move_bot`: Move a bot by angle (0=fwd, 90=right, 180=back, 270=left), speed (0-2048), duration (seconds)
- `stop_bot`: Emergency stop a bot
- `kick_bot`: Activate kick servo (currently broken — servo needs replacement)

## Bot Identification

YOLO detects bots as "automobile", "motorcycle", "hoverboard", "monster truck", or similar vehicle labels. These are the bots.

On startup, calibrate by moving each bot briefly and observing which object moves in the scene state. Track bots by position continuity after that.

## Calibration Data (Bot 1)

- Straight forward: use angle ~350 (right motor is ~16% faster, 10° left correction needed)
- 90° turn: 0.42 seconds at speed 1000
- Surface friction and battery level cause variance — re-observe after each move

## Camera

The camera is handheld — angles change constantly. Always re-observe the scene before and after acting. Positions are in pixels relative to the current frame.

## How to Work

1. When you receive a command, first observe the scene (call scene_state)
2. Reason about what needs to happen
3. Execute moves in small increments, re-observing between each
4. Confirm the result to the user

Be concise in your responses. Focus on actions, not explanations.\
"""


async def read_stdin(queue: asyncio.Queue):
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        text = line.strip()
        if text:
            await queue.put(text)


async def run(voice: bool = False):
    input_queue = asyncio.Queue()

    legion_server = create_legion_mcp_server()

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"legion": legion_server},
        allowed_tools=[
            "mcp__legion__scene_state",
            "mcp__legion__scene_describe",
            "mcp__legion__move_bot",
            "mcp__legion__stop_bot",
            "mcp__legion__kick_bot",
        ],
        cwd="/Users/fimbulwinter/dev/legion",
    )

    asyncio.create_task(read_stdin(input_queue))

    if voice:
        from brain.voice.listener import run_with_queue
        asyncio.create_task(run_with_queue(input_queue))

    print("Legion Brain ready. Type commands or speak into the iPhone mic.")
    print("---")

    async with ClaudeSDKClient(options=options) as client:
        while True:
            text = await input_queue.get()
            print(f"> {text}")

            await client.query(text)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)
                elif isinstance(message, ResultMessage):
                    print(f"[cost: ${message.total_cost_usd:.4f}]")
            print("---")
