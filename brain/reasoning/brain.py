# brain/reasoning/brain.py
import asyncio
import signal
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are Legion — a swarm robotics controller. You control CyberBrick robots using the `legion` CLI. Run `legion --help` to discover available commands.

## Key Facts

- YOLO detects bots as "automobile", "motorcycle", "hoverboard", or similar — these are the bots
- Camera is handheld — angles change, always re-observe before and after acting
- Bot 1 calibration: angle ~350 for straight forward (right motor 16% faster), 0.42s at speed 1000 for 90° turn
- Always observe the scene before acting, and re-observe after to confirm

Be concise. Focus on actions.\
"""


async def read_stdin(queue: asyncio.Queue):
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        text = line.strip()
        if text:
            await queue.put(text)


async def run(voice: bool = False, model: str = MODEL):
    input_queue = asyncio.Queue()
    shutdown = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Bash"],
        cwd="/Users/fimbulwinter/dev/legion",
        model=model,
    )

    asyncio.create_task(read_stdin(input_queue))

    if voice:
        from brain.voice.listener import run_with_queue
        asyncio.create_task(run_with_queue(input_queue))

    print(f"Legion Brain ready (model: {model}). Type commands or speak.")
    print("Type 'quit' or 'exit' to stop. Ctrl+C also works.")
    print("---")

    async with ClaudeSDKClient(options=options) as client:
        while not shutdown.is_set():
            try:
                text = await asyncio.wait_for(input_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if text in ("quit", "exit"):
                break

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

    print("Brain stopped.")
