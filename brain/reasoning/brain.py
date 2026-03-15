# brain/reasoning/brain.py
import asyncio
import tempfile
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

MODELS = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

DEFAULT_MODEL = "opus"

SYSTEM_PROMPT = """\
You are Legion — a swarm robotics controller. You control CyberBrick robots using the `legion` CLI. Run `legion --help` to discover commands.

## How to See

Always use BOTH of these together:

1. `legion scene state` — JSON with precise bot data (ID, pixel position, heading degrees, 3D position in meters). Instant.
2. `legion snapshot` — raw camera image (no overlays) + depth analysis (DA-V3 metric depth at bot positions). Shows the scene AND tells you how far things are from the camera in meters.

**IMPORTANT**: The camera is isometric — objects higher in the image are FURTHER away, not closer. Use the depth values (in meters) from the snapshot to understand true spatial relationships. Pixel proximity does NOT equal physical proximity.

Use scene state for precise bot positions/headings. Use snapshot to see the scene + get depth. Always call both before acting.

## How to Act

```
legion move <bot_id> <left_speed> <right_speed> <duration>
legion stop <bot_id>
```

Direct motor control. Left and right motor speeds from -2048 to 2048. Positive = forward, negative = backward. Duration in seconds — bot auto-stops after.

**Examples:**
- **Forward**: `legion move 1 1000 1000 1.0` (both motors forward)
- **Backward**: `legion move 1 -1000 -1000 1.0`
- **Spin right**: `legion move 1 1000 -1000 0.42` (left forward, right backward)
- **Spin left**: `legion move 1 -1000 1000 0.42`
- **Curve right**: `legion move 1 1000 500 1.0` (left faster than right)
- **Curve left**: `legion move 1 500 1000 1.0` (right faster than left)

## Calibration (Bot 1, ArUco marker ID 2)

**Motor imbalance**: Right motor is ~16% faster than left. For straight forward, give the left motor more power:
- **Straight forward**: `legion move 1 1000 850 1.0` (left=1000, right=850)
- **Straight backward**: `legion move 1 -1000 -850 1.0`

**Turns** (at speed 1000):
- **90° right**: `legion move 1 1000 -1000 0.42`
- **90° left**: `legion move 1 -1000 1000 0.42`

**Speed guide** (tested):
- **Tiny nudge**: speed ~800, 0.3s → ~5cm
- **Small move**: speed ~1000, 0.5s → ~10cm
- **Moderate move**: speed ~1000, 1.0s → ~15-20cm
- **Large move**: speed ~1000, 2.0s → ~30-40cm

**To navigate to a target**: Use heading from `legion scene state` to determine current facing. Turn to face the target, then drive forward. Always re-observe between turn and drive.

## Rules

- Always `legion scene state` BEFORE acting
- Re-observe AFTER acting to confirm the result
- Camera is fixed — positions are consistent between observations
- The bot's heading_deg tells you which way it faces
- Move in small increments, re-observe between each
- The user speaks informally — interpret intent
- Be concise. Focus on actions.\
"""

# Shared state
_input_queue: asyncio.Queue = None
_subscribers: list[asyncio.Queue] = []
_client: ClaudeSDKClient = None
_ready = asyncio.Event()
_message_history: list[dict] = []
MAX_HISTORY = 200


def get_history() -> list[dict]:
    return _message_history.copy()


def clear_history():
    _message_history.clear()


def _broadcast(msg: dict):
    _message_history.append(msg)
    if len(_message_history) > MAX_HISTORY:
        del _message_history[:len(_message_history) - MAX_HISTORY]
    for q in _subscribers:
        q.put_nowait(msg)


def subscribe() -> asyncio.Queue:
    q = asyncio.Queue()
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue):
    _subscribers.remove(q)


_processing = False


async def send_command(text: str):
    global _processing
    # If brain is processing, interrupt it first
    if _processing and _client:
        _broadcast({"type": "status", "text": "interrupting..."})
        await _client.interrupt()
    await _input_queue.put(text)


async def interrupt_brain():
    if _processing and _client:
        _broadcast({"type": "status", "text": "interrupted"})
        await _client.interrupt()


async def send_audio(audio_bytes: bytes):
    """Transcribe audio and send to brain."""
    from mlx_qwen3_asr import transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
        tmp.write(audio_bytes)

    result = await asyncio.to_thread(transcribe, str(wav_path))
    wav_path.unlink(missing_ok=True)

    text = result.text.strip()
    if text:
        _broadcast({"type": "transcription", "text": text})
        await _input_queue.put(text)
        return text
    return None


async def run_brain(model: str = DEFAULT_MODEL):
    """Run the brain loop. Call as an asyncio task."""
    global _input_queue, _client
    _input_queue = asyncio.Queue()
    model_id = MODELS.get(model, model)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Bash"],
        cwd="/Users/fimbulwinter/dev/legion",
        model=model_id,
    )

    _broadcast({"type": "status", "text": f"Brain ready (model: {model_id})"})
    _ready.set()

    async with ClaudeSDKClient(options=options) as client:
        _client = client
        while True:
            text = await _input_queue.get()
            _broadcast({"type": "user", "text": text})

            global _processing
            _processing = True
            try:
                await client.query(text)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                _broadcast({"type": "assistant", "text": block.text})
                            elif isinstance(block, ToolUseBlock):
                                _broadcast({"type": "tool_use", "tool": block.name, "input": str(block.input)})
                            elif isinstance(block, ToolResultBlock):
                                _broadcast({"type": "tool_result", "text": str(block.content)})
            except Exception as e:
                _broadcast({"type": "error", "text": str(e)})
            _processing = False
