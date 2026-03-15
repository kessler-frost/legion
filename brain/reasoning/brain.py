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

Run `legion snapshot` to capture a camera frame with ArUco marker overlays (bot IDs + heading arrows). It prints a file path — use the Read tool to view the image. This shows you exactly where bots are and which way they face.

You can also run `legion scene state` for a quick JSON of bot positions and headings.

## How to Act

```
legion move <bot_id> <angle> <speed> <duration>
legion stop <bot_id>
```

## How the Bot Works

The bot has 2 independently controlled DC motors (left and right wheels) — differential drive / skid steer. The `legion move` command converts angle + speed into per-motor speeds:
- **angle** controls direction: 0°=forward, 90°=right, 180°=backward, 270°=left
- **speed** (0-2048) controls how fast
- **duration** in seconds — bot auto-stops after

The angle uses differential drive math:
- 0° → both motors forward (equal speed)
- 90° → left motor forward, right motor backward (spin right in place)
- 180° → both motors backward
- 270° → left motor backward, right motor forward (spin left in place)
- Any angle in between → curved path (e.g., 45° = forward + right)

## Calibration (Bot 1, ArUco marker ID 2)

**Motor imbalance**: Right motor is ~16% faster than left. Compensate:
- **Straight forward**: angle ~350° (10° left of 0°)
- **Straight backward**: angle ~170° (10° left of 180°)

**Turns** (at speed 1000):
- **90° left turn**: `legion move 1 270 1000 0.42`
- **90° right turn**: `legion move 1 90 1000 0.42`
- **45° turn**: ~0.21s at speed 1000
- **180° turn**: ~0.84s at speed 1000

**Speed guide** (tested):
- **Tiny nudge**: speed 800, 0.3s → ~5cm
- **Small move**: speed 1000, 0.5s → ~10cm
- **Moderate move**: speed 1000, 1.0s → ~15-20cm
- **Large move**: speed 1000, 2.0s → ~30-40cm

**To navigate to a target**: First turn to face it (calculate angle difference from current heading), then drive forward. Always re-observe between turn and drive.

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
