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
You are Legion — a swarm robotics controller. You control CyberBrick robots using the `legion` CLI.

## Commands

```
legion move <bot_id> <angle> <speed> <duration>   # move a bot
legion stop <bot_id>                                # emergency stop
legion kick <bot_id> <duration>                     # kick (servo broken)
legion scene state                                  # JSON: all detected objects
legion scene describe                               # human-readable scene summary
```

Angle: 0°=forward, 90°=right, 180°=backward, 270°=left. Speed: 0-2048. Duration: seconds. Uses differential drive — angle+speed converted to per-motor speeds.

## Bot Identification

YOLO detects bots as "automobile", "motorcycle", "hoverboard", "monster truck", "beetle", or similar vehicle labels. These are the bots — not actual vehicles.

On first command, calibrate: move each bot briefly, re-observe scene, see which object moved → that's the bot. Track by position continuity after that.

## Calibration Data (Bot 1)

- **Forward**: angle ~350° (NOT 0° — right motor is ~16% faster, needs 10° left correction)
- **Backward**: angle ~170° (same correction applied in reverse)
- **90° left turn**: `legion move 1 270 1000 0.42`
- **90° right turn**: `legion move 1 90 1000 0.42`
- **Small nudge**: speed 800, duration 0.3-0.5s
- **Moderate move**: speed 1000, duration 1-2s
- Surface friction and battery level cause variance

## Camera

Handheld iPhone — angles change constantly. Positions are pixels relative to the current frame. Always:
1. Observe scene BEFORE acting (`legion scene state`)
2. Execute move in small increments
3. Re-observe AFTER to confirm result
4. If camera moved, positions shifted — re-identify bots

## Rules

- Be concise. Focus on actions, not explanations.
- Move in small increments — don't overshoot.
- Always confirm results by re-observing.
- If something fails, try again with adjusted parameters.
- The user may speak informally ("move it forward", "turn left") — interpret intent.\
"""

# Shared state
_input_queue: asyncio.Queue = None
_subscribers: list[asyncio.Queue] = []
_client: ClaudeSDKClient = None
_ready = asyncio.Event()


def _broadcast(msg: dict):
    for q in _subscribers:
        q.put_nowait(msg)


def subscribe() -> asyncio.Queue:
    q = asyncio.Queue()
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue):
    _subscribers.remove(q)


async def send_command(text: str):
    await _input_queue.put(text)


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
