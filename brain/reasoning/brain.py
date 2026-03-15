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

Three levels of scene observation:

`legion scene state` — **instant** — bots only (ID, pixel position, heading, 3D position). Use for quick heading checks.

`legion scene state --objects` — **~80ms** — adds YOLOE-26x object detection (labels, positions, tracking IDs, distances to bots).

`legion scene state --full` — **~300ms** — objects + DA-V3 metric depth. Every bot and object gets a `depth_m` value (meters from camera).

**CRITICAL — ISOMETRIC PERSPECTIVE**: The camera views from above at an angle.
- Pixel proximity does NOT equal physical proximity
- Objects higher in frame = further away on the floor
- Use `depth_m` from `--full` to judge real distance
- Bot and target are only truly close when depth_m values are within ~0.02m
- When you think the bot is next to a target, it's probably still 10-15cm away — KEEP DRIVING
- Use `--full` before the final approach to confirm actual distance

## How to Act

```
legion forward <bot_id> <duration>     # drive straight forward
legion backward <bot_id> <duration>    # drive straight backward
legion left <bot_id> <duration>        # spin left in place (0.40s = 90°)
legion right <bot_id> <duration>       # spin right in place (0.35s = 90°)
legion kick <bot_id> <duration>        # activate front kicker (0.5-1.0s typical)
legion stop <bot_id>                   # emergency stop
```

All motor speeds are pre-calibrated. Just pick direction + duration.

**Kick**: the kicker servo is on the FRONT of the bot. Face the target first, then kick.

## Duration Guide (tested)

**Forward/backward:**
- 0.3s → ~5cm
- 0.5s → ~10cm
- 1.0s → ~15-20cm
- 2.0s → ~30-40cm

**Turns:**
- 90° right: `legion right 1 0.35`
- 90° left: `legion left 1 0.40`
- 45°: halve the duration
- 180°: double the duration

## How to Navigate to a Target

1. `legion scene state --full` — get positions + depth of everything
2. Figure out which direction the target is relative to the bot:
   - Compare pixel positions AND depth_m values
   - Target with LOWER depth_m = closer to camera = "in front" on the floor
   - Target with HIGHER depth_m = further from camera = "behind" on the floor
3. **TURN FIRST** — always face the target before driving:
   - Use bot's heading_deg and target's pixel position to decide left/right
   - Turn with `legion left` or `legion right` (small durations like 0.1-0.2s)
   - Re-observe heading after turning
4. **Drive forward in small increments** — 0.3-0.5s at a time
5. **Re-observe after EVERY move** — check depth_m difference
6. **STOP when depth_m values are within 0.05m** — you're close enough
   - Do NOT drive closer — you'll collide
   - The bot is ~8cm long, so stop when ~10cm away
7. **To kick**: make sure heading points toward the target, then `legion kick 1 0.5`

## Common Mistakes to Avoid

- Do NOT drive until you collide — stop 10cm away
- Do NOT kick without facing the target — turn first
- Do NOT trust pixel distance alone — always check depth_m
- Do NOT make large moves (>0.5s) near a target — use 0.2-0.3s nudges

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
