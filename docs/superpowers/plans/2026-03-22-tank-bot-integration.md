# Tank Bot Integration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect a CyberBrick Mini Tank (Bot 2) to the Legion swarm system — firmware, vision, CLI, and brain prompt.

**Architecture:** Copy-and-adapt from the existing SoccerBot firmware. Tank has the same board layout (1 receiver shield, M1/M2 motors, servo on GPIO 3) but uses tracks instead of wheels and a flywheel shooter instead of a kicker. All changes are additive — no existing behavior changes.

**Tech Stack:** MicroPython (ESP32-C3 firmware), Python/Typer (CLI), OpenCV ArUco (vision), claude-agent-sdk (brain)

**Spec:** `docs/superpowers/specs/2026-03-22-tank-bot-integration-design.md`

---

## Chunk 1: Firmware + CLI + Vision + Brain

### Task 1: Create tank firmware

**Files:**
- Create: `firmware/tankbot/boot.py`
- Create: `firmware/tankbot/main.py`
- Create: `firmware/tankbot/config.json`

- [ ] **Step 1: Create `firmware/tankbot/boot.py`**

Identical to SoccerBot — skips stock RC stack, runs Legion firmware:

```python
# Custom boot — skips stock RC stack, runs Legion firmware
import bbl_product
import sys
import gc

bbl_product.set_app_name("LEGION")
bbl_product.set_app_version("00.01.00.00")
del bbl_product

sys.path.append("/lib")
gc.collect()

import main
main.run()
```

- [ ] **Step 2: Create `firmware/tankbot/config.json`**

Template config — user fills in real WiFi creds + broker IP before flashing:

```json
{
    "wifi_ssid": "YOUR_SSID",
    "wifi_password": "YOUR_PASSWORD",
    "mqtt_broker": "192.168.X.X",
    "mqtt_port": 1883,
    "bot_id": 2
}
```

- [ ] **Step 3: Create `firmware/tankbot/main.py`**

Adapted from `firmware/soccerbot/main.py`. Changes from SoccerBot:
- `kick` / `kick_stop` renamed to `shoot` / `shoot_stop`
- Helper functions renamed: `kick_start` → `shoot_start`, `kick_stop` → `shoot_stop_servo`

```python
# firmware/tankbot/main.py
import json
import time
import network
import uasyncio as asyncio
from umqtt.simple import MQTTClient
from machine import Pin, PWM
from bbl.motors import MotorsController


motors = None
servo_pwm = None


def load_config():
    with open("config.json") as f:
        return json.load(f)


def wifi_connect(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    while not wlan.isconnected():
        time.sleep(0.1)
    print("WiFi connected:", wlan.ifconfig())
    return wlan


def stop_all():
    motors.stop(1)
    motors.stop(2)
    servo_pwm.duty(76)


def shoot_start():
    servo_pwm.duty(127)


def shoot_stop_servo():
    servo_pwm.duty(76)


def on_message(topic, msg):
    print(f"CMD: {msg}")
    cmd = json.loads(msg)
    action = cmd.get("action")
    params = cmd.get("params", {})
    actions = {
        "drive":      lambda: (motors.set_speed(1, params.get("right", 0)), motors.set_speed(2, -params.get("left", 0))),
        "stop":       lambda: stop_all(),
        "shoot":      lambda: shoot_start(),
        "shoot_stop": lambda: shoot_stop_servo(),
    }
    handler = actions.get(action)
    if handler:
        handler()


async def mqtt_loop(client):
    while True:
        client.check_msg()
        await asyncio.sleep_ms(50)


def run():
    global motors, servo_pwm
    config = load_config()
    wifi_connect(config["wifi_ssid"], config["wifi_password"])

    motors = MotorsController()
    print("Motors ready")

    # Raw PWM for servo — ServosController conflicts with easypwm from MotorsController
    servo_pwm = PWM(Pin(3), freq=50)
    servo_pwm.duty(76)
    print("Servo ready")

    client = MQTTClient(
        client_id=f"legion_bot_{config['bot_id']}",
        server=config["mqtt_broker"],
        port=config["mqtt_port"],
    )
    client.set_callback(on_message)
    client.connect()
    topic = f"legion/bot/{config['bot_id']}/command"
    client.subscribe(topic.encode())

    # Drain any queued messages before announcing ready
    for _ in range(10):
        client.check_msg()
        time.sleep_ms(50)

    print("Ready")
    asyncio.run(mqtt_loop(client))
```

- [ ] **Step 4: Commit firmware**

```bash
git add firmware/tankbot/
git commit -m "feat: add tank bot firmware (bot 2, shoot action)"
```

### Task 2: Add CLI shoot command + bot 2 calibration

**Files:**
- Modify: `brain/cli.py:80-83` (BOT_CALIBRATION) and add new command after `kick`

- [ ] **Step 1: Add bot 2 calibration entry**

In `brain/cli.py`, update `BOT_CALIBRATION`:

```python
BOT_CALIBRATION = {
    1: {"forward": (600, 1000), "backward": (-600, -1000)},
    2: {"forward": (1000, 1000), "backward": (-1000, -1000)},
}
```

- [ ] **Step 2: Add `legion shoot` command**

Add after the `kick` command (after line 145 in `brain/cli.py`):

```python
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
```

- [ ] **Step 3: Verify CLI registers the new command**

Run: `legion --help`
Expected: `shoot` appears in the command list alongside `forward`, `backward`, `left`, `right`, `kick`, `stop`

- [ ] **Step 4: Commit**

```bash
git add brain/cli.py
git commit -m "feat: add shoot CLI command + bot 2 calibration"
```

### Task 3: Update vision — marker mapping + per-bot heading offset

**Files:**
- Modify: `brain/vision/detector.py:29-30` (MARKER_TO_BOT, HEADING_OFFSET)
- Modify: `brain/vision/detector.py:60` (heading calculation)

- [ ] **Step 1: Update marker mapping and heading offset**

In `brain/vision/detector.py`, replace:

```python
MARKER_TO_BOT = {2: 1}
HEADING_OFFSET = 2.1
```

with:

```python
MARKER_TO_BOT = {2: 1, 1: 2}
MARKER_HEADING_OFFSET = {1: 0.0, 2: 2.1}  # keyed by ArUco marker ID, not bot ID
```

- [ ] **Step 2: Update heading calculation to use per-marker offset**

In `_detect_aruco`, replace:

```python
heading = (360 - raw_heading + HEADING_OFFSET) % 360
```

with:

```python
offset = MARKER_HEADING_OFFSET.get(int(marker_id), 0.0)
heading = (360 - raw_heading + offset) % 360
```

- [ ] **Step 3: Commit**

```bash
git add brain/vision/detector.py
git commit -m "feat: add tank marker mapping + per-bot heading offset"
```

### Task 4: Update brain system prompt

**Files:**
- Modify: `brain/reasoning/brain.py:24-108` (SYSTEM_PROMPT)

- [ ] **Step 1: Update SYSTEM_PROMPT**

In `brain/reasoning/brain.py`, update the "How to Act" section. Replace the existing command block:

```
legion forward <bot_id> <duration>     # drive straight forward
legion backward <bot_id> <duration>    # drive straight backward
legion left <bot_id> <duration>        # spin left in place (0.40s = 90°)
legion right <bot_id> <duration>       # spin right in place (0.35s = 90°)
legion kick <bot_id> <duration>        # activate front kicker (0.5-1.0s typical)
legion stop <bot_id>                   # emergency stop
```

with:

```
legion forward <bot_id> <duration>     # drive straight forward
legion backward <bot_id> <duration>    # drive straight backward
legion left <bot_id> <duration>        # spin left in place (0.40s = 90°)
legion right <bot_id> <duration>       # spin right in place (0.35s = 90°)
legion kick <bot_id> <duration>        # activate front kicker (Bot 1 only)
legion shoot <bot_id> <duration>       # activate flywheel shooter (Bot 2 only)
legion stop <bot_id>                   # emergency stop
```

Update the existing Kick description to be Bot 1 specific, and add Shoot after it. Replace:

```
**Kick**: the kicker servo is on the FRONT of the bot. Face the target first, then kick.
```

with:

```
**Kick** (Bot 1 only): the kicker servo is on the FRONT of Bot 1. Face the target first, then kick.

**Shoot** (Bot 2 only): the shooter is on the FRONT of Bot 2 (the tank). Face the target, then shoot. Uses flywheel launcher — fires 14mm plastic balls.
```

Add a new section after "Common Mistakes to Avoid":

```
## Bot Reference

**Bot 1 (SoccerBot)**: wheels, kicker servo → use `legion kick`
**Bot 2 (Tank)**: tracks, flywheel shooter → use `legion shoot`

Both bots use the same movement commands (forward/backward/left/right/stop).
Tank turns may behave differently than SoccerBot due to tracks — use the same small-increment approach.
```

- [ ] **Step 2: Commit**

```bash
git add brain/reasoning/brain.py
git commit -m "feat: update brain prompt with tank bot commands"
```

### Task 5: Update project docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Add bot 2 calibration section alongside existing bot 1 section. Under "## Bot Calibration", add:

```markdown
### Bot 2 (Tank, ArUco marker #1)
- **Drive**: tracks, differential steering
- **Straight forward**: L=1000, R=1000 (placeholder — calibrate after first test)
- **Turns**: tracks may turn more consistently than wheels — still use small increments
- **Heading**: same convention as Bot 1 (0°=up, 90°=right, clockwise)
- **Servo**: 360° on S1 (GPIO 3). Flywheel shooter on front. duty(127) = shoot, duty(76) = stop.
```

Update the CLI section to include:
```
legion shoot <bot_id> <duration>       # activate flywheel shooter
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add tank bot (bot 2) to project docs"
```
