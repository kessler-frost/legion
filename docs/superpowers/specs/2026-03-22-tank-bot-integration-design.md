# Tank Bot Integration Design

## Summary

Integrate a CyberBrick Mini Tank (Bot 2) into the Legion swarm system alongside the existing SoccerBot (Bot 1). The tank uses tracks for differential steering and a 360° servo-driven flywheel shooter that fires 14mm plastic balls.

## Hardware

- **Platform**: CyberBrick ESP32-C3, single receiver shield
- **Motors**: M1 (left tracks), M2 (right tracks) — 030 DC motors with 1:48 gearboxes
- **Servo**: 360° continuous rotation on S1 (GPIO 3) — spins flywheels to shoot
- **ArUco marker**: #1 (4x4_50 dictionary), mapped to bot_id 2
- **Power**: 14500 7.4V 800mAh Li-ion battery

## Changes

### 1. Firmware — `firmware/tankbot/`

New directory, copied from `firmware/soccerbot/` with modifications:

**`boot.py`** — identical to SoccerBot (skip RC stack, run Legion firmware).

**`config.json`** — `bot_id: 2`, WiFi credentials, broker IP.

**`main.py`** — changes from SoccerBot:
- Rename `kick`/`kick_stop` actions to `shoot`/`shoot_stop`
- Servo setup identical: raw PWM on GPIO 3, `duty(127)` = spin (shoot), `duty(76)` = stop
- Motor mapping: based on designer config, left motor may need positive values (not inverted like SoccerBot). Initial calibration TBD — start with `set_speed(1, params.get("right", 0))` and `set_speed(2, -params.get("left", 0))` same as SoccerBot, then adjust during testing.

### 2. Vision — `brain/vision/detector.py`

Update marker-to-bot mapping:

```python
MARKER_TO_BOT = {2: 1, 1: 2}
```

Make `HEADING_OFFSET` per-bot instead of a single global constant. The tank's marker may be mounted at a different rotation than the SoccerBot's. Initial offset for bot 2 = 0.0, calibrate after first detection.

### 3. CLI — `brain/cli.py`

**New command**: `legion shoot <bot_id> <duration>` — same pattern as `kick`. Publishes `{"action": "shoot"}` then waits, then publishes `{"action": "shoot_stop"}`.

**New calibration entry**: Add bot 2 to `BOT_CALIBRATION` with placeholder values:

```python
BOT_CALIBRATION = {
    1: {"forward": (600, 1000), "backward": (-600, -1000)},
    2: {"forward": (1000, 1000), "backward": (-1000, -1000)},
}
```

Actual values will be determined during motor calibration testing.

### 4. Brain prompt — `brain/reasoning/brain.py`

Update `SYSTEM_PROMPT` to include:
- Bot 2 is a tank with tracks
- `legion shoot <bot_id> <duration>` command for firing (Bot 2 only)
- `legion kick <bot_id> <duration>` is Bot 1 only
- Shooter is on the front of the bot (same orientation as SoccerBot kicker)
- Tank may have different turn characteristics due to tracks vs wheels
- Distance/duration guide for Bot 2 will be added after calibration

### 5. No changes required

These components are already bot-agnostic:
- FastAPI server / API endpoints
- WebSocket video stream
- YOLOE-26x object detection
- DA-V3 depth estimation
- Voice pipeline
- Web UI (command center page — the joystick control page hardcodes `BOT_ID = 1` but that's out of scope for this task)

## Flashing procedure

1. Connect tank via USB-C
2. Open Arduino Lab for MicroPython
3. Upload `boot.py`, `main.py`, `config.json` (with real WiFi creds + broker IP)
4. Reboot the tank
5. Verify it connects to WiFi and MQTT broker
6. Test with `legion forward 2 0.3` and adjust motor calibration

## Testing plan

1. Flash firmware, verify WiFi + MQTT connection
2. Test individual motors: `legion forward 2 0.3`, check direction
3. Fix motor inversion if tracks go backward or bot spins
4. Test turns: `legion left 2 0.2`, `legion right 2 0.2`
5. Calibrate straight-line driving (adjust L/R speed ratio)
6. Test shooter: `legion shoot 2 0.5`
7. Verify ArUco marker #1 is detected by vision system
8. Test CC brain commanding both bots
