# Legion

AI-controlled swarm robotics system. Fixed webcam watches CyberBrick robots; a Mac Mini runs the brain.

## Architecture

CC uses the `legion` CLI for everything — observe, think, act.

```
USB Webcam (C270) → ArUco (every frame) + YOLOE-26x (on demand) + DA-V3 (on demand)
                                                      ↑
                                        CC (claude-agent-sdk) → legion CLI → MQTT → Bots
                                                      ↑
                                        Browser mic → mlx-qwen3-asr → Transcribed Commands
```

### Hardware

- **Bots**: CyberBrick robots (ESP32-C3, MicroPython, Bambu Lab). ArUco marker on top.
- **Brain**: Mac Mini M4 Pro, 48GB RAM
- **Camera**: Logitech C270 720p USB webcam, fixed isometric mount (flipped 180° in software)

### Vision Pipeline

Runs as a background thread inside the FastAPI server.

- **ArUco**: every frame (<1ms) — bot ID, pixel position, heading, 3D pose
- **YOLOE-26x**: on-demand via `--objects` flag (~80ms) — 4,585 class detection + tracking
- **DA-V3 Metric Depth**: on-demand via `--depth` flag (~250ms) — metric depth in meters per object
- **Stream**: WebSocket at `/ws/stream` serves JPEG frames with ArUco overlay

### Voice Pipeline

Browser-based auto-listen with 1s silence threshold.

- **STT**: mlx-qwen3-asr 0.6B (4-bit), ~40ms to transcribe 2s of audio
- **Input**: Browser MediaRecorder → POST /brain/listen → transcribe → CC

### Reasoning (Claude Code)

Starts automatically with the server. CC drives an observe-think-act loop:

1. **Observe**: `legion scene state [--objects] [--depth] [--full]`
2. **Think**: Reason about positions, headings, depth
3. **Act**: `legion forward` / `legion left` / `legion right` / `legion kick` / `legion stop`
4. **Repeat**

### Communication

- **Broker**: Mosquitto running locally on Mac Mini (`localhost:1883`)
- **Command topic**: `legion/bot/{id}/command`
- **Bot firmware actions**: `drive` (per-motor speeds), `stop`, `kick`, `kick_stop`
- **Bot-side MQTT**: `umqtt.simple` (MicroPython)

### Bot Firmware

Custom MicroPython on CyberBrick ESP32-C3. Connects WiFi STA → MQTT broker → receives commands.

- **Actions**: `drive` (per-motor speeds), `stop`, `kick`, `kick_stop`
- **Motor mapping**: Motor 1 = right wheels (positive = forward), Motor 2 = left wheels (inverted)
- **Servo**: Raw PWM on GPIO 3 — duty(127) = kick, duty(76) = stop
- **Upload**: Arduino Lab for MicroPython over USB-C

## Tech Stack

- Python 3.12+, managed with `uv` (not pip/venv)
- FastAPI + uvicorn (web server + API)
- YOLOE-26x (object detection, on-demand)
- DA-V3 Metric Large (depth estimation, on-demand)
- mlx-qwen3-asr (speech-to-text, Apple Silicon optimized)
- OpenCV (`opencv-contrib-python` for ArUco + camera)
- claude-agent-sdk (reasoning)
- MicroPython (bot firmware)
- MQTT (Mosquitto broker, umqtt bot-side)
- Typer (CLI)

## Project Structure

```
legion/
├── brain/
│   ├── api/            # FastAPI server + static frontend
│   │   ├── main.py     # API endpoints, WebSocket stream, brain startup
│   │   └── static/     # HTML pages (home, control, command)
│   ├── vision/         # Vision pipeline
│   │   ├── detector.py # ArUco (every frame) + YOLOE-26x (on demand)
│   │   ├── state.py    # In-memory scene state + frame buffers
│   │   └── depth.py    # DA-V3 metric depth (on demand)
│   ├── voice/          # Voice command pipeline
│   │   └── listener.py # Audio capture + mlx-qwen3-asr transcription
│   ├── reasoning/      # CC brain integration
│   │   └── brain.py    # CC session + system prompt
│   └── cli.py          # Typer CLI (legion command)
├── firmware/
│   ├── soccerbot/      # Bot 1: CyberBrick SoccerBot (wheels + kicker)
│   │   ├── boot.py     # Custom boot (skips RC stack)
│   │   ├── main.py     # WiFi + MQTT + motor control
│   │   └── config.json # WiFi creds + broker IP
│   └── tankbot/        # Bot 2: CyberBrick Tank (tracks + shooter)
│       ├── boot.py     # Custom boot (skips RC stack)
│       ├── main.py     # WiFi + MQTT + motor control
│       └── config.json # WiFi creds + broker IP
└── docs/plans/         # Design docs
```

## CLI (`legion`)

```bash
# Server
legion serve start              # foreground
legion serve start --bg         # background
legion serve stop

# Bot control — all args required
legion forward <bot_id> <duration>     # calibrated straight forward
legion backward <bot_id> <duration>    # calibrated straight backward
legion left <bot_id> <duration>        # spin left in place
legion right <bot_id> <duration>       # spin right in place
legion kick <bot_id> <duration>        # activate front kicker servo (Bot 1)
legion shoot <bot_id> <duration>       # activate flywheel shooter (Bot 2)
legion stop <bot_id>                   # emergency stop

# Scene queries (requires server running with vision active)
legion scene state                     # instant — bots only (ArUco)
legion scene state --objects           # + YOLOE-26x detection (~80ms)
legion scene state --depth             # + DA-V3 metric depth (~250ms)
legion scene state --full              # objects + depth (~300ms)

# Snapshot
legion snapshot                        # save raw frame + depth data
```

`legion` is installed globally as an editable uv tool — use `legion` directly, never `uv run legion`.

## Web UI

- **/** — Homepage with links
- **/command** — Command center: video stream + AI brain + voice control
- **/control** — Joystick manual bot control
- **/docs** — Auto-generated API docs

## Bot Calibration

### Bot 1 (SoccerBot, ArUco marker #2)
- **Straight forward**: L=600, R=1000 (left motor stronger)
- **Turns**: inconsistent due to surface/battery — use small increments (0.1-0.15s) and re-check heading
- **Heading**: 0°=up, 90°=right, 180°=down, 270°=left (clockwise)
- **Servo**: 360° replacement working. Kicker on front of bot.
- **Servo burned out (original)**: Replaced 2026-03-15. Raw PWM, no ServosController.

### Bot 2 (Tank, ArUco marker #1)
- **Drive**: tracks, differential steering
- **Straight forward**: L=-1000, R=-1000 (motors inverted vs SoccerBot)
- **Turns**: tracks may turn more consistently than wheels — still use small increments
- **Heading**: same convention as Bot 1 (0°=up, 90°=right, clockwise)
- **Servo**: 360° on S1 (GPIO 3). Flywheel shooter on front. duty(127) = shoot, duty(76) = stop.
- **Shooter**: ~2s = 1 ball fired

## CyberBrick Reference

Primary reference for all CyberBrick firmware work:
- **Official repo**: [CyberBrick-Official/CyberBrick_Controller_Core](https://github.com/CyberBrick-Official/CyberBrick_Controller_Core)
- **API docs**: [makerworld.com/en/cyberbrick/api-doc/](https://makerworld.com/en/cyberbrick/api-doc/)
- **Community WiFi example**: [shuwn/CyberBrick_V7RC_Controller](https://github.com/shuwn/CyberBrick_V7RC_Controller)

## Conventions

- All bot commands are structured JSON — no free-form text parsing
- Bot command schema: `{"action": str, "params": dict}` sent to `legion/bot/{id}/command`
- Use `pathlib.Path` for all file/directory paths
- When using Claude programmatically, use `claude-agent-sdk`, not the `anthropic` package
- Use `uv` for Python dependency management. Run `uv sync` to install, `uv run` to execute.
- Always use `legion` CLI for server management — never raw uvicorn/nohup
