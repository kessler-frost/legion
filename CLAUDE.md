# Legion

AI-controlled swarm robotics system. Handheld iPhone camera watches CyberBrick robots; a Mac Mini runs the brain.

## Architecture

CC uses the `legion` CLI for everything — observe, think, act.

```
iPhone (Continuity Camera) → Vision Pipeline (YOLO) → Scene State (in-memory)
                                                            ↑
                                              CC (claude-agent-sdk) → legion CLI → MQTT → Bots
                                                            ↑
iPhone (Continuity Camera) → Voice Pipeline (mlx-qwen3-asr) → Transcribed Commands
```

### Hardware

- **Bots**: CyberBrick robots (ESP32-C3, MicroPython, Bambu Lab). Chassis 3D printed on Bambu P1S
- **Brain**: Mac Mini M4 Pro, 48GB RAM
- **Camera**: iPhone held by hand, streaming via Continuity Camera (WiFi/BLE, screen locked)

### Vision Pipeline

Runs as a background thread inside the FastAPI server. YOLO detection on every frame.

- **Input**: iPhone via Continuity Camera (AVFoundation device index 0)
- **Detection**: Ultralytics YOLO11n on every frame
- **Output**: Scene state in memory (bots, objects, positions, bounding boxes)
- **Stream**: WebSocket at `/ws/stream` serves JPEG frames to browser

### Voice Pipeline

Extracts audio from iPhone mic (Continuity Camera), transcribes with mlx-qwen3-asr.

- **Input**: iPhone mic via ffmpeg AVFoundation capture
- **STT**: mlx-qwen3-asr 0.6B (4-bit), ~40ms to transcribe 2s of audio
- **Output**: Transcribed text commands fed to CC

### Reasoning (Claude Code)

Single long-running `claude-agent-sdk` session. CC drives an observe-think-act loop using the `legion` CLI:

1. **Observe**: `legion scene state` / `legion scene describe`
2. **Think**: Reason about what bots should do
3. **Act**: `legion move` / `legion stop` / `legion kick`
4. **Wait**: CC decides interval before next observation
5. **Repeat**

**Latency budget CC must account for:**
- Scene state: updated every frame (~33ms)
- Voice transcription: ~40-100ms
- MQTT command delivery: ~50ms
- Total observe-to-act: ~200ms

### Communication

- **Broker**: Mosquitto running locally on Mac Mini (`localhost:1883`)
- **Command topic**: `legion/bot/{id}/command`
- **Bot firmware action**: `drive` with `{"left": int, "right": int}` per-motor speeds
- **Bot-side MQTT**: `umqtt.simple` (MicroPython)

### Bot Firmware

Custom MicroPython on CyberBrick ESP32-C3. Connects WiFi STA → MQTT broker → receives commands.

- **Actions**: `drive` (per-motor speeds), `stop`, `kick`, `kick_stop`
- **Motor mapping**: Motor 1 = right wheels (positive = forward), Motor 2 = left wheels (inverted)
- **Servo**: Raw PWM on GPIO 3 (ServosController conflicts with easypwm from MotorsController)
- **Upload**: `mpremote` over USB-C, or Arduino Lab for MicroPython

## Tech Stack

- Python 3.12+, managed with `uv` (not pip/venv)
- FastAPI + uvicorn (web server + API)
- Ultralytics YOLO11n (object detection)
- mlx-qwen3-asr (speech-to-text, Apple Silicon optimized)
- OpenCV (`opencv-contrib-python` for camera capture)
- claude-agent-sdk (reasoning)
- MicroPython (bot firmware)
- MQTT (Mosquitto broker, umqtt bot-side)
- Typer (CLI)

## Project Structure

```
legion/
├── brain/
│   ├── api/            # FastAPI server + static frontend
│   │   ├── main.py     # API endpoints, WebSocket stream, vision thread
│   │   └── static/     # HTML pages (home, control, stream)
│   ├── vision/         # YOLO detection pipeline
│   │   ├── detector.py # Camera capture + YOLO inference loop
│   │   └── state.py    # In-memory scene state + frame buffers
│   ├── voice/          # Voice command pipeline
│   │   └── listener.py # Audio capture + mlx-qwen3-asr transcription
│   ├── reasoning/      # CC brain integration
│   │   ├── tools.py    # MCP tools (scene, move, stop, kick)
│   │   └── brain.py    # CC session + async input queue
│   └── cli.py          # Typer CLI (legion command)
├── firmware/
│   └── soccerbot/      # CyberBrick MicroPython bot code
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
legion move <bot_id> <angle> <speed> <duration>
legion kick <bot_id> <duration>
legion stop <bot_id>

# Vision
legion vision start [--bg]
legion vision stop

# Scene queries (requires server running with vision active)
legion scene state              # full JSON
legion scene bots               # bot positions only
legion scene objects            # detected objects only
legion scene describe           # human-readable summary

# Voice
legion listen start [--bg]
legion listen stop
```

Angle mapping: 0°=forward, 90°=right, 180°=backward, 270°=left. Uses differential drive math — angle+speed converted to per-motor speeds. All commands auto-stop after duration.

## Web UI

- **/** — Homepage with links
- **/control** — Joystick + bot commands
- **/stream** — Live camera feed with start/stop capture, raw/YOLO toggle
- **/docs** — Auto-generated API docs

## Bot Calibration

### Bot 1 (SoccerBot)
- **Straight line drift:** Right motor runs ~16% faster. Use angle ~350° (10° left) to go straight. At speed 1000: L=811, R=1158.
- **90° turn:** 0.42s at speed 1000 for both left and right.
- **Servo:** Burned out (needs 360° replacement). Kick commands wired up but no physical servo.
- Surface friction and battery level cause variance.

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
- Always use `legion` CLI for server/vision/listen management — never raw uvicorn/nohup
