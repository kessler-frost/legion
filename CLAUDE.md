# Legion

AI-controlled swarm robotics system. Overhead iPhone camera watches an arena of CyberBrick robots; a Mac Mini runs the brain.

## Architecture

### Hardware

- **Bots**: Up to 4 CyberBrick robots (ESP32-C3, MicroPython, Bambu Lab). Chassis 3D printed on Bambu P1S
- **Brain**: Mac Mini M4 Pro, 48GB RAM
- **Camera**: iPhone mounted overhead, streaming via RTSP app

### Two-Loop System

**Fast Loop (~30fps)** — Computer Vision
- ArUco marker tracking: `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)` — 50 unique IDs, each bot gets one
  - Use `cv2.aruco.ArucoDetector` class (OpenCV 4.7+ API, not the deprecated `Dictionary_get` / `DetectorParameters_create`)
- YOLO obstacle detection: Ultralytics, `device="mps"` for Apple Silicon GPU. CoreML export available for production

**Slow Loop (~1fps)** — Scene Understanding (evaluating)
- Gemini Live API: real-time video via WebSocket, could combine vision + reasoning in one stream
- Qwen3-VL local via Ollama: 8B (`qwen3-vl:8b`, ~6GB) or 32B (`qwen3-vl:32b`, ~21GB)
- Qwen3.5 local via Ollama: native multimodal, 27B (`qwen3.5:27b`, ~17GB) — vision built-in, no separate VL variant needed
- Moondream 2B: smallest/fastest local option (~2GB)

### Reasoning Engine (evaluating)

- Claude API via `claude-agent-sdk` (Python SDK for Claude Code CLI)
- Gemini Live API (combined vision + reasoning in one WebSocket stream)

### Communication

- Bots receive JSON commands over MQTT WiFi
- Topic: `legion/bot/{id}/command`
- Bot-side MQTT: `umqtt.simple` or `umqtt.robust` (MicroPython)
- Broker-side: Paho MQTT or Mosquitto

### Video Pipeline

iPhone RTSP stream → `cv2.VideoCapture(rtsp_url)` → fast loop + slow loop (downsampled)

## Tech Stack

- Python 3.12+
- OpenCV (`opencv-contrib-python` for ArUco)
- Ultralytics (YOLO)
- MicroPython (bot firmware)
- MQTT (paho-mqtt broker-side, umqtt bot-side)

## Project Structure

```
legion/
├── brain/              # Mac Mini orchestrator
│   ├── vision/         # Fast loop — ArUco tracking + YOLO
│   ├── scene/          # Slow loop — VLM scene understanding
│   ├── reasoning/      # Decision engine (Claude or Gemini)
│   └── comms/          # MQTT command dispatch
├── firmware/           # CyberBrick MicroPython bot code
├── arena/              # Arena config, marker-to-bot ID mapping
└── docs/plans/         # Design docs
```

## Conventions

- ArUco marker IDs map 1:1 to MQTT bot IDs
- All bot commands and responses are structured JSON — no free-form text parsing
- Bot command schema: `{"action": str, "params": dict}` sent to `legion/bot/{id}/command`
- Bot telemetry publishes to `legion/bot/{id}/telemetry`
- Use `pathlib.Path` for all file/directory paths
- When using Claude programmatically, use `claude-agent-sdk`, not the `anthropic` package
