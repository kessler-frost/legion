# Legion

AI-controlled swarm robotics system. Overhead iPhone camera watches an arena of CyberBrick robots; a Mac Mini runs the brain.

## Architecture

Two-layer design: passive Detection Pipeline + active Claude Code reasoning loop.

```
iPhone (RTSP) → Detection Pipeline → MCP Server ← Claude Code → MQTT → Bots
                (ArUco + YOLO26)      (bridge)     (reasoning)        (CyberBricks)
```

### Hardware

- **Bots**: Up to 4 CyberBrick robots (ESP32-C3, MicroPython, Bambu Lab). ArUco marker on top. Chassis 3D printed on Bambu P1S
- **Brain**: Mac Mini M4 Pro, 48GB RAM
- **Camera**: iPhone mounted overhead, streaming via RTSP app

### Detection Pipeline (passive, continuous)

Single Python process. Maintains scene state in memory.

- **ArUco tracking** (~30fps): `cv2.aruco.ArucoDetector` with `DICT_4X4_50`. Bot position, heading, marker ID
  - Use OpenCV 4.7+ API (`getPredefinedDictionary`, `ArucoDetector`), not deprecated `Dictionary_get`
- **YOLO26** (~30fps): Ultralytics, CoreML export for Neural Engine acceleration. Detects obstacles, objects, arena features
  - YOLOE-26 for open-vocabulary detection (detect anything by text prompt)
- **Scene state**: Combined ArUco + YOLO data as structured Python object, updated every frame

### MCP Server (bridge)

In-process with detection pipeline. Shares scene state via direct memory access.

| Tool | Description |
|------|-------------|
| `get_scene_state` | Bot positions/headings, detected objects, arena bounds (JSON) |
| `get_snapshot` | Latest camera frame as image |
| `send_command` | Publish JSON to `legion/bot/{id}/command` via MQTT |
| `get_telemetry` | Latest telemetry from `legion/bot/{id}/telemetry` |
| `get_arena_config` | Arena dimensions, marker-to-bot ID mapping |

### Reasoning (Claude Code, active, drives the loop)

Single long-running `claude-agent-sdk` session. CC drives an observe-think-act loop:

1. **Observe**: `get_scene_state` + optionally `get_snapshot`
2. **Think**: Reason about what bots should do
3. **Act**: `send_command` per bot
4. **Wait**: CC decides interval before next observation
5. **Repeat**

CC controls all timing — no external triggering, no cooldown logic, no feedback loops.

### Communication

- **Broker**: Mosquitto running locally on Mac Mini
- **Command topic**: `legion/bot/{id}/command` — JSON `{"action": str, "params": dict}`
- **Telemetry topic**: `legion/bot/{id}/telemetry`
- **Bot-side MQTT**: `umqtt.simple` or `umqtt.robust` (MicroPython)

### Video Pipeline

iPhone RTSP stream → `cv2.VideoCapture(rtsp_url)` → ArUco + YOLO26 (same frames)

## Tech Stack

- Python 3.12+
- OpenCV (`opencv-contrib-python` for ArUco)
- Ultralytics YOLO26 (CoreML export)
- claude-agent-sdk (reasoning)
- MicroPython (bot firmware)
- MQTT (Mosquitto broker, umqtt bot-side)

## Project Structure

```
legion/
├── brain/              # Mac Mini orchestrator
│   ├── detection/      # ArUco + YOLO26 pipeline
│   ├── mcp/            # MCP server (bridge to CC)
│   └── comms/          # MQTT client
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
