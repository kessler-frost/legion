# Legion

Swarm robotics with [Cyberbricks](https://makerworld.com/en/cyberbrick), orchestrated by [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

https://github.com/kessler-frost/legion/raw/main/docs/demo.mp4

> In this demo, I told the AI agent to "avenge Baymax." It figured out what that meant on its own, located the targets, and coordinated both robots to take them out.

## About

Legion lets an AI coding agent control physical robots through natural language. You talk, the AI sees the play area through a webcam, decides what to do, and sends commands to the bots over WiFi.

This is a collaboration between me and [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Most commits are co-authored by Claude. I haven't polished every edge case so there will be bugs. If you run into something, [open an issue](https://github.com/kessler-frost/legion/issues).

The architecture is agent-agnostic. The AI interacts with bots entirely through the `legion` CLI, so you could swap Claude Code for [OpenCode](https://github.com/anomalyco/opencode) or any agent that can run shell commands.

```
┌─────────────┐ ┌─────────────────┐ ┌──────────┐ ┌────────────┐ ┌──────┐ ┌────────┐
│             │ │                 │ │          │ │            │ │      │ │        │
│  USB Webcam ├►│ Computer Vision ├►│ AI Agent ├►│ legion CLI ├►│ MQTT ├►│ Robots │
│             │ │                 │ │          │ │            │ │      │ │        │
└─────────────┘ └─────────────────┘ └──────────┘ └────────────┘ └──────┘ └────────┘
┌─────────────┐ ┌─────────────────┐       ▲
│             │ │                 │       │
│ Browser Mic ├►│  Speech-to-Text ├───────┘
│             │ │                 │
└─────────────┘ └─────────────────┘
```

### Vision pipeline

```
                   Camera Frame
                        |
           +------------+------------+
           |            |            |
           v            v            v
     ArUco Detect   YOLOE-26x   Depth Anything 3
     every frame    on demand      on demand
           |            |            |
           v            v            v
     Bot IDs +    4,585 class   Metric depth
     positions    detection     in meters
     headings
           |            |            |
           +------------+------------+
                        |
                        v
                 Scene State JSON
```

- **ArUco markers** on each robot, detected every frame (<1ms). Gives bot ID, position, and heading.
- **YOLOE-26x** object detection on demand (~80ms). 4,585 classes from the [RAM++](https://github.com/xinyu1205/recognize-anything) tag set.
- **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)** metric depth on demand (~250ms).

Voice goes through the browser mic to [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr) (local STT on Apple Silicon, ~40ms).

## Prerequisites

**Hardware:** One or more [CyberBrick](https://makerworld.com/en/cyberbrick) kits (SoccerBot/Tank tested), [ArUco markers](https://chev.me/arucogen/) (4x4_50) on each bot, a USB webcam overhead, and WiFi.

**Software:** [uv](https://docs.astral.sh/uv/), [Mosquitto](https://mosquitto.org/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (with [API key](https://console.anthropic.com/)), [Arduino Lab for MicroPython](https://labs.arduino.cc/en/labs/micropython).

## Getting started

### 1. Install and clone

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv (manages Python for you)
brew install mosquitto claude-code                 # MQTT broker + Claude Code

git clone https://github.com/kessler-frost/legion.git && cd legion
uv sync                          # installs Python 3.12 and all deps
uv tool install --editable .     # makes `legion` available globally
```

Download the YOLOE-26x ONNX model (~260MB) into the project root:

```bash
curl -L -o yoloe-26x-seg-pf.onnx https://github.com/kessler-frost/legion/releases/download/v0.1.0/yoloe-26x-seg-pf.onnx
```

> Pinned to Python 3.12 because `open3d` (a depth model dependency) doesn't support 3.13 yet.

### 2. Flash firmware to your bots

Use [Arduino Lab for MicroPython](https://labs.arduino.cc/en/labs/micropython):

1. Copy `firmware/soccerbot/config.example.json` to `config.json`, fill in your WiFi SSID, password, and computer's IP
2. Connect bot via USB-C, upload `boot.py`, `main.py`, and `config.json`

Test in the REPL:

```python
import machine
machine.reset()  # restarts the bot, runs boot.py then main.py
```

### 3. Run

```bash
mosquitto -d          # start MQTT broker
legion serve start    # start server + vision + AI agent
```

Open [localhost:8000/command](http://localhost:8000/command) and start talking.

## CLI reference

```bash
legion serve start [--bg]       # start server (--bg for background)
legion serve stop               # stop background server

legion forward <id> <duration>  # drive forward
legion backward <id> <duration> # drive backward
legion left <id> <duration>     # spin left
legion right <id> <duration>    # spin right
legion kick <id> <duration>     # kicker servo (Bot 1)
legion shoot <id> <duration>    # shooter servo (Bot 2)
legion stop <id>                # emergency stop

legion scene state              # bot positions + headings (instant)
legion scene state --objects    # + object detection (~80ms)
legion scene state --depth      # + depth estimation (~250ms)
legion scene state --full       # everything (~300ms)
```

## Web UI

| URL | What |
|-----|------|
| `/` | Home |
| `/command` | Command center: camera + AI chat + voice |
| `/control` | Manual joystick control |
| `/docs` | API docs |

## Adding a new bot

1. Print an ArUco marker (4x4_50) and stick it on top of a CyberBrick kit
2. Copy an existing firmware folder, set a new `bot_id` in `config.example.json`
3. Add motor calibration + marker mapping in `brain/cli.py`
4. Flash via Arduino Lab for MicroPython

## Resources

- [CyberBrick Official Repo](https://github.com/CyberBrick-Official/CyberBrick_Controller_Core)
- [CyberBrick API Docs](https://makerworld.com/en/cyberbrick/api-doc/)

## License

[Apache License 2.0](LICENSE)
