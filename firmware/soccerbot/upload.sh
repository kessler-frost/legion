#!/usr/bin/env bash
# Upload Legion firmware to CyberBrick SoccerBot via mpremote
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Uploading Legion firmware to CyberBrick ==="

# Install umqtt.simple if not present
echo "Installing umqtt.simple..."
mpremote mip install umqtt.simple

# Upload firmware files
echo "Uploading config.json..."
mpremote fs cp "$SCRIPT_DIR/config.json" :config.json

echo "Uploading main.py..."
mpremote fs cp "$SCRIPT_DIR/main.py" :main.py

echo "Uploading boot.py..."
mpremote fs cp "$SCRIPT_DIR/boot.py" :boot.py

echo "Resetting device..."
mpremote reset

echo "=== Done. Bot should connect to WiFi and MQTT. ==="
