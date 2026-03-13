# firmware/soccerbot/main.py
import json
import time
import network
import uasyncio as asyncio
from umqtt.simple import MQTTClient
from bbl.motors import MotorsController
from bbl.servos import ServosController


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


motors = MotorsController()
servos = ServosController()

ACTIONS = {
    "forward":  lambda p: (motors.set_speed(1,  p.get("speed", 1500)), motors.set_speed(2,  p.get("speed", 1500))),
    "backward": lambda p: (motors.set_speed(1, -p.get("speed", 1500)), motors.set_speed(2, -p.get("speed", 1500))),
    "left":     lambda p: (motors.set_speed(1, -p.get("speed", 1500)), motors.set_speed(2,  p.get("speed", 1500))),
    "right":    lambda p: (motors.set_speed(1,  p.get("speed", 1500)), motors.set_speed(2, -p.get("speed", 1500))),
    "stop":     lambda p: (motors.set_speed(1, 0), motors.set_speed(2, 0)),
}


async def kick():
    servos.set_angle(1, 180)
    await asyncio.sleep(0.3)
    servos.set_angle(1, 0)


def on_message(topic, msg):
    cmd = json.loads(msg)
    action = cmd.get("action")
    params = cmd.get("params", {})
    if action == "kick":
        asyncio.create_task(kick())
    elif action in ACTIONS:
        ACTIONS[action](params)


async def mqtt_loop(client):
    while True:
        client.check_msg()
        await asyncio.sleep_ms(50)


def run():
    config = load_config()
    wifi_connect(config["wifi_ssid"], config["wifi_password"])

    client = MQTTClient(
        client_id=f"legion_bot_{config['bot_id']}",
        server=config["mqtt_broker"],
        port=config["mqtt_port"],
    )
    client.set_callback(on_message)
    client.connect()
    topic = f"legion/bot/{config['bot_id']}/command"
    client.subscribe(topic.encode())
    print(f"Subscribed to {topic}")

    asyncio.run(mqtt_loop(client))
