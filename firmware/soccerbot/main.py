# firmware/soccerbot/main.py
import json
import time
import network
import uasyncio as asyncio
from umqtt.simple import MQTTClient
from bbl.motors import MotorsController
from bbl.servos import ServosController


motors = None
servos = None


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
    servos.set_angle(1, 90)


def kick():
    print("KICK: spin")
    servos.set_angle(1, 0)
    time.sleep(1)
    print("KICK: stop")
    servos.set_angle(1, 90)
    print("KICK: done")


def on_message(topic, msg):
    print(f"CMD: {msg}")
    cmd = json.loads(msg)
    action = cmd.get("action")
    params = cmd.get("params", {})
    actions = {
        "forward":  lambda: (motors.set_speed(1,  params.get("speed", 1500)), motors.set_speed(2, -params.get("speed", 1500))),
        "backward": lambda: (motors.set_speed(1, -params.get("speed", 1500)), motors.set_speed(2,  params.get("speed", 1500))),
        "left":     lambda: (motors.set_speed(1, params.get("speed", 1500)), motors.stop(2)),
        "right":    lambda: (motors.stop(1), motors.set_speed(2, -params.get("speed", 1500))),
        "stop":     lambda: stop_all(),
        "kick":     lambda: kick(),
    }
    handler = actions.get(action)
    if handler:
        handler()


async def mqtt_loop(client):
    while True:
        client.check_msg()
        await asyncio.sleep_ms(50)


def run():
    global motors, servos
    config = load_config()
    wifi_connect(config["wifi_ssid"], config["wifi_password"])

    # Init motors first (easypwm.init()), then servos after so servo PWM isn't clobbered
    motors = MotorsController()
    print("Motors ready")

    servos = ServosController()
    servos.set_angle(1, 90)
    print("Servos ready")

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
