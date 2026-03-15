# firmware/soccerbot/main.py
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


def kick_start():
    servo_pwm.duty(127)


def kick_stop():
    servo_pwm.duty(76)


def on_message(topic, msg):
    print(f"CMD: {msg}")
    cmd = json.loads(msg)
    action = cmd.get("action")
    params = cmd.get("params", {})
    actions = {
        "drive":     lambda: (motors.set_speed(1, params.get("right", 0)), motors.set_speed(2, -params.get("left", 0))),
        "stop":      lambda: stop_all(),
        "kick":      lambda: kick_start(),
        "kick_stop": lambda: kick_stop(),
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
