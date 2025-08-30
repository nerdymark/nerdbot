"""
Fan Control script for Raspberry Pi 5
Inspired by https://gist.github.com/James-Ansley/32f72729487c8f287a801abcc7a54f38

Run as root
sudo python3 fan-control.py
"""
from enum import Enum
import time
import logging
import json


TEMP_PATH = "/sys/devices/virtual/thermal/thermal_zone0/temp"
FAN_PATH = "/sys/class/thermal/cooling_device0/cur_state"
CONFIG_FILE = '/home/mark/nerdbot-scripts/config.json'
FAN_SPEED = None

# Load config
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}
    print("Config file not found")
except json.JSONDecodeError:
    config = {}
    print("Config file is not a valid JSON")

LOG_FILE = config.get('log_file')
if LOG_FILE:
    logging.basicConfig(level=logging.INFO, filename=LOG_FILE)
    # logging.info("Logging to file %s", LOG_FILE)
else:
    logging.basicConfig(level=logging.INFO)

class FanSpeed(Enum):
    """
    Fan speed levels
    """
    OFF = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FULL = 4


def main():
    """
    Main loop
    """
    start = time.time()
    # logger = logging.getLogger(main.__name__)
    while time.time() - start < 59:
        temp = get_temp()
        if temp > 70:            
            speed = FanSpeed.FULL
        elif temp > 60:
            speed = FanSpeed.FULL
        elif temp > 55:
            speed = FanSpeed.HIGH
        elif temp > 45:
            speed = FanSpeed.HIGH
        else:
            speed = FanSpeed.MEDIUM
        fan_speed = get_fan_speed()
        if speed != fan_speed:
            set_fan_speed(speed)
        time.sleep(2)


def get_temp() -> int:
    """
    Get the temperature of the CPU
    """
    with open(TEMP_PATH, "r", encoding="utf-8") as tf:
        data = tf.read()
    return int(data) // 1000


def set_fan_speed(speed: FanSpeed):
    """
    Set the fan speed
    """
    logger = logging.getLogger(set_fan_speed.__name__)
    with open(FAN_PATH, "w", encoding="utf-8") as fs:
        logger.info("Setting fan speed to %s", speed.name)
        fs.write(str(speed.value))
    return speed


def get_fan_speed() -> FanSpeed:
    """
    Get the current fan speed
    """
    with open(FAN_PATH, "r", encoding="utf-8") as ff:
        data = ff.read()
    return FanSpeed(int(data))

if __name__ == "__main__":
    # logging.info("Fan Control started")
    main()
