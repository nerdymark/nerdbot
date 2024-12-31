"""
Basic watchdog script to monitor the bot and restart it if it crashes.
Try to re-connect WiFi if it's down.
"""
import os
import time
import subprocess
import logging
import json
import requests


CONFIG_FILE = '/home/mark/nerdbot-scripts/config.json'
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

LOG_FILE = config.get('log_file')
logging.basicConfig(level=logging.INFO, filename=LOG_FILE)
logging.info("Logging to file %s", LOG_FILE)

TIMEOUT = config.get('watchdog_timeout')
DEFAULT_GATEWAY = config.get('default_gateway')
VITE_REACT_HEALTH_URL = config.get('health_url_vite_react')
FLASK_HEALTH_URL = config.get('health_url_flask')
TTS_HEALTH_URL = config.get('health_url_coqui_tts')
DARKICE_HEALTH_URL = config.get('health_url_darkice')
VITE_REACT_STARTUP_SCRIPT = config.get('vite_react_startup_script')
FLASK_STARTUP_SCRIPT = config.get('flask_startup_script')
TTS_STARTUP_SCRIPT = config.get('coqui_tts_startup_script')
DARKICE_STARTUP_SCRIPT = config.get('darkice_startup_script')
BOT_RUNAS_USER = config.get('runas_user')


def test_node_health():
    try:
        response = requests.get(VITE_REACT_HEALTH_URL)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False


def test_flask_health():
    try:
        response = requests.get(FLASK_HEALTH_URL)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False


def test_tts_health():
    try:
        response = requests.get(TTS_HEALTH_URL)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False


def test_darkice_health():
    try:
        response = requests.get(DARKICE_HEALTH_URL)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False


def ping_host(host):
    return os.system("ping -c 1 -W 1 " + host + " > /dev/null") == 0


def restart_bot():
    """
    Reboot the whole system
    """
    print("Restarting bot...")
    subprocess.call(["reboot"])


def restart_wifi():
    """
    issue wpa_cli -i wlan0 reconfigure
    """
    print("Restarting WiFi...")
    subprocess.call(["wpa_cli", "-i", "wlan0", "reconfigure"])
    time.sleep(10)


logger = logging.getLogger(__name__)
logger.info("Watchdog started")
print("Watchdog started")
# Wait 10 seconds for the bot to start
time.sleep(10)
while True:
    if not ping_host(DEFAULT_GATEWAY):
        logger.warning("WiFi is down, waiting for %d seconds", TIMEOUT)
        time.sleep(TIMEOUT)
        if not ping_host(DEFAULT_GATEWAY):
            logger.warning("WiFi is still down, restarting WiFi")
            restart_wifi()
        time.sleep(TIMEOUT)
        if not ping_host(DEFAULT_GATEWAY):
            logger.warning("WiFi is still down, restarting bot")
            restart_bot()
    # else:
    #     logger.info("WiFi is up")
    if not test_node_health():
        logger.warning("Vite React is down, restarting Node as %s", BOT_RUNAS_USER)
        # Restart Vite React as BOT_RUNAS_USER in a detached process
        subprocess.call(["sudo", "-u", BOT_RUNAS_USER, VITE_REACT_STARTUP_SCRIPT, "&"])
    # else:
    #     logger.info("Vite React is up")
    if not test_flask_health():
        logger.warning("Flask is down, restarting Flask as %s", BOT_RUNAS_USER)
        # Restart Flask as BOT_RUNAS_USER
        subprocess.call(["sudo", "-u", BOT_RUNAS_USER, FLASK_STARTUP_SCRIPT, "&"])

    if not test_darkice_health():
        logger.warning("Darkice is down, restarting Darkice as root")
        # Restart Darkice as BOT_RUNAS_USER
        subprocess.call([DARKICE_STARTUP_SCRIPT, "&"])


    # Delete the log file is it's too big
    if os.path.getsize(LOG_FILE) > 1000000:
        with open(LOG_FILE, 'w') as f:
            f.write("")
            f.close()
        logger.info("Log file cleared")
    time.sleep(10)


