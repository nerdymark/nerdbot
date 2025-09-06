#!/usr/bin/env python3
"""
Simple audio service that monitors for audio requests and plays them
This runs in the user session and has proper audio access
"""
import os
import time
import subprocess
import json
import logging
from pathlib import Path
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_REQUEST_DIR = '/tmp/nerdbot_audio'
MEME_SOUNDS_DIR = '/home/mark/nerdbot-backend/assets/meme_sounds_converted'

def ensure_audio_request_dir():
    """Ensure the audio request directory exists"""
    Path(AUDIO_REQUEST_DIR).mkdir(exist_ok=True)
    logger.info(f"Audio request directory: {AUDIO_REQUEST_DIR}")

def get_meme_sounds():
    """Get list of available meme sounds"""
    if not os.path.exists(MEME_SOUNDS_DIR):
        logger.error(f"Meme sounds directory not found: {MEME_SOUNDS_DIR}")
        return []
    
    sounds = [f for f in os.listdir(MEME_SOUNDS_DIR) 
              if f.endswith(('.mp3', '.wav', '.ogg'))]
    logger.info(f"Found {len(sounds)} meme sounds")
    return sounds

def play_audio_file(file_path):
    """Play an audio file using paplay"""
    try:
        # Set up proper environment
        env = os.environ.copy()
        env['PULSE_RUNTIME_PATH'] = '/run/user/1000/pulse'
        env['XDG_RUNTIME_DIR'] = '/run/user/1000'
        env['PULSE_SINK'] = 'alsa_output.usb-ANKER_Anker_PowerConf_S330_ACCUDP1E24202697-00.analog-stereo'
        
        logger.info(f"Playing audio: {os.path.basename(file_path)}")
        
        # Use subprocess.Popen for fire-and-forget playback
        process = subprocess.Popen(
            ['paplay', file_path],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        
        logger.info(f"Audio process started with PID: {process.pid}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to play audio {file_path}: {e}")
        return False

def process_audio_requests():
    """Process pending audio requests"""
    request_files = list(Path(AUDIO_REQUEST_DIR).glob('*.json'))
    
    for request_file in request_files:
        try:
            # Read the request
            with open(request_file, 'r') as f:
                request = json.load(f)
            
            # Process the request
            if request['type'] == 'random':
                sounds = get_meme_sounds()
                if sounds:
                    sound = random.choice(sounds)
                    sound_path = os.path.join(MEME_SOUNDS_DIR, sound)
                    play_audio_file(sound_path)
                    logger.info(f"Played random sound: {sound}")
            
            elif request['type'] == 'specific':
                sound_path = request['file_path']
                if os.path.exists(sound_path):
                    play_audio_file(sound_path)
                    logger.info(f"Played specific sound: {os.path.basename(sound_path)}")
                else:
                    logger.error(f"Sound file not found: {sound_path}")
            
            # Remove processed request
            request_file.unlink()
            logger.info(f"Processed and removed request: {request_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process request {request_file}: {e}")
            # Remove failed request to prevent infinite loop
            try:
                request_file.unlink()
            except:
                pass

def main():
    """Main audio service loop"""
    logger.info("Starting NerdBot Audio Service")
    ensure_audio_request_dir()
    
    # Get initial sound count
    sounds = get_meme_sounds()
    logger.info(f"Audio service ready with {len(sounds)} sounds")
    
    try:
        while True:
            process_audio_requests()
            time.sleep(0.1)  # Check for requests every 100ms
            
    except KeyboardInterrupt:
        logger.info("Audio service stopped by user")
    except Exception as e:
        logger.error(f"Audio service error: {e}")

if __name__ == "__main__":
    main()