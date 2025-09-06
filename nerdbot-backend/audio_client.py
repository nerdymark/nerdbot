#!/usr/bin/env python3
"""
Client for communicating with the audio service
"""
import os
import json
import time
import uuid
from pathlib import Path

AUDIO_REQUEST_DIR = '/tmp/nerdbot_audio'

def ensure_request_dir():
    """Ensure the audio request directory exists"""
    Path(AUDIO_REQUEST_DIR).mkdir(exist_ok=True)

def play_random_meme():
    """Request a random meme sound"""
    print(f"[DEBUG] play_random_meme called")  # Debug logging
    ensure_request_dir()
    
    request = {
        'type': 'random',
        'timestamp': time.time()
    }
    
    request_file = Path(AUDIO_REQUEST_DIR) / f"random_{uuid.uuid4().hex[:8]}.json"
    print(f"[DEBUG] Creating request file: {request_file}")  # Debug logging
    
    with open(request_file, 'w') as f:
        json.dump(request, f)
    
    print(f"[DEBUG] Request file created successfully")  # Debug logging
    return True

def play_specific_sound(file_path):
    """Request a specific sound file"""
    print(f"[DEBUG] play_specific_sound called with: {file_path}")  # Debug logging
    ensure_request_dir()
    
    request = {
        'type': 'specific',
        'file_path': file_path,
        'timestamp': time.time()
    }
    
    request_file = Path(AUDIO_REQUEST_DIR) / f"specific_{uuid.uuid4().hex[:8]}.json"
    print(f"[DEBUG] Creating request file: {request_file}")  # Debug logging
    
    with open(request_file, 'w') as f:
        json.dump(request, f)
    
    print(f"[DEBUG] Request file created successfully")  # Debug logging
    return True

if __name__ == "__main__":
    # Test the client
    print("Testing audio client...")
    result = play_random_meme()
    print(f"Random meme request: {result}")
    time.sleep(2)
    print("Test completed")