#!/usr/bin/env python3
"""
Test script to debug the ultimate concurrent audio manager
"""
import os
import sys
import logging
import time

# Add the current directory to the path
sys.path.insert(0, '/home/mark/nerdbot-backend')

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from ultimate_concurrent_audio import get_ultimate_audio_manager

def test_audio():
    print("Testing Ultimate Concurrent Audio Manager")
    
    # Set environment variables like the Flask service
    os.environ['PULSE_RUNTIME_PATH'] = '/run/user/1000/pulse'
    os.environ['XDG_RUNTIME_DIR'] = '/run/user/1000'
    
    audio_manager = get_ultimate_audio_manager()
    
    test_file = '/home/mark/nerdbot-backend/assets/meme_sounds_converted/applepay.mp3'
    
    print(f"Testing audio file: {test_file}")
    print(f"File exists: {os.path.exists(test_file)}")
    
    if not os.path.exists(test_file):
        print("Test file not found!")
        return False
    
    print("Calling play_audio_file_instant...")
    result = audio_manager.play_audio_file_instant(test_file)
    print(f"Result: {result}")
    
    # Wait a bit to see if process starts
    time.sleep(2)
    
    # Check for paplay processes
    import subprocess
    try:
        ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        paplay_processes = [line for line in ps_result.stdout.split('\n') if 'paplay' in line and 'mp3' in line]
        print(f"Found {len(paplay_processes)} paplay processes:")
        for proc in paplay_processes:
            print(f"  {proc}")
    except Exception as e:
        print(f"Error checking processes: {e}")
    
    return result

if __name__ == "__main__":
    test_audio()