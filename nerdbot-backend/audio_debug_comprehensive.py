#!/usr/bin/env python3
"""
Comprehensive audio debugging script
"""
import os
import sys
import subprocess
import time
import threading
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_direct_paplay():
    """Test paplay directly"""
    print("\n=== TESTING DIRECT PAPLAY ===")
    test_file = '/home/mark/nerdbot-backend/assets/meme_sounds_converted/applepay.mp3'
    
    # Set environment like Flask service
    env = os.environ.copy()
    env['PULSE_RUNTIME_PATH'] = '/run/user/1000/pulse'
    env['XDG_RUNTIME_DIR'] = '/run/user/1000'
    env['PULSE_SINK'] = 'alsa_output.usb-ANKER_Anker_PowerConf_S330_ACCUDP1E24202697-00.analog-stereo'
    
    print(f"Testing file: {test_file}")
    print(f"Environment: PULSE_RUNTIME_PATH={env.get('PULSE_RUNTIME_PATH')}")
    print(f"Environment: XDG_RUNTIME_DIR={env.get('XDG_RUNTIME_DIR')}")
    print(f"Environment: PULSE_SINK={env.get('PULSE_SINK')}")
    
    try:
        print("Running paplay with timeout=5 seconds...")
        result = subprocess.run(
            ['paplay', test_file],
            env=env,
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Timeout - process was killed")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_subprocess_popen():
    """Test subprocess.Popen like the audio manager"""
    print("\n=== TESTING SUBPROCESS POPEN (LIKE AUDIO MANAGER) ===")
    test_file = '/home/mark/nerdbot-backend/assets/meme_sounds_converted/applepay.mp3'
    
    # Set environment like Flask service
    env = os.environ.copy()
    env['PULSE_RUNTIME_PATH'] = '/run/user/1000/pulse'
    env['XDG_RUNTIME_DIR'] = '/run/user/1000'
    env['PULSE_SINK'] = 'alsa_output.usb-ANKER_Anker_PowerConf_S330_ACCUDP1E24202697-00.analog-stereo'
    
    try:
        print("Creating subprocess.Popen...")
        process = subprocess.Popen(
            ['paplay', test_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env
        )
        
        print(f"Process started with PID: {process.pid}")
        
        # Wait a bit and check if it's still running
        time.sleep(0.5)
        poll_result = process.poll()
        print(f"Poll after 0.5s: {poll_result}")
        
        # Wait for completion
        try:
            stdout, stderr = process.communicate(timeout=10)
            print(f"Process finished with return code: {process.returncode}")
            if stderr:
                print(f"Stderr: {stderr.decode()}")
            return process.returncode == 0
        except subprocess.TimeoutExpired:
            print("Process timeout - killing")
            process.kill()
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_flask_api():
    """Test the Flask API"""
    print("\n=== TESTING FLASK API ===")
    import requests
    try:
        response = requests.post('http://10.0.1.204:5000/api/meme_sound/random', timeout=5)
        print(f"API Response code: {response.status_code}")
        print(f"API Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"API Exception: {e}")
        return False

def monitor_processes():
    """Monitor for paplay processes"""
    print("\n=== MONITORING PROCESSES ===")
    
    def check_processes():
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            paplay_lines = [line for line in result.stdout.split('\n') if 'paplay' in line and 'mp3' in line]
            if paplay_lines:
                print(f"Found {len(paplay_lines)} paplay processes:")
                for line in paplay_lines:
                    print(f"  {line}")
            else:
                print("No paplay processes found")
        except Exception as e:
            print(f"Error checking processes: {e}")
    
    # Start monitoring in a thread
    def monitor():
        for i in range(10):
            print(f"Check {i+1}:")
            check_processes()
            time.sleep(0.5)
    
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    return monitor_thread

if __name__ == "__main__":
    print("COMPREHENSIVE AUDIO DEBUG TEST")
    print("=" * 50)
    
    # Start process monitoring
    monitor_thread = monitor_processes()
    
    # Test 1: Direct paplay
    direct_result = test_direct_paplay()
    time.sleep(1)
    
    # Test 2: Subprocess Popen
    popen_result = test_subprocess_popen()
    time.sleep(1)
    
    # Test 3: Flask API
    api_result = test_flask_api()
    
    # Wait for monitor to finish
    monitor_thread.join()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Direct paplay: {'PASS' if direct_result else 'FAIL'}")
    print(f"Subprocess Popen: {'PASS' if popen_result else 'FAIL'}")
    print(f"Flask API: {'PASS' if api_result else 'FAIL'}")
    print("=" * 50)