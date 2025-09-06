#!/usr/bin/env python3
"""
Ultimate concurrent audio solution - Fire and forget subprocess approach
"""

import subprocess
import threading
import logging
import time
import os


class UltimateConcurrentAudio:
    def __init__(self):
        """Initialize ultimate concurrent audio manager"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ultimate concurrent audio manager initialized")
    
    def play_audio_file(self, file_path: str) -> bool:
        """
        Play audio file in completely separate process - guaranteed non-blocking
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if process started successfully
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Audio file not found: {file_path}")
            return False
        
        try:
            # Use bash -c to execute paplay with proper environment in one command
            cmd = ['bash', '-c', f'export PULSE_RUNTIME_PATH=/run/user/1000/pulse && export XDG_RUNTIME_DIR=/run/user/1000 && export PULSE_SINK=alsa_output.usb-ANKER_Anker_PowerConf_S330_ACCUDP1E24202697-00.analog-stereo && paplay "{file_path}"']
            
            # Use current environment as base
            env = os.environ.copy()
            
            self.logger.info(f"Running command: {' '.join(cmd)} with audio env")
            
            # Start process completely detached to avoid zombie processes
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True  # This creates a new session for the child process
            )
            
            # Create a separate thread to wait for process completion to avoid zombies
            def wait_for_completion():
                try:
                    stdout, stderr = process.communicate()  # Wait for the process to complete
                    if stderr:
                        self.logger.error(f"Audio process stderr: {stderr.decode()}")
                    if stdout:
                        self.logger.info(f"Audio process stdout: {stdout.decode()}")
                    self.logger.info(f"Audio process completed with return code: {process.returncode}")
                except Exception as e:
                    self.logger.error(f"Process wait error: {e}")
            
            threading.Thread(target=wait_for_completion, daemon=True).start()
            
            self.logger.info(f"Started background audio: {os.path.basename(file_path)} (PID: {process.pid})")
            
            # Immediately return without waiting - true fire-and-forget
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start background audio for {file_path}: {e}")
            return False
    
    def play_audio_file_instant(self, file_path: str) -> bool:
        """
        Ultra-fast audio playback - starts in thread to be extra sure it's non-blocking
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True always (fire and forget)
        """
        def start_audio():
            self.play_audio_file(file_path)
        
        # Start in thread for absolute guarantee of non-blocking
        threading.Thread(target=start_audio, daemon=True).start()
        return True


# Global instance
_ultimate_audio = None

def get_ultimate_audio_manager():
    """Get the global ultimate audio manager instance"""
    global _ultimate_audio
    if _ultimate_audio is None:
        _ultimate_audio = UltimateConcurrentAudio()
    return _ultimate_audio


if __name__ == "__main__":
    # Test rapid fire playback
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python ultimate_concurrent_audio.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    manager = get_ultimate_audio_manager()
    
    print(f"Rapid fire test - playing {audio_file} 10 times as fast as possible...")
    
    start_time = time.time()
    for i in range(10):
        success = manager.play_audio_file_instant(audio_file)
        print(f"Sound {i+1}: {'started' if success else 'failed'}")
        time.sleep(0.05)  # Tiny delay to see the effect
    
    end_time = time.time()
    print(f"Started 10 sounds in {end_time - start_time:.2f} seconds")
    print("All sounds should be playing simultaneously now!")
    
    # Give sounds time to play
    time.sleep(5)
    print("Test completed!")