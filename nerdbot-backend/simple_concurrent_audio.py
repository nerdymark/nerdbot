#!/usr/bin/env python3
"""
Simple Concurrent Audio Manager using subprocess for guaranteed simultaneous playback
"""

import subprocess
import threading
import logging
import time
import os
from typing import Dict, Optional


class SimpleConcurrentAudio:
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize simple concurrent audio manager
        
        Args:
            max_concurrent: Maximum number of concurrent audio processes
        """
        self.logger = logging.getLogger(__name__)
        self.max_concurrent = max_concurrent
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.process_counter = 0
        self.lock = threading.Lock()
        
        self.logger.info(f"Simple concurrent audio manager initialized: max_concurrent={max_concurrent}")
    
    def play_audio_file(self, file_path: str, volume: float = 1.0) -> Optional[str]:
        """
        Play audio file using subprocess - guaranteed concurrent playback
        
        Args:
            file_path: Path to audio file
            volume: Volume multiplier (0.0 to 1.0) - implemented via amixer if needed
            
        Returns:
            Process ID if successful, None if failed
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Audio file not found: {file_path}")
            return None
        
        # Check concurrent limit
        with self.lock:
            # Clean up finished processes
            self._cleanup_finished_processes()
            
            if len(self.active_processes) >= self.max_concurrent:
                self.logger.warning(f"Maximum concurrent processes ({self.max_concurrent}) reached, skipping playback")
                return None
            
            # Generate process ID
            process_id = f"audio_{self.process_counter}"
            self.process_counter += 1
        
        try:
            # Use a simple shell pipeline for true non-blocking concurrent playback
            # This is the most reliable way to get simultaneous audio
            shell_cmd = f'ffmpeg -i "{file_path}" -f wav - 2>/dev/null | paplay'
            
            process = subprocess.Popen(
                shell_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            with self.lock:
                self.active_processes[process_id] = process
            
            self.logger.info(f"Started audio process {process_id} for {os.path.basename(file_path)} (PID: {process.pid})")
            
            # Start monitoring thread (non-blocking)
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process_id, process),
                daemon=True
            )
            monitor_thread.start()
            
            return process_id
            
        except Exception as e:
            self.logger.error(f"Failed to start audio process for {file_path}: {e}")
            return None
    
    def _monitor_process(self, process_id: str, process: subprocess.Popen):
        """
        Monitor subprocess and clean up when finished
        
        Args:
            process_id: Process identifier
            process: Subprocess object
        """
        try:
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Audio process {process_id} completed successfully")
            else:
                self.logger.error(f"Audio process {process_id} failed with code {process.returncode}: {stderr.decode()}")
                
        except Exception as e:
            self.logger.error(f"Error monitoring process {process_id}: {e}")
        
        finally:
            # Remove from active processes
            with self.lock:
                if process_id in self.active_processes:
                    del self.active_processes[process_id]
    
    def _cleanup_finished_processes(self):
        """Clean up finished processes (called with lock held)"""
        finished_processes = []
        for process_id, process in self.active_processes.items():
            if process.poll() is not None:  # Process has finished
                finished_processes.append(process_id)
        
        for process_id in finished_processes:
            del self.active_processes[process_id]
    
    def stop_process(self, process_id: str) -> bool:
        """
        Stop a specific audio process
        
        Args:
            process_id: Process identifier to stop
            
        Returns:
            True if process was found and stopped
        """
        with self.lock:
            if process_id in self.active_processes:
                process = self.active_processes[process_id]
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()  # Force kill if it doesn't terminate
                    
                    del self.active_processes[process_id]
                    self.logger.info(f"Stopped audio process {process_id}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error stopping process {process_id}: {e}")
            return False
    
    def stop_all_processes(self):
        """Stop all active audio processes"""
        with self.lock:
            active_count = len(self.active_processes)
            if active_count > 0:
                self.logger.info(f"Stopping {active_count} active audio processes")
                for process_id, process in list(self.active_processes.items()):
                    try:
                        process.terminate()
                    except:
                        pass
                
                # Give processes time to terminate, then force kill remaining
                time.sleep(0.5)
                for process_id, process in list(self.active_processes.items()):
                    try:
                        if process.poll() is None:  # Still running
                            process.kill()
                    except:
                        pass
                
                self.active_processes.clear()
    
    def get_active_process_count(self) -> int:
        """Get number of currently active processes"""
        with self.lock:
            self._cleanup_finished_processes()
            return len(self.active_processes)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_all_processes()
        self.logger.info("Simple concurrent audio manager cleaned up")


# Global instance
_simple_audio = None

def get_simple_audio_manager() -> SimpleConcurrentAudio:
    """Get the global simple audio manager instance"""
    global _simple_audio
    if _simple_audio is None:
        _simple_audio = SimpleConcurrentAudio()
    return _simple_audio


if __name__ == "__main__":
    # Test the simple audio manager
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python simple_concurrent_audio.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    manager = get_simple_audio_manager()
    
    print(f"Playing {audio_file} multiple times simultaneously...")
    
    # Play the same sound 5 times with slight delay
    process_ids = []
    for i in range(5):
        pid = manager.play_audio_file(audio_file)
        if pid:
            process_ids.append(pid)
        time.sleep(0.3)
    
    print(f"Started {len(process_ids)} concurrent audio processes")
    
    # Wait for all sounds to finish
    while manager.get_active_process_count() > 0:
        print(f"Active processes: {manager.get_active_process_count()}")
        time.sleep(1)
    
    manager.cleanup()
    print("Test completed!")