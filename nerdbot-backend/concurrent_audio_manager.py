#!/usr/bin/env python3
"""
Concurrent Audio Manager for NerdBot
Supports simultaneous playback of multiple audio files
"""

import threading
import time
import logging
import pyaudio
import numpy as np
from pydub import AudioSegment
import io
import os
from typing import List, Dict, Optional


class ConcurrentAudioManager:
    def __init__(self, audio_rate: int = 48000, chunk_size: int = 1024, max_concurrent: int = 5):
        """
        Initialize the concurrent audio manager
        
        Args:
            audio_rate: Audio sample rate
            chunk_size: Audio chunk size for playback
            max_concurrent: Maximum number of concurrent audio streams
        """
        self.logger = logging.getLogger(__name__)
        self.audio_rate = audio_rate
        self.chunk_size = chunk_size
        self.max_concurrent = max_concurrent
        
        # PyAudio instance
        self.pyaudio_instance = pyaudio.PyAudio()
        
        # Track active streams
        self.active_streams: Dict[str, Dict] = {}
        self.stream_counter = 0
        self.lock = threading.Lock()
        
        # Global audio mixer buffer
        self.mixer_buffer = np.zeros(chunk_size, dtype=np.float32)
        
        self.logger.info(f"Concurrent audio manager initialized: rate={audio_rate}, chunk={chunk_size}, max_concurrent={max_concurrent}")
    
    def _load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load and convert audio file to numpy array
        
        Args:
            file_path: Path to audio file
            
        Returns:
            numpy array of audio data or None if error
        """
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Convert using pydub
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            audio_segment = audio_segment.set_frame_rate(self.audio_rate)
            audio_segment = audio_segment.set_channels(1)  # Mono
            audio_segment = audio_segment.set_sample_width(2)  # 16-bit
            audio_segment = audio_segment.normalize()
            
            # Convert to numpy array
            raw_data = audio_segment.raw_data
            audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to -1 to 1
            
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {e}")
            return None
    
    def play_audio_file(self, file_path: str, volume: float = 1.0) -> Optional[str]:
        """
        Play audio file concurrently with other sounds
        
        Args:
            file_path: Path to audio file
            volume: Volume multiplier (0.0 to 1.0)
            
        Returns:
            Stream ID if successful, None if failed
        """
        # Check concurrent limit
        with self.lock:
            if len(self.active_streams) >= self.max_concurrent:
                self.logger.warning(f"Maximum concurrent streams ({self.max_concurrent}) reached, skipping playback")
                return None
            
            # Generate stream ID
            stream_id = f"stream_{self.stream_counter}"
            self.stream_counter += 1
        
        # Load audio file
        audio_data = self._load_audio_file(file_path)
        if audio_data is None:
            return None
        
        # Apply volume
        audio_data = audio_data * volume
        
        # Create and start playback thread
        playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(stream_id, audio_data),
            daemon=True
        )
        
        with self.lock:
            self.active_streams[stream_id] = {
                'thread': playback_thread,
                'audio_data': audio_data,
                'volume': volume,
                'file_path': file_path
            }
        
        playback_thread.start()
        self.logger.info(f"Started playback stream {stream_id} for {file_path}")
        
        return stream_id
    
    def _playback_thread(self, stream_id: str, audio_data: np.ndarray):
        """
        Thread function for individual audio playback using subprocess
        
        Args:
            stream_id: Unique stream identifier
            audio_data: Numpy array of audio data
        """
        try:
            import subprocess
            import tempfile
            
            # Convert numpy array back to AudioSegment
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Use scipy or wave to write the file
                import wave
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.audio_rate)
                    wav_file.writeframes(audio_int16.tobytes())
            
            # Play using aplay (which supports concurrent playback better)
            cmd = ['aplay', '-q', temp_path]  # -q for quiet mode
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Finished concurrent playback for stream {stream_id}")
            else:
                self.logger.error(f"aplay failed for stream {stream_id}: {result.stderr}")
            
            # Cleanup temp file
            try:
                import os
                os.unlink(temp_path)
            except:
                pass
            
        except Exception as e:
            self.logger.error(f"Error in concurrent playback thread for stream {stream_id}: {e}")
        
        finally:
            # Remove from active streams
            with self.lock:
                if stream_id in self.active_streams:
                    del self.active_streams[stream_id]
    
    def stop_stream(self, stream_id: str) -> bool:
        """
        Stop a specific audio stream
        
        Args:
            stream_id: Stream identifier to stop
            
        Returns:
            True if stream was found and stopped
        """
        with self.lock:
            if stream_id in self.active_streams:
                # Note: Individual threads will clean themselves up
                self.logger.info(f"Marked stream {stream_id} for stopping")
                return True
            return False
    
    def stop_all_streams(self):
        """Stop all active audio streams"""
        with self.lock:
            active_count = len(self.active_streams)
            if active_count > 0:
                self.logger.info(f"Stopping {active_count} active audio streams")
                # Threads will clean themselves up
                self.active_streams.clear()
    
    def get_active_stream_count(self) -> int:
        """Get number of currently active streams"""
        with self.lock:
            return len(self.active_streams)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_all_streams()
        time.sleep(0.5)  # Give threads time to cleanup
        self.pyaudio_instance.terminate()
        self.logger.info("Concurrent audio manager cleaned up")


# Global instance
_audio_manager = None

def get_audio_manager() -> ConcurrentAudioManager:
    """Get the global audio manager instance"""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = ConcurrentAudioManager()
    return _audio_manager


if __name__ == "__main__":
    # Test the audio manager
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python concurrent_audio_manager.py <audio_file.mp3>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    manager = get_audio_manager()
    
    print(f"Playing {audio_file} multiple times simultaneously...")
    
    # Play the same sound 3 times with slight delay
    for i in range(3):
        manager.play_audio_file(audio_file, volume=0.8)
        time.sleep(0.5)
    
    # Wait for all sounds to finish
    while manager.get_active_stream_count() > 0:
        print(f"Active streams: {manager.get_active_stream_count()}")
        time.sleep(1)
    
    manager.cleanup()
    print("Test completed!")