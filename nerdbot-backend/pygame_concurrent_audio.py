#!/usr/bin/env python3
"""
Pygame-based concurrent audio manager for true simultaneous sound playback
"""

import pygame
import threading
import logging
import time
import os
from typing import Dict, Optional
import io
from pydub import AudioSegment


class PygameConcurrentAudio:
    def __init__(self, max_channels: int = 8):
        """
        Initialize pygame mixer for concurrent audio playback
        
        Args:
            max_channels: Maximum number of concurrent audio channels
        """
        self.logger = logging.getLogger(__name__)
        self.max_channels = max_channels
        self.initialized = False
        
        try:
            # Initialize pygame mixer with settings optimized for concurrent playback
            pygame.mixer.pre_init(
                frequency=44100,    # Sample rate
                size=-16,          # 16-bit signed samples
                channels=2,        # Stereo
                buffer=1024        # Buffer size
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(max_channels)
            
            self.initialized = True
            self.logger.info(f"Pygame concurrent audio initialized: {max_channels} channels")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")
            self.initialized = False
    
    def play_audio_file(self, file_path: str, volume: float = 0.8) -> Optional[str]:
        """
        Play audio file on an available channel
        
        Args:
            file_path: Path to audio file
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            Channel ID if successful, None if failed
        """
        if not self.initialized:
            self.logger.error("Pygame mixer not initialized")
            return None
            
        if not os.path.exists(file_path):
            self.logger.error(f"Audio file not found: {file_path}")
            return None
        
        try:
            # Find an available channel
            available_channel = None
            for i in range(self.max_channels):
                channel = pygame.mixer.Channel(i)
                if not channel.get_busy():
                    available_channel = channel
                    break
            
            if available_channel is None:
                # All channels busy, force use channel 0
                available_channel = pygame.mixer.Channel(0)
                self.logger.warning("All channels busy, using channel 0")
            
            # Load and convert audio file
            sound_data = self._load_sound(file_path)
            if sound_data is None:
                return None
            
            # Set volume and play
            available_channel.set_volume(volume)
            available_channel.play(sound_data)
            
            channel_id = f"channel_{available_channel}"
            self.logger.info(f"Playing {os.path.basename(file_path)} on {channel_id}")
            
            return channel_id
            
        except Exception as e:
            self.logger.error(f"Error playing audio file {file_path}: {e}")
            return None
    
    def _load_sound(self, file_path: str) -> Optional[pygame.mixer.Sound]:
        """
        Load audio file and convert to pygame Sound object
        
        Args:
            file_path: Path to audio file
            
        Returns:
            pygame.mixer.Sound object or None
        """
        try:
            if file_path.lower().endswith('.mp3'):
                # Convert MP3 to WAV in memory using pydub
                audio_segment = AudioSegment.from_mp3(file_path)
                # Convert to format pygame likes
                audio_segment = audio_segment.set_frame_rate(44100)
                audio_segment = audio_segment.set_channels(2)
                audio_segment = audio_segment.set_sample_width(2)
                
                # Export to bytes
                wav_data = io.BytesIO()
                audio_segment.export(wav_data, format="wav")
                wav_data.seek(0)
                
                # Load into pygame
                sound = pygame.mixer.Sound(wav_data)
                return sound
            else:
                # Load directly (WAV, OGG, etc.)
                sound = pygame.mixer.Sound(file_path)
                return sound
                
        except Exception as e:
            self.logger.error(f"Failed to load sound {file_path}: {e}")
            return None
    
    def stop_all_sounds(self):
        """Stop all playing sounds"""
        if self.initialized:
            pygame.mixer.stop()
            self.logger.info("Stopped all sounds")
    
    def get_active_channel_count(self) -> int:
        """Get number of currently active channels"""
        if not self.initialized:
            return 0
            
        active_count = 0
        for i in range(self.max_channels):
            channel = pygame.mixer.Channel(i)
            if channel.get_busy():
                active_count += 1
        return active_count
    
    def cleanup(self):
        """Cleanup pygame mixer"""
        if self.initialized:
            pygame.mixer.quit()
            self.logger.info("Pygame concurrent audio cleaned up")


# Global instance
_pygame_audio = None

def get_pygame_audio_manager() -> PygameConcurrentAudio:
    """Get the global pygame audio manager instance"""
    global _pygame_audio
    if _pygame_audio is None:
        _pygame_audio = PygameConcurrentAudio()
    return _pygame_audio


if __name__ == "__main__":
    # Test the pygame audio manager
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python pygame_concurrent_audio.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    manager = get_pygame_audio_manager()
    
    if not manager.initialized:
        print("Failed to initialize pygame mixer")
        sys.exit(1)
    
    print(f"Playing {audio_file} on multiple channels simultaneously...")
    
    # Play the same sound on multiple channels with delays
    for i in range(5):
        channel_id = manager.play_audio_file(audio_file, volume=0.7)
        print(f"Started playback {i+1}: {channel_id}")
        time.sleep(0.2)  # Small delay between starts
    
    # Wait for sounds to finish
    print("Waiting for sounds to finish...")
    while manager.get_active_channel_count() > 0:
        print(f"Active channels: {manager.get_active_channel_count()}")
        time.sleep(1)
    
    manager.cleanup()
    print("Test completed!")