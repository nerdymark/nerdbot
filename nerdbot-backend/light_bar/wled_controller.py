"""
WLED Controller for NerdBot Light Bar
Controls an LED strip via WLED JSON API for robot personality effects
"""

import requests
import json
import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
import random
from collections import deque
from threading import Lock, Thread

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Robot states for light bar effects"""
    IDLE = "idle"
    MOVING = "moving"
    SPEAKING = "speaking"
    LISTENING = "listening"
    DETECTION = "detection"
    THINKING = "thinking"
    ERROR = "error"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class WLEDController:
    """Controller for WLED LED strip via JSON API"""
    
    # WLED Effect IDs (mapped from effect names)
    EFFECTS = {
        "solid": 0,
        "blink": 1,
        "breathe": 2,
        "wipe": 3,
        "rainbow": 9,
        "scan": 10,
        "theater": 13,
        "sparkle": 20,
        "strobe": 23,
        "android": 27,
        "chase": 28,
        "colorwaves": 31,
        "fire": 35,
        "pride": 39,
        "colortwinkles": 51,
        "lake": 52,
        "meteor": 53,
        "smooth_meteor": 54,
        "railway": 55,
        "ripple": 56,
        "twinklefox": 57,
        "fireworks": 58,
        "rain": 59,
        "tetris": 61,
        "plasma": 62,
        "bpm": 64,
        "pacifica": 66,
        "candle": 67,
        "starburst": 69,
        "exploding_fireworks": 70,
        "bouncingballs": 71,
        "sinelon": 72,
        "popcorn": 74,
        "drip": 75,
        "wavesin": 78,
        "phased": 84,
        "lighthouse": 88,
        "colorful": 97,
        "traffic_light": 98,
        "halloween_eyes": 100,
        "solid_glitter": 101,
        "sunrise": 102,
        "phased_noise": 103,
        "twinkleup": 106,
        "aurora": 110,
        "candle_multi": 111,
        "waterfall": 113,
        "freqwave": 114,
        "oscillate": 115,
        "pride_2015": 116,
        "juggle": 117,
        "palette": 118,
        "fire_2012": 119,
        "colorwaves_2": 120,
        "noise_pal": 123,
        "sine": 124,
        "phased_2": 125,
        "flow": 126,
        "chunchun": 127,
        "dancing_shadows": 128,
        "sun_radiation": 130,
        "noise_fire": 131,
        "noise_mover": 132,
        "noise1": 133,
        "colorful_2": 134,
        "colorful_3": 135,
        "colorful_4": 136,
        "colorful_5": 137,
        "colorful_6": 138
    }
    
    # Personality presets for different robot states
    STATE_PRESETS = {
        RobotState.IDLE: [
            {
                "name": "Peaceful Breathing",
                "fx": "breathe",
                "speed": 128,
                "intensity": 128,
                "palette": 0,
                "colors": [[0, 100, 200], [0, 50, 150], [0, 20, 100]]
            },
            {
                "name": "Gentle Aurora",
                "fx": "aurora",
                "speed": 80,
                "intensity": 100,
                "palette": 6,
                "colors": [[0, 100, 200], [100, 0, 200], [0, 200, 100]]
            },
            {
                "name": "Calm Lake",
                "fx": "lake",
                "speed": 60,
                "intensity": 80,
                "palette": 36,
                "colors": [[0, 50, 100], [0, 100, 150], [50, 100, 200]]
            }
        ],
        RobotState.MOVING: [
            {
                "name": "Scanner",
                "fx": "scan",
                "speed": 200,
                "intensity": 255,
                "palette": 0,
                "colors": [[255, 0, 0], [200, 0, 0], [100, 0, 0]]
            },
            {
                "name": "Railway",
                "fx": "railway",
                "speed": 180,
                "intensity": 200,
                "palette": 0,
                "colors": [[0, 255, 0], [0, 200, 0], [0, 100, 0]]
            },
            {
                "name": "Chase",
                "fx": "chase",
                "speed": 220,
                "intensity": 180,
                "palette": 0,
                "colors": [[255, 100, 0], [200, 80, 0], [150, 60, 0]]
            }
        ],
        RobotState.SPEAKING: [
            {
                "name": "Voice Wave",
                "fx": "wavesin",
                "speed": 180,
                "intensity": 200,
                "palette": 0,
                "colors": [[0, 255, 255], [0, 200, 200], [0, 150, 150]]
            },
            {
                "name": "Oscillate",
                "fx": "oscillate",
                "speed": 160,
                "intensity": 180,
                "palette": 0,
                "colors": [[100, 200, 255], [50, 150, 200], [0, 100, 150]]
            },
            {
                "name": "Flow",
                "fx": "flow",
                "speed": 140,
                "intensity": 160,
                "palette": 46,
                "colors": [[0, 200, 255], [0, 150, 200], [0, 100, 150]]
            }
        ],
        RobotState.LISTENING: [
            {
                "name": "Waiting Pulse",
                "fx": "sinelon",
                "speed": 100,
                "intensity": 150,
                "palette": 0,
                "colors": [[255, 255, 0], [200, 200, 0], [150, 150, 0]]
            },
            {
                "name": "Gentle Twinkle",
                "fx": "twinkleup",
                "speed": 80,
                "intensity": 100,
                "palette": 0,
                "colors": [[255, 200, 100], [200, 150, 50], [150, 100, 0]]
            }
        ],
        RobotState.DETECTION: [
            {
                "name": "Alert Strobe",
                "fx": "strobe",
                "speed": 255,
                "intensity": 255,
                "palette": 0,
                "colors": [[255, 0, 0], [255, 255, 0], [255, 100, 0]]
            },
            {
                "name": "Fireworks",
                "fx": "fireworks",
                "speed": 200,
                "intensity": 200,
                "palette": 0,
                "colors": [[255, 0, 100], [255, 100, 0], [100, 0, 255]]
            },
            {
                "name": "Starburst",
                "fx": "starburst",
                "speed": 180,
                "intensity": 220,
                "palette": 0,
                "colors": [[255, 0, 0], [255, 100, 0], [255, 200, 0]]
            }
        ],
        RobotState.THINKING: [
            {
                "name": "Processing",
                "fx": "phased",
                "speed": 120,
                "intensity": 150,
                "palette": 0,
                "colors": [[100, 0, 255], [50, 0, 200], [0, 0, 150]]
            },
            {
                "name": "Plasma Think",
                "fx": "plasma",
                "speed": 100,
                "intensity": 130,
                "palette": 44,
                "colors": [[150, 0, 255], [100, 0, 200], [50, 0, 150]]
            }
        ],
        RobotState.ERROR: [
            {
                "name": "Error Blink",
                "fx": "blink",
                "speed": 255,
                "intensity": 255,
                "palette": 0,
                "colors": [[255, 0, 0], [200, 0, 0], [150, 0, 0]]
            }
        ],
        RobotState.STARTUP: [
            {
                "name": "Boot Sequence",
                "fx": "wipe",
                "speed": 200,
                "intensity": 255,
                "palette": 0,
                "colors": [[0, 255, 0], [0, 200, 100], [0, 150, 50]]
            }
        ],
        RobotState.SHUTDOWN: [
            {
                "name": "Power Down",
                "fx": "fade",
                "speed": 50,
                "intensity": 100,
                "palette": 0,
                "colors": [[100, 0, 0], [50, 0, 0], [0, 0, 0]]
            }
        ]
    }
    
    def __init__(self, host: str = "http://10.0.1.166", timeout: float = 2.0):
        """
        Initialize WLED controller

        Args:
            host: WLED device URL
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.current_state = RobotState.IDLE
        self.last_preset = None
        self._brightness = 13  # 5% brightness (13/255) - dimmer than cat's eyes
        self._is_on = False

        # Brightness levels
        self.NORMAL_BRIGHTNESS = 13    # 5% for normal operation - dimmer than cat's eyes
        self.ALERT_BRIGHTNESS = 200    # 78% for attention-getting effects (reduced from 100%)
        self.HEADLIGHT_BRIGHTNESS = 100  # 39% for headlights mode (reduced from 50%)

        # Headlights state
        self.headlights_active = False

        # Rate limiting configuration
        self.rate_limit_enabled = True
        self.min_request_interval = 0.1  # Minimum 100ms between requests
        self.max_queue_size = 10  # Maximum pending requests
        self.last_request_time = 0
        self.request_queue = deque(maxlen=self.max_queue_size)
        self.queue_lock = Lock()
        self.processing_thread = None
        self.processing_active = False
        self.last_state_cache = None  # Cache last successful state
        self.cache_timeout = 0.5  # Cache valid for 500ms
        self.last_cache_time = 0

        # Start the request processing thread
        self._start_processing_thread()
        
    def _start_processing_thread(self):
        """Start the background thread for processing queued requests"""
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.processing_active = True
            self.processing_thread = Thread(target=self._process_queue, daemon=True)
            self.processing_thread.start()
            logger.info("Started WLED request processing thread")

    def _stop_processing_thread(self):
        """Stop the background processing thread"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            logger.info("Stopped WLED request processing thread")

    def _process_queue(self):
        """Background thread that processes queued requests with rate limiting"""
        while self.processing_active:
            try:
                with self.queue_lock:
                    if not self.request_queue:
                        time.sleep(0.05)  # Sleep briefly if queue is empty
                        continue

                    # Check rate limit
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < self.min_request_interval:
                        time.sleep(self.min_request_interval - time_since_last)
                        continue

                    # Get next request
                    request_data = self.request_queue.popleft()

                # Process the request
                endpoint = request_data['endpoint']
                data = request_data.get('data')
                method = request_data.get('method', 'POST')

                result = self._send_request_internal(endpoint, data, method)

                # Update cache if successful
                if result and endpoint == "/json/state":
                    self.last_state_cache = data
                    self.last_cache_time = time.time()

                self.last_request_time = time.time()

            except Exception as e:
                logger.error(f"Error processing WLED request queue: {e}")
                time.sleep(0.1)

    def _send_request_internal(self, endpoint: str, data: Optional[Dict] = None, method: str = "POST") -> Optional[Dict]:
        """Internal method to actually send the request (not rate limited)"""
        url = f"{self.host}{endpoint}"

        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                response = requests.get(url, timeout=self.timeout)

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"WLED request failed: {e}")
            return None

    def _send_request(self, endpoint: str, data: Optional[Dict] = None, method: str = "POST") -> Optional[Dict]:
        """
        Send request to WLED API with rate limiting

        Args:
            endpoint: API endpoint
            data: JSON data to send
            method: HTTP method

        Returns:
            Response JSON or None if failed
        """
        if not self.rate_limit_enabled:
            # Bypass rate limiting if disabled
            return self._send_request_internal(endpoint, data, method)

        # For GET requests, check cache first
        if method == "GET" and endpoint == "/json/state":
            current_time = time.time()
            if (self.last_state_cache and
                current_time - self.last_cache_time < self.cache_timeout):
                logger.debug("Returning cached state")
                return self.last_state_cache

        # Queue the request
        with self.queue_lock:
            # If queue is full, drop oldest request
            if len(self.request_queue) >= self.max_queue_size:
                dropped = self.request_queue.popleft()
                logger.warning(f"Dropped WLED request due to queue overflow: {dropped}")

            # Add to queue
            request_data = {
                'endpoint': endpoint,
                'data': data,
                'method': method,
                'timestamp': time.time()
            }
            self.request_queue.append(request_data)
            logger.debug(f"Queued WLED request: {endpoint} (queue size: {len(self.request_queue)})")

        # For state updates, return success immediately (fire and forget)
        if endpoint == "/json/state":
            return {'success': True}

        # For GET requests, wait briefly for result
        if method == "GET":
            time.sleep(0.2)  # Wait briefly for processing
            return self.last_state_cache

        return {'success': True}
    
    def get_info(self) -> Optional[Dict]:
        """Get WLED device information"""
        return self._send_request("/json/info", method="GET")
    
    def get_state(self) -> Optional[Dict]:
        """Get current WLED state"""
        return self._send_request("/json/state", method="GET")
    
    def set_state(self, state_data: Dict) -> bool:
        """
        Set WLED state
        
        Args:
            state_data: State configuration dictionary
            
        Returns:
            True if successful
        """
        result = self._send_request("/json/state", state_data)
        return result is not None
    
    def turn_on(self) -> bool:
        """Turn on LEDs"""
        self._is_on = True
        return self.set_state({"on": True})
    
    def turn_off(self) -> bool:
        """Turn off LEDs"""
        self._is_on = False
        return self.set_state({"on": False})
    
    def set_brightness(self, brightness: int) -> bool:
        """
        Set brightness level
        
        Args:
            brightness: Brightness level (0-255)
        """
        self._brightness = max(0, min(255, brightness))
        return self.set_state({"on": True, "bri": self._brightness})
    
    def set_color(self, r: int, g: int, b: int, brightness: Optional[int] = None) -> bool:
        """
        Set solid color
        
        Args:
            r, g, b: RGB values (0-255)
            brightness: Brightness level (uses normal brightness if None)
        """
        if brightness is None:
            brightness = self.NORMAL_BRIGHTNESS
            
        return self.set_state({
            "on": True,
            "bri": brightness,
            "seg": [{"col": [[r, g, b]]}]
        })
    
    def set_effect(self, effect_name: str, speed: int = 128, intensity: int = 128, brightness: Optional[int] = None) -> bool:
        """
        Set effect by name
        
        Args:
            effect_name: Effect name from EFFECTS dict
            speed: Effect speed (0-255)
            intensity: Effect intensity (0-255)
            brightness: Brightness level (uses normal brightness if None)
        """
        if effect_name not in self.EFFECTS:
            logger.error(f"Unknown effect: {effect_name}")
            return False
            
        if brightness is None:
            brightness = self.NORMAL_BRIGHTNESS
            
        effect_id = self.EFFECTS[effect_name]
        return self.set_state({
            "on": True,
            "bri": brightness,
            "seg": [{
                "fx": effect_id,
                "sx": speed,
                "ix": intensity
            }]
        })
    
    def apply_preset(self, preset: Dict, brightness: Optional[int] = None) -> bool:
        """
        Apply a preset configuration
        
        Args:
            preset: Preset dictionary with fx, speed, intensity, colors, etc.
            brightness: Override brightness level, uses normal brightness if None
        """
        effect_id = self.EFFECTS.get(preset["fx"], 0)
        
        # Use normal brightness unless overridden
        if brightness is None:
            brightness = self.NORMAL_BRIGHTNESS
            
        state_data = {
            "on": True,
            "bri": brightness,
            "seg": [{
                "fx": effect_id,
                "sx": preset.get("speed", 128),
                "ix": preset.get("intensity", 128),
                "col": preset.get("colors", [[255, 255, 255]])
            }]
        }
        
        if "palette" in preset:
            state_data["seg"][0]["pal"] = preset["palette"]
            
        return self.set_state(state_data)
    
    def set_robot_state(self, state: RobotState, variant: Optional[int] = None) -> bool:
        """
        Set light effect based on robot state
        
        Args:
            state: Robot state enum
            variant: Optional variant index, random if None
        """
        # Prevent state changes when headlights are active
        if self.headlights_active:
            logger.debug(f"Ignoring robot state change to {state.value} - headlights active")
            return False
            
        self.current_state = state
        
        if state not in self.STATE_PRESETS:
            logger.warning(f"No presets for state: {state}")
            return False
            
        presets = self.STATE_PRESETS[state]
        
        if variant is None:
            # Choose random preset, but avoid repeating the last one if possible
            if len(presets) > 1 and self.last_preset:
                available = [p for p in presets if p != self.last_preset]
                preset = random.choice(available if available else presets)
            else:
                preset = random.choice(presets)
        else:
            preset = presets[variant % len(presets)]
            
        self.last_preset = preset
        logger.info(f"Setting light state: {state.value} - {preset['name']}")
        
        # Use alert brightness for attention-getting states
        if state in [RobotState.DETECTION, RobotState.ERROR, RobotState.STARTUP]:
            brightness = self.ALERT_BRIGHTNESS
        else:
            brightness = self.NORMAL_BRIGHTNESS
            
        return self.apply_preset(preset, brightness)
    
    # Convenience methods for specific states
    def idle(self, variant: Optional[int] = None) -> bool:
        """Set idle animation"""
        return self.set_robot_state(RobotState.IDLE, variant)
    
    def moving(self, variant: Optional[int] = None) -> bool:
        """Set moving animation"""
        return self.set_robot_state(RobotState.MOVING, variant)
    
    def speaking(self, variant: Optional[int] = None) -> bool:
        """Set speaking animation"""
        return self.set_robot_state(RobotState.SPEAKING, variant)
    
    def listening(self, variant: Optional[int] = None) -> bool:
        """Set listening animation"""
        return self.set_robot_state(RobotState.LISTENING, variant)
    
    def detection(self, variant: Optional[int] = None) -> bool:
        """Set detection alert animation"""
        return self.set_robot_state(RobotState.DETECTION, variant)
    
    def thinking(self, variant: Optional[int] = None) -> bool:
        """Set thinking animation"""
        return self.set_robot_state(RobotState.THINKING, variant)
    
    def error(self) -> bool:
        """Set error animation"""
        return self.set_robot_state(RobotState.ERROR)
    
    def startup(self) -> bool:
        """Set startup animation"""
        return self.set_robot_state(RobotState.STARTUP)
    
    def shutdown(self) -> bool:
        """Set shutdown animation"""
        return self.set_robot_state(RobotState.SHUTDOWN)
    
    def dim_to_normal(self) -> bool:
        """Dim current effect to normal brightness"""
        return self.set_state({"bri": self.NORMAL_BRIGHTNESS})
    
    # Special effects for audio/meme playback
    def audio_reactive(self, intensity_level: float = 0.5) -> bool:
        """
        Audio reactive effect based on intensity
        
        Args:
            intensity_level: Audio intensity (0.0-1.0)
        """
        # Map intensity to different effects and brightness
        if intensity_level < 0.3:
            effect = "colortwinkles"
            speed = int(100 * intensity_level / 0.3)
            brightness = self.NORMAL_BRIGHTNESS
        elif intensity_level < 0.6:
            effect = "bpm"
            speed = int(128 + 127 * (intensity_level - 0.3) / 0.3)
            brightness = int(self.NORMAL_BRIGHTNESS + (self.ALERT_BRIGHTNESS - self.NORMAL_BRIGHTNESS) * 0.5)
        else:
            effect = "fireworks"
            speed = int(200 + 55 * (intensity_level - 0.6) / 0.4)
            brightness = self.ALERT_BRIGHTNESS  # Full brightness for high intensity
            
        return self.set_effect(effect, speed=speed, intensity=int(255 * intensity_level), brightness=brightness)
    
    def celebration(self) -> bool:
        """Celebration effect for special moments"""
        effects = ["fireworks", "exploding_fireworks", "starburst", "pride"]
        effect = random.choice(effects)
        return self.set_effect(effect, speed=200, intensity=255, brightness=self.ALERT_BRIGHTNESS)
    
    def rainbow(self) -> bool:
        """Classic rainbow effect"""
        return self.set_effect("rainbow", speed=128, intensity=255, brightness=self.NORMAL_BRIGHTNESS)
    
    def fire(self) -> bool:
        """Fire effect"""
        return self.set_effect("fire", speed=150, intensity=200, brightness=self.NORMAL_BRIGHTNESS)
    
    def police(self) -> bool:
        """Police light effect"""
        # Alternating red and blue strobe
        return self.set_state({
            "on": True,
            "bri": self.ALERT_BRIGHTNESS,
            "seg": [{
                "fx": self.EFFECTS["strobe"],
                "sx": 255,
                "ix": 255,
                "col": [[255, 0, 0], [0, 0, 255]]
            }]
        })
    
    # Headlights functionality
    def toggle_headlights(self) -> bool:
        """Toggle headlights mode on/off"""
        if self.headlights_active:
            return self.headlights_off()
        else:
            return self.headlights_on()
    
    def headlights_on(self) -> bool:
        """Turn on headlights mode (solid white at medium brightness)"""
        self.headlights_active = True
        logger.info("Headlights ON - locking light bar state")
        return self.set_state({
            "on": True,
            "bri": self.HEADLIGHT_BRIGHTNESS,
            "seg": [{
                "fx": self.EFFECTS["solid"],
                "col": [[255, 255, 255]]  # Pure white
            }]
        })
    
    def headlights_off(self) -> bool:
        """Turn off headlights mode and return to previous state"""
        self.headlights_active = False
        logger.info("Headlights OFF - unlocking light bar state")
        # Return to idle state
        return self.set_robot_state(RobotState.IDLE)
    
    def is_headlights_active(self) -> bool:
        """Check if headlights mode is active"""
        return self.headlights_active

    def set_rate_limit(self, enabled: bool = True, min_interval: float = 0.1):
        """Configure rate limiting

        Args:
            enabled: Enable/disable rate limiting
            min_interval: Minimum seconds between requests
        """
        self.rate_limit_enabled = enabled
        self.min_request_interval = max(0.05, min_interval)  # Minimum 50ms
        logger.info(f"Rate limiting {'enabled' if enabled else 'disabled'} "
                   f"(min interval: {self.min_request_interval}s)")

    def get_queue_status(self) -> Dict:
        """Get current queue status for monitoring"""
        with self.queue_lock:
            return {
                'queue_size': len(self.request_queue),
                'max_queue_size': self.max_queue_size,
                'rate_limit_enabled': self.rate_limit_enabled,
                'min_interval': self.min_request_interval,
                'processing_active': self.processing_active
            }

    def clear_queue(self):
        """Clear all pending requests"""
        with self.queue_lock:
            self.request_queue.clear()
            logger.info("Cleared WLED request queue")

    def __del__(self):
        """Cleanup on deletion"""
        self._stop_processing_thread()


# Global instance
wled_controller = WLEDController()


# Test function
if __name__ == "__main__":
    import sys
    
    print("Testing WLED Controller...")
    
    # Get device info
    info = wled_controller.get_info()
    if info:
        print(f"Connected to WLED v{info['ver']} with {info['leds']['count']} LEDs")
    else:
        print("Failed to connect to WLED")
        sys.exit(1)
    
    # Test different states
    print("\nTesting robot states...")
    
    states_to_test = [
        (RobotState.STARTUP, 3),
        (RobotState.IDLE, 3),
        (RobotState.THINKING, 2),
        (RobotState.MOVING, 3),
        (RobotState.DETECTION, 2),
        (RobotState.SPEAKING, 3),
        (RobotState.LISTENING, 2),
        (RobotState.SHUTDOWN, 3)
    ]
    
    for state, duration in states_to_test:
        print(f"Testing {state.value}...")
        wled_controller.set_robot_state(state)
        time.sleep(duration)
    
    print("\nTesting special effects...")
    
    # Test audio reactive with different intensities
    for intensity in [0.2, 0.5, 0.8, 1.0]:
        print(f"Audio reactive (intensity {intensity})...")
        wled_controller.audio_reactive(intensity)
        time.sleep(2)
    
    print("Celebration...")
    wled_controller.celebration()
    time.sleep(3)
    
    print("Rainbow...")
    wled_controller.rainbow()
    time.sleep(3)
    
    print("Fire...")
    wled_controller.fire()
    time.sleep(3)
    
    print("Police...")
    wled_controller.police()
    time.sleep(3)
    
    print("\nTurning off...")
    wled_controller.turn_off()
    
    print("Test complete!")