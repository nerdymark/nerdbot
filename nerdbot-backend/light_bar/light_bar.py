"""
Control the LED strip using the connected USB CDC device

The code below is on the device, use it to determine how to control
it over USB as a serial device.

Available Serial Commands:
-------------------------
Basic Colors:
  white      - All LEDs bright white
  red        - All LEDs red
  green      - All LEDs green
  blue       - All LEDs blue
  off        - Turn off all LEDs
  clear      - Same as off

Animations:
  rainbow    - Knight Rider effect with rainbow colors (2 cycles)
  knight_red - Knight Rider effect with red colors (2 cycles)
  speech     - Cyan speech simulation animation (3 seconds)
  idle       - Peaceful warm color breathing effect
  waterfall  - Blue waterfall animation (reverse direction)

Testing/Diagnostics:
  test       - Run startup test sequence
  count      - Progressive pixel count test
  conservative - Low-power individual pixel test
  mapping    - Show array position to LED mapping
  pixels <N> - Light up first N unique pixels (1-8)

Special:
  demo       - Run sequence of all animations

Usage:
Connect via serial terminal and type any command followed by Enter.

"""

import serial
import time
import threading
import logging
import glob
from typing import Optional
try:
    from light_bar.wled_controller import wled_controller, RobotState
    WLED_AVAILABLE = True
except ImportError:
    WLED_AVAILABLE = False
    print("WLED controller not available, using USB CDC mode")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USBCDCLightBarController:
    """Controller for USB CDC connected LED light bar."""
    
    def __init__(self, device_path: Optional[str] = None, baudrate: int = 115200, timeout: float = 1.0):
        """
        Initialize the light bar controller.
        
        Args:
            device_path: Path to the USB CDC device (e.g., '/dev/ttyACM0')
            baudrate: Serial communication baudrate
            timeout: Serial timeout in seconds
        """
        self.device_path = device_path
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection: Optional[serial.Serial] = None
        self._connected = False
        
    def find_device(self) -> Optional[str]:
        """
        Automatically find the USB CDC device.
        
        Returns:
            Device path if found, None otherwise
        """
        # Common USB CDC device paths
        potential_devices = [
            '/dev/ttyACM*',
            '/dev/ttyUSB*',
            '/dev/cu.usbmodem*',  # macOS
            'COM*'  # Windows
        ]
        
        for pattern in potential_devices:
            devices = glob.glob(pattern)
            if devices:
                logger.info("Found potential devices: %s", devices)
                return devices[0]  # Return the first found device
                
        return None
    
    def connect(self) -> bool:
        """
        Connect to the USB CDC device.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.device_path:
            self.device_path = self.find_device()
            
        if not self.device_path:
            logger.error("No USB CDC device found")
            return False
            
        try:
            self.serial_connection = serial.Serial(
                self.device_path,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for device to be ready
            self._connected = True
            logger.info("Connected to light bar at %s", self.device_path)
            return True
            
        except serial.SerialException as e:
            logger.error("Failed to connect to %s: %s", self.device_path, e)
            return False
    
    def disconnect(self):
        """Disconnect from the USB CDC device."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self._connected = False
            logger.info("Disconnected from light bar")
    
    def is_connected(self) -> bool:
        """Check if connected to the device."""
        return self._connected and self.serial_connection and self.serial_connection.is_open
    
    def send_command(self, command: str) -> bool:
        """
        Send a command to the light bar.
        
        Args:
            command: Command to send
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to light bar")
            return False
            
        try:
            command_bytes = f"{command}\n".encode('utf-8')
            self.serial_connection.write(command_bytes)
            self.serial_connection.flush()
            logger.debug("Sent command: %s", command)
            return True
            
        except serial.SerialException as e:
            logger.error("Failed to send command '%s': %s", command, e)
            return False
    
    def rainbow(self) -> bool:
        """Start rainbow knight rider animation."""
        return self.send_command("rainbow")
    
    def red_knight_rider(self) -> bool:
        """Start red knight rider animation."""
        return self.send_command("knight_red")
    
    def knight_red(self) -> bool:
        """Start red knight rider animation (alias for red_knight_rider)."""
        return self.send_command("knight_red")
    
    def speech_animation(self) -> bool:
        """Start speech animation."""
        return self.send_command("speech")
    
    def idle_animation(self) -> bool:
        """Start idle animation."""
        return self.send_command("idle")
    
    def waterfall_animation(self) -> bool:
        """Start waterfall animation."""
        return self.send_command("waterfall")
    
    def clear(self) -> bool:
        """Clear all pixels (turn off)."""
        return self.send_command("clear")
    
    def demo(self) -> bool:
        """Run demo sequence."""
        return self.send_command("demo")
    
    # Basic Color Commands
    def white(self) -> bool:
        """Set all LEDs to bright white."""
        return self.send_command("white")
    
    def red(self) -> bool:
        """Set all LEDs to red."""
        return self.send_command("red")
    
    def green(self) -> bool:
        """Set all LEDs to green."""
        return self.send_command("green")
    
    def blue(self) -> bool:
        """Set all LEDs to blue."""
        return self.send_command("blue")
    
    def off(self) -> bool:
        """Turn off all LEDs."""
        return self.send_command("off")
    
    # Testing/Diagnostic Commands
    def test(self) -> bool:
        """Run startup test sequence."""
        return self.send_command("test")
    
    def count(self) -> bool:
        """Progressive pixel count test."""
        return self.send_command("count")
    
    def conservative(self) -> bool:
        """Low-power individual pixel test."""
        return self.send_command("conservative")
    
    def mapping(self) -> bool:
        """Show array position to LED mapping."""
        return self.send_command("mapping")
    
    def pixels(self, n: int) -> bool:
        """
        Light up first N unique pixels.
        
        Args:
            n: Number of pixels to light up (1-8)
            
        Returns:
            True if command sent successfully
        """
        if not 1 <= n <= 8:
            logger.error("Invalid pixel count: %d. Must be between 1 and 8.", n)
            return False
        return self.send_command(f"pixels {n}")


class LightBarController:
    """Unified controller that uses WLED if available, otherwise USB CDC."""
    
    def __init__(self, use_wled: bool = True, device_path: Optional[str] = None):
        """
        Initialize the light bar controller.
        
        Args:
            use_wled: Use WLED controller if available
            device_path: USB CDC device path (only used if not using WLED)
        """
        self.use_wled = use_wled and WLED_AVAILABLE
        
        if self.use_wled:
            self.controller = wled_controller
            logger.info("Using WLED controller")
        else:
            self.controller = USBCDCLightBarController(device_path)
            logger.info("Using USB CDC controller")
            
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to the light bar."""
        if self.use_wled:
            # WLED is always "connected" via HTTP
            info = self.controller.get_info()
            if info:
                self._connected = True
                logger.info(f"Connected to WLED with {info['leds']['count']} LEDs")
                # Set initial state
                self.controller.startup()
                time.sleep(2)
                self.controller.idle()
                return True
            return False
        else:
            self._connected = self.controller.connect()
            return self._connected
    
    def disconnect(self):
        """Disconnect from the light bar."""
        if self.use_wled:
            self.controller.shutdown()
            time.sleep(2)
            self.controller.turn_off()
        else:
            self.controller.disconnect()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected."""
        if self.use_wled:
            return self._connected
        else:
            return self.controller.is_connected()
    
    # Unified command methods that work with both controllers
    def send_command(self, command: str) -> bool:
        """Send a command (USB CDC mode) or map to WLED effect."""
        if self.use_wled:
            # Map USB CDC commands to WLED effects
            command_map = {
                "rainbow": lambda: self.controller.rainbow(),
                "knight_red": lambda: self.controller.moving(),
                "speech": lambda: self.controller.speaking(),
                "idle": lambda: self.controller.idle(),
                "waterfall": lambda: self.controller.set_effect("waterfall", 100, 150),
                "clear": lambda: self.controller.turn_off(),
                "off": lambda: self.controller.turn_off(),
                "demo": lambda: self.controller.celebration(),
                "white": lambda: self.controller.set_color(255, 255, 255),
                "red": lambda: self.controller.set_color(255, 0, 0),
                "green": lambda: self.controller.set_color(0, 255, 0),
                "blue": lambda: self.controller.set_color(0, 0, 255),
                "test": lambda: self.controller.startup()
            }
            
            if command in command_map:
                return command_map[command]()
            else:
                logger.warning(f"Unknown command for WLED: {command}")
                return False
        else:
            return self.controller.send_command(command)
    
    # New methods for robot state management (WLED only)
    def set_robot_state(self, state: str) -> bool:
        """Set robot state for light effects."""
        if self.use_wled:
            try:
                robot_state = RobotState(state)
                return self.controller.set_robot_state(robot_state)
            except ValueError:
                logger.error(f"Invalid robot state: {state}")
                return False
        else:
            # Map states to USB CDC commands
            state_map = {
                "idle": "idle",
                "moving": "knight_red",
                "speaking": "speech",
                "listening": "waterfall",
                "detection": "rainbow",
                "thinking": "waterfall",
                "error": "red",
                "startup": "test",
                "shutdown": "off"
            }
            return self.send_command(state_map.get(state, "idle"))
    
    # Pass through methods
    def rainbow(self) -> bool:
        return self.send_command("rainbow")
    
    def red_knight_rider(self) -> bool:
        return self.send_command("knight_red")
    
    def knight_red(self) -> bool:
        return self.send_command("knight_red")
    
    def speech_animation(self) -> bool:
        return self.send_command("speech")
    
    def idle_animation(self) -> bool:
        return self.send_command("idle")
    
    def waterfall_animation(self) -> bool:
        return self.send_command("waterfall")
    
    def clear(self) -> bool:
        return self.send_command("clear")
    
    def demo(self) -> bool:
        return self.send_command("demo")
    
    def white(self) -> bool:
        return self.send_command("white")
    
    def red(self) -> bool:
        return self.send_command("red")
    
    def green(self) -> bool:
        return self.send_command("green")
    
    def blue(self) -> bool:
        return self.send_command("blue")
    
    def off(self) -> bool:
        return self.send_command("off")
    
    def test(self) -> bool:
        return self.send_command("test")
    
    # WLED-specific methods
    def audio_reactive(self, intensity: float = 0.5) -> bool:
        """Audio reactive effect (WLED only)."""
        if self.use_wled:
            return self.controller.audio_reactive(intensity)
        else:
            # Fallback for USB CDC
            if intensity > 0.7:
                return self.send_command("rainbow")
            elif intensity > 0.3:
                return self.send_command("speech")
            else:
                return self.send_command("idle")
    
    def celebration(self) -> bool:
        """Celebration effect."""
        if self.use_wled:
            return self.controller.celebration()
        else:
            return self.send_command("demo")
    
    def fire(self) -> bool:
        """Fire effect (WLED only)."""
        if self.use_wled:
            return self.controller.fire()
        else:
            return self.send_command("red")
    
    def vu_meter(self, volume: float) -> bool:
        """VU meter effect based on volume."""
        if self.use_wled:
            return self.controller.audio_reactive(volume / 100.0)
        else:
            # Original USB CDC implementation
            if volume < 10:
                return self.clear()
            elif volume < 30:
                return self.idle_animation()
            elif volume < 70:
                return self.speech_animation()
            else:
                return self.rainbow()
    
    def loading_bar(self, progress: float) -> bool:
        """Loading bar effect."""
        if self.use_wled:
            if progress >= 100:
                return self.controller.celebration()
            else:
                # Use wipe effect with progress
                speed = int(50 + progress * 2)
                return self.controller.set_effect("wipe", speed=speed, intensity=200)
        else:
            # Original USB CDC implementation
            if progress >= 100:
                return self.rainbow()
            elif progress > 0:
                return self.speech_animation()
            else:
                return self.clear()
    
    # Headlights functionality
    def toggle_headlights(self) -> bool:
        """Toggle headlights mode on/off"""
        if self.use_wled:
            return self.controller.toggle_headlights()
        else:
            # For USB CDC, just toggle white color
            return self.send_command("white")
    
    def headlights_on(self) -> bool:
        """Turn on headlights mode"""
        if self.use_wled:
            return self.controller.headlights_on()
        else:
            return self.send_command("white")
    
    def headlights_off(self) -> bool:
        """Turn off headlights mode"""
        if self.use_wled:
            return self.controller.headlights_off()
        else:
            return self.send_command("idle")
    
    def is_headlights_active(self) -> bool:
        """Check if headlights mode is active"""
        if self.use_wled:
            return self.controller.is_headlights_active()
        else:
            return False  # USB CDC doesn't track headlights state


class LightBarManager:
    """High-level manager for light bar operations with automatic reconnection."""
    
    def __init__(self, use_wled: bool = True, device_path: Optional[str] = None):
        """
        Initialize the light bar manager.
        
        Args:
            use_wled: Use WLED if available
            device_path: Optional specific device path for USB CDC
        """
        self.controller = LightBarController(use_wled, device_path)
        self._stop_monitor = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """
        Start the light bar manager and connect to device.
        
        Returns:
            True if successfully started
        """
        if self.controller.connect():
            self._start_connection_monitor()
            return True
        return False
    
    def stop(self):
        """Stop the light bar manager."""
        self._stop_monitor = True
        if self._monitor_thread:
            self._monitor_thread.join()
        self.controller.disconnect()
    
    def _start_connection_monitor(self):
        """Start a background thread to monitor connection."""
        self._monitor_thread = threading.Thread(target=self._monitor_connection)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _monitor_connection(self):
        """Monitor connection and attempt reconnection if needed."""
        while not self._stop_monitor:
            if not self.controller.is_connected():
                logger.warning("Light bar disconnected, attempting reconnection...")
                time.sleep(2)
                self.controller.connect()
            time.sleep(5)  # Check every 5 seconds
    
    # Additional state management methods
    def set_robot_state(self, state: str) -> bool:
        """Set robot state for appropriate lighting."""
        return self.controller.set_robot_state(state)
    
    def audio_reactive(self, intensity: float = 0.5) -> bool:
        """Set audio reactive lighting."""
        return self.controller.audio_reactive(intensity)
    
    def celebration(self) -> bool:
        """Trigger celebration lighting."""
        return self.controller.celebration()
    
    def __getattr__(self, name):
        """Delegate method calls to the controller."""
        return getattr(self.controller, name)


# Create global instance with monitoring
def create_light_bar():
    """Create and configure the light bar instance"""
    lb = LightBarManager(use_wled=True)
    if lb.use_wled and hasattr(lb.controller, 'set_rate_limit'):
        # Configure WLED rate limiting
        lb.controller.set_rate_limit(enabled=True, min_interval=0.1)
        # Log queue status periodically for monitoring
        def monitor_queue():
            while True:
                time.sleep(10)  # Check every 10 seconds
                if hasattr(lb.controller, 'get_queue_status'):
                    status = lb.controller.get_queue_status()
                    if status['queue_size'] > 5:
                        logger.warning(f"WLED queue building up: {status}")
        monitor_thread = threading.Thread(target=monitor_queue, daemon=True)
        monitor_thread.start()
    return lb

# Global light bar instance - defaults to WLED if available
light_bar = create_light_bar()


def rainbow_cycle(delay: float = 0.1) -> bool:  # pylint: disable=unused-argument
    """
    Create a rainbow cycle effect.
    
    Args:
        delay: Delay between color changes (not used with USB device)
        
    Returns:
        True if command sent successfully
    """
    return light_bar.rainbow()


def loading_bar(progress: float = 100) -> bool:
    """
    Create a loading bar effect.
    
    Args:
        progress: Loading progress 0-100
        
    Returns:
        True if command sent successfully
    """
    return light_bar.loading_bar(progress)


def vu_meter(volume: float) -> bool:
    """
    Create a VU meter effect.
    
    Args:
        volume: Volume level 0-100
        
    Returns:
        True if command sent successfully
    """
    return light_bar.vu_meter(volume)


# Basic color wrapper functions
def white() -> bool:
    """Set all LEDs to bright white."""
    return light_bar.white()


def red() -> bool:
    """Set all LEDs to red."""
    return light_bar.red()


def green() -> bool:
    """Set all LEDs to green."""
    return light_bar.green()


def blue() -> bool:
    """Set all LEDs to blue."""
    return light_bar.blue()


def off() -> bool:
    """Turn off all LEDs."""
    return light_bar.off()


def clear() -> bool:
    """Clear all pixels (turn off) - alias for off()."""
    return light_bar.clear()


# Animation wrapper functions
def rainbow() -> bool:
    """Start rainbow knight rider animation."""
    return light_bar.rainbow()


def knight_red() -> bool:
    """Start red knight rider animation."""
    return light_bar.knight_red()


def speech() -> bool:
    """Start speech animation."""
    return light_bar.speech_animation()


def idle() -> bool:
    """Start idle animation."""
    return light_bar.idle_animation()


def waterfall() -> bool:
    """Start waterfall animation."""
    return light_bar.waterfall_animation()


def demo() -> bool:
    """Run demo sequence."""
    return light_bar.demo()


# Testing/Diagnostic wrapper functions
def test() -> bool:
    """Run startup test sequence."""
    return light_bar.test()


def count() -> bool:
    """Progressive pixel count test."""
    return light_bar.count()


def conservative() -> bool:
    """Low-power individual pixel test."""
    return light_bar.conservative()


def mapping() -> bool:
    """Show array position to LED mapping."""
    return light_bar.mapping()


def pixels(n: int) -> bool:
    """
    Light up first N unique pixels.

    Args:
        n: Number of pixels to light up (1-8)

    Returns:
        True if command sent successfully
    """
    # This function gets called frequently in tracking loops
    # Add extra throttling
    current_time = time.time()
    if hasattr(pixels, '_last_call') and current_time - pixels._last_call < 0.1:
        return True  # Throttle to max 10 calls per second
    pixels._last_call = current_time
    return light_bar.send_command(f"pixels {n}")


# Auto-start on import if this is the main module
if __name__ == "__main__":
    # Example usage
    if light_bar.start():
        logger.info("Light bar started successfully")
        
        # Demo sequence
        time.sleep(1)
        light_bar.rainbow()
        time.sleep(3)
        light_bar.speech_animation()
        time.sleep(3)
        light_bar.idle_animation()
        time.sleep(3)
        light_bar.clear()
        
        light_bar.stop()
    else:
        logger.error("Failed to start light bar")
