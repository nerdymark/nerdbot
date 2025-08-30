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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightBarController:
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
    
    def vu_meter(self, volume: float) -> bool:
        """
        Create a VU meter effect based on volume level.
        
        Args:
            volume: Volume level 0-100
            
        Returns:
            True if command sent successfully
        """
        # For now, map volume to different animations
        if volume < 10:
            return self.clear()
        elif volume < 30:
            return self.idle_animation()
        elif volume < 70:
            return self.speech_animation()
        else:
            return self.rainbow()
    
    def loading_bar(self, progress: float) -> bool:
        """
        Simulate a loading bar effect.
        
        Args:
            progress: Progress percentage 0-100
            
        Returns:
            True if commands sent successfully
        """
        # For a loading bar, we'll use a sequence of commands
        if progress >= 100:
            return self.rainbow()
        elif progress > 0:
            return self.speech_animation()
        else:
            return self.clear()


class LightBarManager:
    """High-level manager for light bar operations with automatic reconnection."""
    
    def __init__(self, device_path: Optional[str] = None):
        """
        Initialize the light bar manager.
        
        Args:
            device_path: Optional specific device path
        """
        self.controller = LightBarController(device_path)
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
    
    def __getattr__(self, name):
        """Delegate method calls to the controller."""
        return getattr(self.controller, name)


# Global light bar instance
light_bar = LightBarManager()


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
    return light_bar.pixels(n)


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
