# Light Bar USB CDC Controller

This module provides a Python interface for controlling an LED light bar connected via USB CDC (Communication Device Class).

## Overview

The light bar is now a USB-connected CircuitPython device that accepts commands over serial communication. This module provides both low-level and high-level interfaces for controlling the device.

## Device Commands

The USB CDC device accepts the following text commands:

- `rainbow` - Knight Rider style animation using rainbow colors
- `red` - Knight Rider style animation using red colors  
- `speech` - Rapid radial animation from center simulating speech
- `idle` - Peaceful idle animation with slow breathing effect
- `waterfall` - Waterfall animation
- `clear` - Turn off all pixels
- `demo` - Run a quick demo of all animations

## Usage

### Simple Usage

```python
from light_bar import light_bar

# Start the light bar manager
if light_bar.start():
    # Send commands
    light_bar.rainbow()
    light_bar.speech_animation()
    light_bar.clear()
    
    # Stop when done
    light_bar.stop()
```

### Advanced Usage

```python
from light_bar import LightBarController

# Create a controller with specific device path
controller = LightBarController('/dev/ttyACM0')

if controller.connect():
    # Send raw commands
    controller.send_command('rainbow')
    
    # Use convenience methods
    controller.speech_animation()
    controller.vu_meter(75)  # 75% volume level
    
    controller.disconnect()
```

### Legacy API Compatibility

The module maintains compatibility with the previous pi5neo-based API:

```python
from light_bar import rainbow_cycle, loading_bar, vu_meter

# These functions now send commands to the USB device
rainbow_cycle()
loading_bar(50)  # 50% progress
vu_meter(80)     # 80% volume
```

## Features

- **Automatic Device Detection**: Automatically finds USB CDC devices
- **Connection Monitoring**: Automatic reconnection if device disconnects
- **Thread-Safe**: Safe to use from multiple threads
- **Error Handling**: Robust error handling and logging
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Requirements

- `pyserial` - For serial communication
- USB CDC compatible device running CircuitPython

## Installation

```bash
pip install pyserial
```

## Device Setup

The device should be running CircuitPython code similar to:

```python
import supervisor
import usb_cdc
import time

def process_command(command):
    command = command.strip().lower()
    if command == "rainbow":
        # Rainbow animation
        pass
    elif command == "clear":
        # Clear pixels
        pass
    # ... other commands

while True:
    if supervisor.runtime.serial_connected:
        if usb_cdc.console.in_waiting > 0:
            try:
                command = usb_cdc.console.readline().decode('utf-8')
                if command:
                    process_command(command)
            except:
                pass
    time.sleep(0.1)
```

## Troubleshooting

### Device Not Found

1. Check that the device is connected via USB
2. Verify the device is running CircuitPython with USB CDC enabled
3. Check system permissions for serial devices
4. Try specifying the device path manually

### Connection Issues

1. Ensure no other application is using the device
2. Check the device appears in system device manager
3. Try unplugging and reconnecting the device
4. Verify the baudrate matches (default: 115200)

### Permission Errors (Linux)

Add your user to the dialout group:
```bash
sudo usermod -a -G dialout $USER
```
Log out and back in for changes to take effect.
