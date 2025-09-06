# Light Bar WLED Controller

This module provides a Python interface for controlling an LED light bar using WLED over WiFi.

## Overview

The light bar is now a WiFi-connected WLED device that accepts HTTP API commands. This module provides both low-level HTTP API access and high-level interfaces for controlling robot state-based lighting effects.

## WLED Device

The system uses a WLED-compatible ESP32/ESP8266 device with the following default configuration:
- **IP Address**: `10.0.1.166`
- **WLED Version**: Compatible with WLED 0.13+
- **Connection**: WiFi HTTP API

## Robot States and Effects

The light bar responds to different robot states with appropriate lighting effects:

### Available States
- **`startup`** - Boot sequence with rainbow effect
- **`idle`** - Peaceful breathing animation 
- **`speaking`** - Dynamic speech visualization
- **`listening`** - Pulsing blue effect
- **`thinking`** - Scanning yellow effect
- **`alert`** - Rapid red warning
- **`error`** - Flashing red error state
- **`shutdown`** - Fade out sequence

### Special Modes
- **`headlights`** - Solid white front lighting (overrides other states)
- **`audio_reactive`** - Dynamic response to audio levels

## Usage

### Simple Usage

```python
from light_bar.light_bar import light_bar

# Initialize and start the light bar
if light_bar.start():
    # Set robot states
    light_bar.set_robot_state("idle")
    light_bar.set_robot_state("speaking")
    light_bar.set_robot_state("alert")
    
    # Special functions
    light_bar.headlights_on()
    light_bar.audio_reactive(0.8)  # 80% intensity
    
    # Stop when done
    light_bar.stop()
```

### Advanced Usage with WLED Controller

```python
from light_bar.wled_controller import WLEDController
from light_bar.light_bar import RobotState

# Create controller with custom host
controller = WLEDController("http://10.0.1.166")

# Set specific effects
controller.set_robot_state(RobotState.SPEAKING)

# Direct WLED API access
controller.set_state({
    "on": True,
    "bri": 128,
    "seg": [{
        "fx": 12,  # Rainbow effect
        "sx": 128,  # Speed
        "ix": 128   # Intensity
    }]
})

# Headlights mode (overrides other effects)
controller.headlights_on()
controller.headlights_off()
```

### Legacy API Compatibility

The module maintains compatibility with previous USB CDC API:

```python
from light_bar.light_bar import rainbow_cycle, loading_bar, vu_meter

# These functions now send HTTP requests to WLED device
rainbow_cycle()
loading_bar(50)  # 50% progress
vu_meter(80)     # 80% volume
```

## Features

- **WiFi Communication**: HTTP-based control over WiFi network
- **State-Based Lighting**: Automatic effects based on robot behavior
- **Headlights Mode**: Override system for solid white lighting
- **Audio Reactive**: Dynamic response to audio input
- **Brightness Control**: Automatic brightness adjustment
- **Error Recovery**: Automatic reconnection and error handling
- **Thread-Safe**: Safe to use from multiple threads

## Configuration

### WLED Device Setup

1. Flash WLED firmware to ESP32/ESP8266
2. Connect to WiFi network
3. Configure static IP `10.0.1.166` (or update host in code)
4. Set up LED strip configuration in WLED web interface

### Network Configuration

Default WLED device configuration:
```
IP: 10.0.1.166
Port: 80 (HTTP)
API Endpoint: /json/state
```

To use a different IP address, update the host in your code:
```python
from light_bar.light_bar import light_bar
# Configure before starting
light_bar.controller.host = "http://YOUR_WLED_IP"
```

## WLED Effects Used

The system uses these WLED effects:
- **0**: Solid color
- **1**: Blink
- **12**: Rainbow
- **13**: Rainbow cycle
- **46**: Colorwaves
- **54**: Spots fade
- **110**: Flow

## Requirements

- `requests` - For HTTP communication
- WLED-compatible device on same network
- Python 3.7+

## Installation

```bash
pip install requests
```

## Network Setup

Ensure the WLED device and NerdBot are on the same WiFi network:

1. Configure WLED device WiFi through web interface
2. Set static IP or update code with device IP
3. Test connectivity: `ping 10.0.1.166`

## API Integration

The light bar integrates with Flask server endpoints:

```python
# Flask routes automatically control light bar
@app.route('/api/speak', methods=['POST'])
def speak():
    light_bar.set_robot_state("speaking")
    # ... handle speech
    light_bar.set_robot_state("idle")
```

## Troubleshooting

### Device Not Responding
1. Check WLED device is powered and connected to WiFi
2. Verify IP address is correct: `ping 10.0.1.166`
3. Check WLED web interface is accessible
4. Ensure no firewall blocking HTTP traffic

### Connection Issues
1. Verify network connectivity between devices
2. Check WLED device status LED
3. Try accessing WLED web interface directly
4. Restart WLED device if unresponsive

### Effect Issues
1. Check WLED firmware version (0.13+ recommended)
2. Verify LED strip configuration in WLED
3. Test effects through WLED web interface
4. Check log output for HTTP errors

### Headlights Not Working
1. Ensure headlights mode is enabled: `light_bar.headlights_on()`
2. Check headlights state: `light_bar.is_headlights_active()`
3. Verify WLED brightness settings
4. Test with direct WLED API call