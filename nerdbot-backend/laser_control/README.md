# Laser Control Module

This module provides GPIO-based relay control for a laser pointer connected to the NerdBot.

## Overview

The laser control system uses a relay module connected to GPIO pin 21 to control power to a laser pointer. The relay acts as an electronic switch, controlling a separate 5V power supply that drives the laser, providing safe isolation between the Raspberry Pi GPIO and the laser's power circuit.

## Hardware Setup

- **Laser Module**: Powered by 12V battery stepped down to 5V through relay
- **Relay Module**: Electronic switch controlled by GPIO pin 21
- **GPIO Pin**: Pin 21 (BCM numbering) - 3.3V control signal
- **Power Supply**: 12V battery → 5V step-down adapter → relay → laser
- **Isolation**: Relay provides electrical isolation between Pi and laser circuit

## Features

- **Safe GPIO Control**: Uses gpiozero library for reliable GPIO operations
- **State Management**: Tracks laser on/off state
- **Web API Integration**: Flask endpoints for remote control
- **Joystick Integration**: Direct control via Steam Controller
- **Simulation Mode**: Falls back to simulation if GPIO unavailable

## Usage

### Direct Control

```python
from laser_control.laser_control import LaserControl

# Initialize laser control
laser = LaserControl()

# Turn laser on/off
laser.activate_laser()
laser.deactivate_laser()

# Toggle laser state
laser.toggle_laser()

# Check current state
is_on = laser.is_laser_active()

# Cleanup when done
laser.cleanup()
```

### Flask API Endpoints

The laser integrates with the Flask server providing these endpoints:

- **`GET /api/laser/status`** - Get current laser state
- **`POST /api/laser/toggle`** - Toggle laser on/off
- **`POST /api/laser/on`** - Turn laser on
- **`POST /api/laser/off`** - Turn laser off

### Joystick Control

- **Button 11**: Toggle laser on/off (500ms debounce)

## API Examples

```bash
# Check laser status
curl http://localhost:5000/api/laser/status

# Toggle laser
curl -X POST http://localhost:5000/api/laser/toggle

# Turn laser on
curl -X POST http://localhost:5000/api/laser/on

# Turn laser off
curl -X POST http://localhost:5000/api/laser/off
```

## Configuration

Default settings can be modified in `laser_control.py`:

```python
LASER_PIN = 21  # GPIO pin for laser relay
```

## Safety Features

- **Electrical Isolation**: Relay provides complete isolation between Pi and laser circuit
- **Battery Power**: Laser powered by 12V battery through 5V step-down adapter
- **Low Voltage Control**: Pi GPIO only switches 3.3V control signal to relay
- **State Tracking**: Always knows current laser state
- **Graceful Cleanup**: Proper GPIO cleanup on shutdown
- **Error Handling**: Handles GPIO errors gracefully

## Requirements

- `gpiozero` - GPIO control library
- Raspberry Pi with GPIO access
- Relay module connected to GPIO pin 21

## Installation

```bash
pip install gpiozero
```

## Troubleshooting

### GPIO Busy Error
If you get "GPIO busy" errors:
1. Make sure no other process is using GPIO pin 21
2. Check if laser control is already initialized
3. Ensure proper cleanup of previous instances

### Permission Errors
Add your user to the gpio group:
```bash
sudo usermod -a -G gpio $USER
```

### Simulation Mode
If GPIO is unavailable, the system falls back to simulation mode:
- All commands will be logged but no actual GPIO control occurs
- Useful for development and testing

## Integration

The laser control integrates with:
- **Flask Server**: Web API endpoints
- **Joystick Manager**: Steam Controller button 11
- **Web UI**: Toggle button in React interface