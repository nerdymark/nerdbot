# Servo Control Module

This module provides PWM-based control for pan/tilt servos that control the NerdBot's camera gimbal.

## Overview

The servo control system manages two servos for camera pan and tilt operations. It provides smooth movement, position tracking, and integration with joystick controls and web API.

## Hardware Setup

- **Pan Servo**: Connected to GPIO pin 18 (PWM0)
- **Tilt Servo**: Connected to GPIO pin 19 (PWM1)
- **Power**: 5V external power supply recommended for servos
- **Control**: 3.3V PWM signals from Raspberry Pi

## Features

- **Dual Servo Control**: Independent pan and tilt movement
- **Position Tracking**: Remembers current servo positions
- **Smooth Movement**: Configurable step size for smooth transitions
- **Centering Function**: Return servos to default center position
- **Joystick Integration**: Real-time control via Steam Controller
- **Web API**: HTTP endpoints for remote control

## Usage

### Direct Control

```python
from servo_control.servos import servo_manager

# Initialize servo control
servo_manager.start()

# Pan left/right (0-180 degrees)
servo_manager.pan_left()
servo_manager.pan_right()
servo_manager.set_pan_position(90)  # Center

# Tilt up/down (0-180 degrees)
servo_manager.tilt_up()
servo_manager.tilt_down()
servo_manager.set_tilt_position(90)  # Center

# Center both servos
servo_manager.center_servos()
servo_manager.pan_center()  # Pan only

# Get current positions
pan_pos = servo_manager.get_pan_position()
tilt_pos = servo_manager.get_tilt_position()

# Stop servos (releases PWM control)
servo_manager.stop()
```

### Flask API Endpoints

The servo system integrates with Flask server:

- **`POST /api/servos/pan/left`** - Pan camera left
- **`POST /api/servos/pan/right`** - Pan camera right  
- **`POST /api/servos/tilt/up`** - Tilt camera up
- **`POST /api/servos/tilt/down`** - Tilt camera down
- **`POST /api/servos/center`** - Center both servos
- **`GET /api/servos/position`** - Get current positions

### Joystick Control

- **Left Analog Stick**: Real-time pan/tilt control
  - X-axis: Pan left/right
  - Y-axis: Tilt up/down
- **A Button**: Center pan servo (preserves tilt tracking)

## Configuration

Servo settings in `servos.py`:

```python
# GPIO Pins
PAN_PIN = 18    # PWM0
TILT_PIN = 19   # PWM1

# Position Limits
MIN_ANGLE = 0
MAX_ANGLE = 180
CENTER_ANGLE = 90

# Movement Settings  
STEP_SIZE = 5     # Degrees per step
UPDATE_DELAY = 0.05  # Seconds between steps
```

## Movement Parameters

- **Range**: 0-180 degrees for both servos
- **Center Position**: 90 degrees
- **Step Size**: 5 degrees per movement command
- **PWM Frequency**: 50Hz (standard servo frequency)

## API Examples

```bash
# Pan camera left
curl -X POST http://localhost:5000/api/servos/pan/left

# Tilt camera up  
curl -X POST http://localhost:5000/api/servos/tilt/up

# Center both servos
curl -X POST http://localhost:5000/api/servos/center

# Get current positions
curl http://localhost:5000/api/servos/position
```

## Requirements

- `gpiozero` - GPIO control library
- Raspberry Pi with PWM-capable GPIO pins
- Two servo motors (SG90 or similar)
- External 5V power supply for servos

## Installation

```bash
pip install gpiozero
```

## Wiring

```
Pan Servo (GPIO 18):
├── Red Wire → 5V External Power
├── Brown Wire → Ground (common with Pi)
└── Orange Wire → GPIO 18

Tilt Servo (GPIO 19):
├── Red Wire → 5V External Power  
├── Brown Wire → Ground (common with Pi)
└── Orange Wire → GPIO 19
```

**Important**: Use external 5V power supply for servos. Connect grounds together but don't power servos from Pi's 5V rail.

## Troubleshooting

### Servo Jitter
- Check power supply capacity (servos need adequate current)
- Verify ground connections between Pi and servo power
- Ensure PWM frequency is set to 50Hz

### No Movement
- Verify GPIO pin connections
- Check servo power supply
- Test servos individually with simple PWM commands

### Position Drift
- Servos may drift slightly when not actively controlled
- This is normal behavior - use `stop()` to release PWM control
- Re-center periodically if needed for precision applications

### Permission Errors
Add user to gpio group:
```bash
sudo usermod -a -G gpio $USER
```

## Integration

The servo control integrates with:
- **Flask Server**: Web API for remote control
- **Joystick Manager**: Steam Controller analog stick input
- **Video Streaming**: Camera follows servo movement
- **Web UI**: Pan/tilt controls in React interface

## Performance Notes

- Servos update at ~20Hz for smooth movement
- Position tracking maintains state across sessions
- PWM control is released when servos are idle to prevent overheating