# NerdBot

A multi-project robotics system built on Raspberry Pi 5 with Hailo AI acceleration, featuring zero-latency Steam Controller integration, computer vision, and autonomous capabilities.

## 🎮 Steam Controller Integration

This project now features comprehensive Steam Controller integration with zero-latency robot control, bypassing HTTP API overhead for responsive real-time operation.

### Steam Controller Mapping

| Control | Function | Description |
|---------|----------|-------------|
| **Left Analog Stick** | Pan/Tilt Camera | Fine servo control for camera positioning |
| **Left Touchpad (groove)** | Robot Movement | Forward/back/strafe with WASD-style control |
| **Right Circular Touchpad** | Robot Rotation | Radial speed control (distance from center = speed) |
| **A Button** | Center Servos | Reset camera to default position |
| **B Button** | Emergency Stop | Immediate motor shutdown for safety |

## Project Structure

- **nerdbot-backend**: Flask-based Python backend with Hailo AI processor integration and joystick control
- **nerdbot-ui**: React frontend built with Vite
- **systemd-services**: System service configurations including joystick service
- **nerdbot-scripts**: Startup and utility scripts  
- **darkice**: Audio streaming software

## Key Features

### Zero-Latency Control
- **Direct Hardware Integration**: Bypasses HTTP API for instant response
- **100Hz Update Rate**: Smooth, responsive control experience
- **Safety Features**: Emergency stop and proper error handling

### AI-Powered Vision
- Real-time object detection using Hailo AI acceleration
- Pose estimation and instance segmentation
- Multiple AI model support

### Hardware Control
- **6-Wheel Mecanum Drive**: Omnidirectional movement capability
- **Pan/Tilt Camera System**: Servo-controlled camera with smooth tracking
- **Emergency Safety**: Instant motor cutoff and boundary protection

## Quick Start

### 1. Setup Environment
```bash
cd ~/nerdbot-backend
source setup_env.sh
./install.sh
```

### 2. Install Services
```bash
cd ~/systemd-services
sudo ./install-services.sh
sudo systemctl enable nerdbot-joystick nerdbot-flask
```

### 3. Start Joystick Control
```bash
sudo systemctl start nerdbot-joystick
sudo systemctl status nerdbot-joystick
```

## Joystick Service Management

```bash
# Check service status
sudo systemctl status nerdbot-joystick

# View live logs  
journalctl -u nerdbot-joystick -f

# Restart service
sudo systemctl restart nerdbot-joystick
```

## Hardware Requirements

- **Raspberry Pi 5**: Main computing platform
- **Hailo AI Processor**: Hailo8/8L for AI acceleration  
- **Steam Controller**: Wireless controller for robot control
- **Motor Controllers**: Dual Adafruit Motor HATs (addresses 0x60, 0x61)
- **Servo Controller**: Adafruit ServoKit (address 0x42)
- **Mecanum Wheels**: 6-wheel omnidirectional drive system

## Technical Implementation

### New Components Added
- `integrated_joystick_service.py`: Main Steam Controller integration service
- `joystick_input_manager.py`: Input processing and event handling system
- `systemd-services/nerdbot-joystick.service`: Service configuration

### Enhanced Modules
- **Motor Control**: Added joystick handlers with movement, strafing, rotation
- **Servo Control**: Enhanced with boundary checking and safety features
- **Service Integration**: Complete systemd integration with auto-startup

🤖 The Steam Controller now provides console-quality robot control with enterprise-grade reliability!
