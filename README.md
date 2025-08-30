# NerdBot

A multi-project robotics system built on Raspberry Pi 5 with Hailo AI acceleration, featuring zero-latency Steam Controller integration, computer vision, and autonomous capabilities.

## Project Structure

- **nerdbot-backend**: Flask-based Python backend with Hailo AI processor integration
- **nerdbot-ui**: React frontend built with Vite
- **TTS**: Coqui TTS (Text-to-Speech) library for speech synthesis
- **hailo_model_zoo**: Pre-trained models and training utilities for Hailo AI processors (excluded from git)
- **darkice**: Audio streaming software
- **systemd-services**: System service configurations
- **nerdbot-scripts**: Startup and utility scripts
- **x120x**: Battery monitoring and power management

## Features

### AI-Powered Vision
- Real-time object detection using Hailo AI acceleration
- Pose estimation and instance segmentation
- Optimized C++ post-processing for YOLO models
- Multiple AI model support (detection, segmentation, pose estimation)

### Zero-Latency Control
- **Steam Controller Integration**: Direct hardware control bypassing HTTP API
- **Mecanum Wheel Drive**: 6-wheel rover with omnidirectional movement
- **Pan/Tilt Camera System**: Servo-controlled camera with smooth tracking
- **Emergency Stop**: Instant motor cutoff for safety

### Audio & Communication
- Meme sound playback system
- Text-to-Speech integration
- Audio streaming via DarkIce
- Real-time communication capabilities

## Steam Controller Mapping

| Control | Function |
|---------|----------|
| Left Analog Stick | Pan/tilt camera servos (fine control) |
| Left Touchpad (groove) | Robot movement (forward/back/strafe) |
| Right Circular Touchpad | Robot rotation (radial speed control) |
| A Button | Center servos to default position |
| B Button | Emergency stop all motors |

## Quick Start

### 1. Setup Environment
```bash
cd ~/nerdbot-backend
source setup_env.sh
./install.sh
```

### 2. Install System Services
```bash
cd ~/systemd-services
sudo ./install-services.sh
sudo systemctl enable nerdbot-flask nerdbot-joystick nerdbot-ui
```

### 3. Start Services
```bash
sudo systemctl start nerdbot-flask
sudo systemctl start nerdbot-joystick
sudo systemctl start nerdbot-ui
```

### 4. Access Web Interface
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Development Commands

### Backend
```bash
cd ~/nerdbot-backend
python -m flask_server.server    # Run Flask server
./run_tests.sh                   # Run tests
meson setup build.release        # Build C++ libraries
ninja -C build.release
```

### Frontend
```bash
cd ~/nerdbot-ui
npm install
npm run dev      # Development server
npm run build    # Production build
npm run lint     # Run linter
```

### Joystick Service
```bash
sudo systemctl status nerdbot-joystick    # Check status
journalctl -u nerdbot-joystick -f         # View live logs
sudo systemctl restart nerdbot-joystick   # Restart service
```

## System Requirements

### Hardware
- **Raspberry Pi 5**: Main computing platform
- **Hailo AI Processor**: Hailo8/8L for AI acceleration
- **Steam Controller**: Wireless controller for robot control
- **Camera Module**: Compatible with picamera2
- **Motor Controllers**: Dual Adafruit Motor HATs (addresses 0x60, 0x61)
- **Servo Controller**: Adafruit ServoKit (address 0x42)
- **Mecanum Wheels**: 6-wheel omnidirectional drive system

### Software Dependencies
- **Hailo Runtime & TAPPAS**: AI acceleration framework
- **OpenCV**: Computer vision processing
- **Flask/Flask-CORS**: Web API framework
- **React**: Frontend user interface
- **pygame**: Controller input handling
- **picamera2**: Camera interface

## Architecture

### Motor Control
- **6-Wheel Mecanum Drive**: Omnidirectional movement capability
- **Dual Motor HAT Setup**: 3 motors per side for smooth operation
- **Zero-Latency Control**: Direct joystick integration bypasses HTTP overhead
- **Safety Features**: Emergency stop and automatic timeout

### Camera System
- **Pan/Tilt Servos**: Smooth camera positioning
- **Real-time Streaming**: Live video feed with AI overlay
- **Object Tracking**: Automatic target following
- **Servo Boundary Protection**: Prevents over-rotation damage

### AI Processing
- **Hailo Acceleration**: Hardware-optimized inference
- **Multiple Model Support**: Detection, segmentation, pose estimation
- **Real-time Performance**: Low-latency processing pipeline
- **Configurable Models**: Easy model switching and updates

## Service Management

All services are configured for automatic startup with proper dependencies:

```bash
# View all service status
sudo systemctl status nerdbot-*

# Stop all services
sudo systemctl stop nerdbot-*

# Restart specific service
sudo systemctl restart nerdbot-joystick

# View service logs
journalctl -u nerdbot-flask -f
```

## Troubleshooting

### Joystick Not Responding
```bash
# Check if controller is detected
ls /dev/input/js*

# Verify service is running
sudo systemctl status nerdbot-joystick

# Check service logs
journalctl -u nerdbot-joystick -f
```

### Camera Issues
```bash
# Test camera access
libcamera-hello --list-cameras

# Check Flask service logs
journalctl -u nerdbot-flask -f
```

### AI Model Loading
```bash
# Download required models
cd ~/nerdbot-backend
./download_resources.sh --all

# Check Hailo runtime
hailortcli fw-control identify
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly on hardware
5. Submit a pull request

## License

[License information to be added]

## Contact

[Contact information to be added]
