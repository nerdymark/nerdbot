# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-project repository containing:
- **nerdbot-backend**: A Flask-based Python backend with Hailo AI processor integration for Raspberry Pi 5, featuring object detection, pose estimation, and instance segmentation
- **nerdbot-ui**: A React frontend built with Vite
- **TTS**: Coqui TTS (Text-to-Speech) library for speech synthesis
- **hailo_model_zoo**: Pre-trained models and training utilities for Hailo AI processors
- **darkice**: Audio streaming software

## Key Development Commands

### Backend (nerdbot-backend)

```bash
# Setup environment and install dependencies
cd ~/nerdbot-backend
source setup_env.sh
./install.sh

# Run Flask server
python -m flask_server.server

# Run tests
./run_tests.sh

# Build C++ postprocessing libraries
meson setup build.release --release
ninja -C build.release
```

### Frontend (nerdbot-ui)

```bash
cd ~/nerdbot-ui
npm install
npm run dev      # Development server
npm run build    # Production build
npm run lint     # Run linter
```

### TTS

```bash
cd ~/TTS
pip install -e .[all,dev,notebooks]
# Run TTS server
python TTS/server/server.py --list_models
python TTS/server/server.py --model_name tts_models/en/vctk/vits
```

## Architecture Overview

### nerdbot-backend
- **Flask API server** (`flask_server/server.py`): Main application handling video streaming, AI inference, motor/servo control
- **Basic Pipelines** (`basic_pipelines/`): Python implementations for detection, pose estimation, and instance segmentation using Hailo AI
- **Hardware Control**: Modules for motors (`motor_control/`), servos (`servo_control/`), battery monitoring (`x120/`)
- **C++ Libraries** (`cpp/`): High-performance post-processing for YOLO models
- **Resources** (`resources/`): Pre-trained Hailo model files (.hef), configuration JSONs

### Key Dependencies
- Hailo Runtime and TAPPAS for AI acceleration
- OpenCV for image processing
- Flask/Flask-CORS for web API
- picamera2 for Raspberry Pi camera interface
- pyaudio/pydub for audio processing

### Joystick Control System
- **Integrated Joystick Service** (`integrated_joystick_service.py`): Zero-latency Steam Controller integration
- **Input Manager** (`joystick_input_manager.py`): Handles Steam Controller input processing
- **Hardware Integration**: Direct motor and servo control bypassing HTTP API

#### Steam Controller Mapping:
- **Left analog stick**: Pan/tilt camera servos (fine control)
- **Left touchpad (with groove)**: Robot movement (forward/back/strafe left/right)
- **Right circular touchpad**: Robot rotation (distance from center = speed)
- **A button**: Center servos to default position
- **B button**: Emergency stop all motors

#### Joystick Service Management:
```bash
# Start joystick service
sudo systemctl start nerdbot-joystick

# Check service status
sudo systemctl status nerdbot-joystick

# View live logs
journalctl -u nerdbot-joystick -f

# Enable auto-start on boot
sudo systemctl enable nerdbot-joystick
```

### Startup Scripts
Located in `nerdbot-scripts/`:
- `flask-startup.sh`: Starts the Flask backend server
- `vite-react-startup.sh`: Starts the React development server
- `tts-startup.sh`: Starts the TTS service
- `darkice-startup.sh`: Starts audio streaming

### System Services
Located in `systemd-services/`:
- `nerdbot-flask.service`: Main Flask application
- `nerdbot-joystick.service`: Integrated joystick control
- `nerdbot-ui.service`: React frontend
- `nerdbot-tts.service`: Text-to-speech service
- `nerdbot-darkice.service`: Audio streaming
- `nerdbot-fan.service`: System cooling

Install all services: `sudo ./install-services.sh`

## Important Notes

- The project requires Hailo AI processor (Hailo8/8L) connected to Raspberry Pi 5
- Virtual environment is automatically managed by `setup_env.sh`
- Tests require downloading model resources via `./download_resources.sh --all`
- The system supports multiple AI models for detection, segmentation, and pose estimation
- Audio features include meme sound playback and TTS integration
- Steam Controller provides zero-latency robot control with intuitive mapping
- All services are configured for automatic startup and proper dependency management