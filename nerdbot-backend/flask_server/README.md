# Flask Server

The main HTTP API server for NerdBot, providing web endpoints for robot control, video streaming, AI inference, and hardware management.

## Overview

The Flask server acts as the central control hub for NerdBot, exposing REST APIs for all robot functionality including video streaming, AI-powered object detection, motor control, servo movement, lighting effects, and audio playback.

## Key Features

- **Video Streaming**: Multi-camera support with real-time streaming
- **AI Inference**: Object detection, pose estimation, instance segmentation
- **Hardware Control**: Motors, servos, lasers, lights, audio
- **Web Interface**: RESTful APIs for web and mobile clients
- **Real-time Communication**: WebSocket support for live updates
- **Security**: CORS handling and request validation

## API Endpoints

### Video & AI
- `GET /cam0`, `/cam1` - Camera streams
- `GET /api/visual_awareness` - AI scene analysis
- `POST /api/detection` - Object detection on image
- `GET /api/inference_status` - AI inference statistics

### Robot Control  
- `POST /api/move/*` - Motor movement commands
- `POST /api/servos/*` - Camera pan/tilt control
- `POST /api/speak` - Text-to-speech with lip sync
- `GET /api/vitals` - Robot health status

### Lighting & Effects
- `POST /api/headlights/*` - Headlight control
- `POST /api/laser/*` - Laser pointer control
- `POST /api/light_bar/*` - LED strip effects

### Audio
- `POST /api/meme_sound/random` - Random meme sound
- `POST /api/meme_sound/specific` - Play specific sound
- `GET /api/audio/status` - Audio system status

### System
- `GET /api/system/stats` - CPU, memory, temperature
- `POST /api/system/shutdown` - Safe system shutdown
- `GET /api/battery` - Battery level and charging status

## Configuration

Key settings in `server.py`:

```python
# Server Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000
DEBUG = False

# AI Models
DETECTION_MODEL = "yolov8n.hef"
POSE_MODEL = "yolov8n-pose.hef"
SEGMENTATION_MODEL = "yolov8n-seg.hef"

# Hardware Pins
LASER_PIN = 21
SERVO_PAN_PIN = 18  
SERVO_TILT_PIN = 19

# Network
WLED_HOST = "http://10.0.1.166"
CAMERA_RESOLUTION = (640, 480)
```

## Startup Sequence

1. **Environment Setup**: Load virtual environment and dependencies
2. **Hardware Initialization**: Connect to cameras, GPIO, servos
3. **AI Model Loading**: Initialize Hailo AI models
4. **Service Integration**: Connect to audio service, light bar
5. **Server Launch**: Start Flask on port 5000
6. **Health Checks**: Verify all systems operational

## AI Integration

The server uses Hailo AI acceleration:

```python
# Detection Pipeline
detection_pipeline = DetectionPipeline("yolov8n.hef")
results = detection_pipeline.run(image_path)

# Pose Estimation
pose_pipeline = PoseEstimationPipeline("yolov8n-pose.hef") 
poses = pose_pipeline.run(image_path)

# Instance Segmentation  
segmentation_pipeline = InstanceSegmentationPipeline("yolov8n-seg.hef")
masks = segmentation_pipeline.run(image_path)
```

## Video Streaming

Multi-camera streaming with format negotiation:

```python
# Camera initialization
camera0 = Picamera2(0)  # Front camera
camera1 = Picamera2(1)  # Rear camera (optional)

# Stream configuration
config = camera.create_video_configuration(
    main={"size": (640, 480)},
    lores={"size": (320, 240)}
)

# MJPEG streaming
@app.route('/cam0')
def camera0_stream():
    return Response(generate_frames(camera0),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
```

## Hardware Integration

### Motor Control
```python
@app.route('/api/move/forward', methods=['POST'])
def move_forward():
    duration = request.json.get('duration', 1.0)
    motor_manager.move_forward(duration)
    return jsonify({'status': 'success'})
```

### Servo Control  
```python
@app.route('/api/servos/pan/left', methods=['POST'])
def pan_left():
    servo_manager.pan_left()
    return jsonify({'status': 'panned left'})
```

### Laser Control
```python
@app.route('/api/laser/toggle', methods=['POST'])
def toggle_laser():
    result = laser_control.toggle_laser()
    status = "on" if laser_control.is_laser_active() else "off"
    return jsonify({'laser_on': laser_control.is_laser_active()})
```

## Service Dependencies

The Flask server coordinates with:

- **nerdbot-audio**: Audio playback service
- **WLED Controller**: WiFi LED strip control  
- **Hailo AI**: Hardware-accelerated inference
- **PiCamera2**: Camera interface
- **GPIO Hardware**: Motors, servos, laser, sensors

## SystemD Service

Managed by `nerdbot-flask.service`:

```bash
# Service management
sudo systemctl start nerdbot-flask
sudo systemctl status nerdbot-flask
sudo systemctl restart nerdbot-flask

# Logs
journalctl -u nerdbot-flask -f
tail -f /var/log/nerdbot-flask.log
```

## Development

### Local Development
```bash
cd /home/mark/nerdbot-backend
source setup_env.sh
python -m flask_server.server
```

### API Testing
```bash
# Health check
curl http://localhost:5000/api/vitals

# Camera test
curl http://localhost:5000/cam0

# Move robot
curl -X POST http://localhost:5000/api/move/forward \
     -H "Content-Type: application/json" \
     -d '{"duration": 2.0}'
```

## Performance

- **Startup Time**: ~10 seconds (including AI model loading)
- **Response Time**: <50ms for simple API calls
- **Video Latency**: <200ms for local network streaming
- **AI Inference**: 20-30 FPS object detection
- **Memory Usage**: ~2GB (includes AI models)
- **CPU Usage**: 40-60% during active inference

## Security

- **CORS Headers**: Configured for cross-origin requests
- **Input Validation**: Request parameter validation
- **Rate Limiting**: Built-in Flask throttling
- **Network Isolation**: Runs on private network
- **No Authentication**: Designed for trusted local network

## Error Handling

```python
# Graceful degradation
try:
    result = ai_pipeline.run(image)
except Exception as e:
    logging.error(f"AI inference failed: {e}")
    return jsonify({'error': 'AI unavailable'}), 503

# Hardware fallbacks
if not laser_control:
    return jsonify({'error': 'Laser control unavailable'}), 503
```

## Troubleshooting

### Server Won't Start
1. Check port 5000 availability: `netstat -tlnp | grep 5000`
2. Verify virtual environment: `source setup_env.sh`
3. Check hardware connections (cameras, GPIO)
4. Review startup logs: `journalctl -u nerdbot-flask -n 50`

### Camera Issues
1. Test cameras: `libcamera-hello --list-cameras`
2. Check camera permissions: `ls -l /dev/video*`  
3. Verify camera modules enabled in raspi-config
4. Try different camera indices (0, 1)

### AI Inference Errors
1. Confirm Hailo hardware connected: `lsusb | grep Hailo`
2. Check model files exist: `ls resources/*.hef`
3. Verify TAPPAS environment: `echo $TAPPAS_POST_PROC_DIR`
4. Test basic pipeline: `python basic_pipelines/detection.py`

### GPIO Errors
1. Check user permissions: `groups $USER | grep gpio`
2. Verify pin connections and wiring
3. Test individual components (laser, servos)
4. Check for pin conflicts with other services

## Integration

The Flask server is the primary integration point:
- **Web UI**: React frontend connects to Flask APIs
- **Joystick Service**: Uses Flask APIs for hardware control
- **Audio Service**: Receives requests via Flask endpoints  
- **Mobile Apps**: Can connect to Flask REST APIs
- **External Systems**: HTTP API allows third-party integration