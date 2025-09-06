# Audio Service

This service provides background audio playback for NerdBot, handling meme sounds and audio effects through a request-based system.

## Overview

The audio service runs as a background daemon that monitors for audio requests and plays sounds using pygame. It supports random meme sounds, specific sound files, and integrates with the light bar for visual effects.

## Features

- **Background Processing**: Runs as systemd service
- **Request-Based**: Uses JSON files for audio requests
- **Concurrent Playback**: Multiple sounds can play simultaneously
- **Light Bar Integration**: Triggers visual effects during playback
- **Auto-Discovery**: Automatically finds available meme sounds
- **Robust Error Handling**: Continues running even if individual sounds fail

## Audio Collection

The service manages 89+ meme sounds including:
- Classic memes (emotional-damage, among-us sounds)
- Cartoon effects (duck-toy-sound, cartoon sounds)
- System sounds (Windows startup, notification sounds)
- Voice clips (let-him-cook, hello-how-are-you)

## Service Management

### SystemD Service

The audio service runs as `nerdbot-audio.service`:

```bash
# Start/stop service
sudo systemctl start nerdbot-audio
sudo systemctl stop nerdbot-audio

# Enable/disable auto-start
sudo systemctl enable nerdbot-audio
sudo systemctl disable nerdbot-audio

# Check status
sudo systemctl status nerdbot-audio

# View logs
journalctl -u nerdbot-audio -f
tail -f /home/mark/nerdbot-audio.log
```

### Manual Operation

```bash
# Run manually for testing
cd /home/mark/nerdbot-backend
source setup_env.sh
python audio_service.py
```

## Request System

The service monitors `/tmp/nerdbot_audio/` for JSON request files:

### Random Sound Request
```json
{
    "type": "random",
    "timestamp": 1234567890
}
```

### Specific Sound Request  
```json
{
    "type": "specific", 
    "file_path": "duck-toy-sound.mp3",
    "timestamp": 1234567890
}
```

## Flask Integration

The Flask server creates audio requests:

```python
# Random meme sound
def play_random_meme():
    request = {
        'type': 'random',
        'timestamp': time.time()
    }
    request_file = Path(AUDIO_REQUEST_DIR) / f"random_{uuid.uuid4().hex[:8]}.json"
    with open(request_file, 'w') as f:
        json.dump(request, f)
    return True

# Specific sound
def play_specific_sound(file_path):
    request = {
        'type': 'specific',
        'file_path': file_path, 
        'timestamp': time.time()
    }
    request_file = Path(AUDIO_REQUEST_DIR) / f"specific_{uuid.uuid4().hex[:8]}.json"
    with open(request_file, 'w') as f:
        json.dump(request, f)
    return True
```

## API Endpoints

Audio is triggered through Flask API:

```bash
# Play random meme sound
curl -X POST http://localhost:5000/api/meme_sound/random

# Response
{
    "message": "FIRED: duck-toy-sound.mp3",
    "sound": "duck-toy-sound.mp3", 
    "status": "INSTANT_SUCCESS",
    "success": true
}
```

## Joystick Integration

- **Y Button (Button 5)**: Triggers random meme sound
- **Debounce**: 1 second between button presses
- **Non-blocking**: Button press immediately queues audio request

## File Structure

```
/home/mark/nerdbot-backend/
├── audio_service.py                 # Main service daemon
├── assets/meme_sounds_converted/    # Audio files (89+ sounds)
├── /tmp/nerdbot_audio/             # Request queue directory
└── /home/mark/nerdbot-audio.log    # Service logs
```

## Audio Processing

1. **Service monitors** `/tmp/nerdbot_audio/` directory
2. **Request detected** - JSON file appears
3. **Audio selected** - Random or specific file chosen
4. **Playback started** - Pygame spawns audio process
5. **Light effects** - WLED triggers visual effects  
6. **Request cleaned** - JSON file deleted
7. **Process tracked** - PID logged for monitoring

## Configuration

Key settings in `audio_service.py`:

```python
AUDIO_REQUEST_DIR = "/tmp/nerdbot_audio"
MEME_SOUNDS_FOLDER_CONVERTED = "/home/mark/nerdbot-backend/assets/meme_sounds_converted"
POLLING_INTERVAL = 0.1  # 100ms check rate
SUPPORTED_FORMATS = ['.mp3', '.wav', '.ogg']
```

## Requirements

- **pygame**: Audio playback engine
- **requests**: HTTP communication with Flask
- **pathlib**: File system operations  
- **json**: Request parsing
- **subprocess**: Process management

## Installation

Dependencies installed automatically with:

```bash
cd /home/mark/nerdbot-backend
source setup_env.sh
pip install -r requirements.txt
```

## Troubleshooting

### Service Not Starting
1. Check systemd service status: `sudo systemctl status nerdbot-audio`
2. Verify audio request directory exists: `ls /tmp/nerdbot_audio`
3. Check pygame installation in virtual environment
4. Review logs: `tail -f /home/mark/nerdbot-audio.log`

### No Audio Output
1. Verify PulseAudio/ALSA configuration
2. Check speaker volume: `pactl get-sink-volume @DEFAULT_SINK@`
3. Test pygame audio directly in Python shell
4. Ensure Anker speaker is connected and selected as default

### Sounds Not Playing from Joystick
1. Confirm nerdbot-joystick service is running
2. Check joystick logs: `tail -f /home/mark/nerdbot-joystick.log`
3. Verify Flask server is responding to API calls
4. Test API directly: `curl -X POST http://localhost:5000/api/meme_sound/random`

### Request Files Accumulating
1. Service may have crashed - restart: `sudo systemctl restart nerdbot-audio`
2. Check disk space in `/tmp/`
3. Manually clean old requests: `rm /tmp/nerdbot_audio/*.json`

## Performance

- **Polling Rate**: 100ms for responsive audio triggering
- **Concurrent Sounds**: Multiple pygame processes can run simultaneously
- **Memory Usage**: ~50MB base + ~10MB per active sound
- **Startup Time**: Service ready within 2 seconds
- **Response Time**: <100ms from button press to audio start

## Integration Points

The audio service integrates with:
- **Flask Server**: HTTP API endpoints for audio requests
- **Joystick Manager**: Y button triggers random memes
- **Light Bar**: Visual effects during audio playback
- **Web UI**: Soundboard interface for manual control
- **SystemD**: Auto-start and process management