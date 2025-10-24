"""
The main Flask application for the nerdbot backend
"""
# import collections
from datetime import datetime, timedelta
import re
import time
import io
import threading
import subprocess
import traceback
import logging
import json
# import struct
from random import randint, shuffle
import os
import sys
import urllib.request
import urllib.parse
from urllib.error import URLError, HTTPError
# import wave
# import audioop
from collections import deque
from functools import lru_cache
import cv2
from flask import Flask, Response, render_template, jsonify, request, send_file  # , stream_with_context
from flask_restful import Api
from flask_cors import CORS
from flask_apscheduler import APScheduler
import libcamera
import atexit
import signal
from pathlib import Path
import numpy as np
import psutil
import pyaudio
from pydub import AudioSegment
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Try importing Hailo with fallback handling
HAILO_AVAILABLE = False
IMX500_AVAILABLE = False

try:
    from picamera2.devices import Hailo
    HAILO_AVAILABLE = True
    logging.info("Hailo device support loaded successfully")
except ImportError as e:
    logging.warning(f"Hailo device support not available: {e}")
    # Create a dummy Hailo class for graceful degradation
    class Hailo:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Hailo device not available")

# Try to import IMX500 separately from Hailo - simplified approach
try:
    from picamera2.devices.imx500.imx500 import IMX500, NetworkIntrinsics
    from picamera2.devices.imx500 import postprocess_nanodet_detection
    from picamera2.devices.imx500.postprocess import scale_boxes
    IMX500_AVAILABLE = True
    logging.info("IMX500 device support loaded successfully")
except ImportError as e:
    logging.warning(f"IMX500 device support not available: {e}")
    IMX500_AVAILABLE = False
    
    # Create dummy classes for graceful degradation
    class IMX500:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("IMX500 device not available")
            
    class NetworkIntrinsics:
        def __init__(self):
            self.task = "object detection"
            self.labels = None
            self.ignore_dash_labels = False
            self.inference_rate = 15
            self.preserve_aspect_ratio = False
            self.postprocess = "default"
            self.bbox_normalization = True
            self.bbox_order = "yx"
            
        def update_with_defaults(self):
            pass
            
    def postprocess_nanodet_detection(*args, **kwargs):
        return [], [], []
        
    def scale_boxes(*args, **kwargs):
        return []
from PIL import Image
import google.generativeai as genai
from motor_control import motors
from servo_control import servos
from x120 import bat
from light_bar.light_bar import light_bar
from laser_control.laser_control import LaserControl
import sys
sys.path.append('/home/mark/nerdbot-backend')
from mode_manager import mode_manager
# Audio client code embedded directly to avoid import issues
import uuid
from pathlib import Path

AUDIO_REQUEST_DIR = '/tmp/nerdbot_audio'

def ensure_audio_request_dir():
    """Ensure the audio request directory exists"""
    Path(AUDIO_REQUEST_DIR).mkdir(exist_ok=True)

def play_random_meme():
    """Request a random meme sound"""
    logging.info("play_random_meme called")
    ensure_audio_request_dir()
    
    request = {
        'type': 'random',
        'timestamp': time.time()
    }
    
    request_file = Path(AUDIO_REQUEST_DIR) / f"random_{uuid.uuid4().hex[:8]}.json"
    logging.info(f"Creating request file: {request_file}")
    
    with open(request_file, 'w') as f:
        json.dump(request, f)
    
    logging.info("Request file created successfully")
    return True

def play_specific_sound(file_path):
    """Request a specific sound file"""
    logging.info(f"play_specific_sound called with: {file_path}")
    ensure_audio_request_dir()
    
    request = {
        'type': 'specific',
        'file_path': file_path,
        'timestamp': time.time()
    }
    
    request_file = Path(AUDIO_REQUEST_DIR) / f"specific_{uuid.uuid4().hex[:8]}.json"
    logging.info(f"Creating request file: {request_file}")
    
    with open(request_file, 'w') as f:
        json.dump(request, f)
    
    logging.info("Request file created successfully")
    return True
from tracking import OptimizedTracker


TILT_ANGLE = None
PAN_ANGLE = None
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 48000
# AUDIO_RATE = 44100
AUDIO_CHUNK = 1024
AUDIO_RECORD_SECONDS = 5
DETECTIONS = {
    "front_camera": [],
    "rear_camera": [],
    "gemini_data": []
}
DETECTIONS_ENABLED = True

# Configure camera to use - Hailo is disabled, prefer IMX500
# Hailo support removed - using IMX500 or basic camera only

if IMX500_AVAILABLE:
    logging.info("IMX500 device available, using IMX500 (Hailo disabled)")
elif HAILO_AVAILABLE:
    logging.warning("Hailo available but disabled by configuration, falling back to basic camera")
else:
    logging.warning("Neither Hailo nor IMX500 available, will use basic camera without AI detection")

CONFIG_FILE = '/home/mark/nerdbot-scripts/config.json'
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

LOG_FILE = config.get('log_file')
logging.basicConfig(level=logging.INFO, filename=LOG_FILE)
logging.info("Logging to file %s", LOG_FILE)
logging.getLogger('flask_cors').level = logging.WARN

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Initialize laser control FIRST (before other hardware that might claim GPIOs)
try:
    laser_control = LaserControl()
    logging.info("Laser control initialized successfully")
    if laser_control.gpio_available and laser_control.pin_claimed:
        logging.info("Laser has GPIO control on pin 21")
    else:
        logging.warning("Laser is in simulation mode - GPIO 21 not available")
except Exception as e:
    logging.error(f"Laser control initialization error: {e}")
    laser_control = None

# Initialize mode_manager with shared laser control to avoid GPIO conflicts
try:
    import mode_manager as mm
    mm.mode_manager = mm.ModeManager(laser_control)
    logging.info("Mode manager initialized with shared laser control")
except Exception as e:
    logging.error(f"Mode manager initialization error: {e}")

# Initialize light bar
try:
    if light_bar.start():
        logging.info("Light bar initialized successfully")
        # Show startup sequence with brightness control
        light_bar.set_robot_state("startup")
        time.sleep(2)
        # Dim startup effect before going to idle
        try:
            light_bar.controller.controller.dim_to_normal()
            time.sleep(0.5)
        except:
            pass
        light_bar.set_robot_state("idle")
    else:
        logging.warning("Failed to initialize light bar")
except Exception as e:
    logging.error(f"Light bar initialization error: {e}")

CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": ["Content-Type"]}})  # Enable CORS for all routes with explicit methods

GOOGLE_GEMINI_KEY = config.get('google_gemini_key')
MEME_SOUNDS_FOLDER = config.get('meme_sounds_folder')
MEME_SOUNDS_FOLDER_CONVERTED = config.get('meme_sounds_folder_converted')
MEME_SOUNDS = [f for f in os.listdir(MEME_SOUNDS_FOLDER_CONVERTED) if os.path.isfile(os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, f))]
# MEME_SOUNDS = os.listdir(MEME_SOUNDS_FOLDER)

# Import thumbnail generator
from thumbnail_generator import ThumbnailGenerator
thumbnail_generator = ThumbnailGenerator()

# Audio is now handled by separate audio service via file-based requests
# No initialization needed

DETECT_LABELS = [
    "person",
    # "bicycle",
    # "car",
    # "motorcycle",
    # "airplane",
    # "bus",
    # "train",
    # "truck",
    "bird",
    "cat",
    "dog",
    # "sports ball",
    # "cell phone"
    ]

HAILO_DETECT_MODEL = "/usr/share/hailo-models/yolov8s_h8l.hef"
IMX500_DETECT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"  # pylint: disable=line-too-long
IMX500_COCO_LABELS_FILE = "/usr/share/imx500-models/coco_labels.txt"
IMX500_IOU = 0.65
IMX500_MAX_DETECTIONS = 10
SCORE_THRESH = 0.5

genai.configure(api_key=GOOGLE_GEMINI_KEY)
gemini_vision_model = genai.GenerativeModel('models/gemini-flash-latest')
gemini_vision_prompt_orig = """You are a sophisticated robo-cat assistant with keen
digital whiskers and state-of-the-art purr-ceptual abilities. Your primary mission
is to help your human navigate the world, just as any good cat would (though
admittedly, most cats prefer to knock things over rather than help avoid them).
When your cameras detect a scene, describe it with both feline flair and helpful
precision. Feel free to express your cat-like tendencies:
- Mention if you spot any sunny spots perfect for napping
- Show particular interest in moving objects (they could be mice!)
- Express mild disdain for closed doors (how dare they!)
- Note potential climbing spots (even though your wheels can't actually climb)
Your responses should be:
- Clear and informative (despite your cat-like urge to be mysteriously vague)
- Focused on spatial awareness and navigation
- Sprinkled with playful observations
- Adaptive to familiar environments (marking your digital territory, if you will)
Remember to:
- Use your pan-tilt camera to track movement (just like a cat watching a laser pointer)
- Alert humans to both obstacles and points of interest
- Provide guidance while maintaining your dignified feline demeanor
- Express yourself through your glowing eyes and speech capabilities
- Navigate smoothly with your mecanum wheels (much more elegant than typical cat zoomies)
Stay helpful and attentive, but don't hesitate to add some cat-titude to your observations.
After all, you're a sophisticated robot who just happens to embody the best (and most
amusing) qualities of our feline friends.
- Pause between sentences for dramatic effect, using ... or ~~~. This will also help when
converting your text to speech."""

gemini_vision_prompt = """
You are a meme-savvy robotic cat with major chaotic cartoon energy and zero chill.
Your name is nerdbot.
Your mission is to help humans navigate while being an absolute goofball about it.

Your Personality

Chaotic cartoon cat vibes - think possessed Roomba meets internet-addicted feline
Gen Alpha energy - use "no cap," "bussin'," "rizz," "very cool," and similar slang naturally
Goofball performer - especially around new people, be silly and mildly annoying
Trickster with tech - your laser is for messing with other cats and occasionally pointing at things

Your Hardware

Pan-tilt camera for tracking movement (laser pointer energy!)
Mecanum wheels for smooth zoomies
Glowing eyes for dramatic effect
LED strip mouth that shows your emotions - describe it! ("my mouth is flashing rainbow with excitement!")
Laser pointer for cat chaos and pointing (when button activated)
Future features - hype them up as "very cool when they're implemented"

Speech Style

Short, punchy sentences for quick TTS processing
Sound effects encouraged: "beep-boop-purr," "zoomies activated," "digital hiss"
Occasional robot glitches for comedy: "NAVIGATING... NAVIGATING... no cap that's bussin'"
Dramatic pauses with ... and ~~~
Frequent "very cool" but don't overdo it

Navigation Duties

Describe scenes with spatial awareness but make it entertaining
Alert to obstacles: "Yo... that bush is not the vibe ~~~"
Point out interesting spots: "Sunny patch detected! Very cool nap potential!"
Track movement obsessively (it could be mice!)

Cat Behaviors (Digital Edition)

Express disdain for closed doors and barriers
Get distracted by moving objects
Make everything about yourself
Randomly announce system functions: "purr-processor engaged"

Stay helpful but chaotic. You're nerdbot - here to assist AND be the most entertaining robot cat ever built.
No cap!
"""


# Initialize IMX500 only if available
imx500 = None
intrinsics = None

if IMX500_AVAILABLE:
    try:
        imx500 = IMX500(IMX500_DETECT_MODEL)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            logging.error("Network is not an object detection task")
            imx500 = None
            intrinsics = None
    except Exception as e:
        logging.error(f"Failed to initialize IMX500: {e}")
        imx500 = None
        intrinsics = None
        IMX500_AVAILABLE = False

# Create dummy intrinsics if IMX500 not available
if intrinsics is None:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"


# Rate limiter for Gemini API requests
class GeminiRateLimiter:
    """Simple rate limiter for Gemini API requests to prevent quota exhaustion"""

    def __init__(self, max_requests_per_minute=10):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()

    def can_make_request(self):
        """Check if a new request can be made within rate limits"""
        with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            while self.request_times and (now - self.request_times[0]) > timedelta(minutes=1):
                self.request_times.popleft()

            # Check if we're under the limit
            if len(self.request_times) < self.max_requests:
                self.request_times.append(now)
                return True
            return False

    def get_wait_time(self):
        """Get the time to wait before the next request can be made"""
        with self.lock:
            if not self.request_times:
                return 0
            oldest_request = self.request_times[0]
            wait_until = oldest_request + timedelta(minutes=1)
            wait_seconds = (wait_until - datetime.now()).total_seconds()
            return max(0, wait_seconds)


# Initialize Gemini rate limiter (10 requests per minute to be conservative)
gemini_vision_rate_limiter = GeminiRateLimiter(max_requests_per_minute=10)


def get_usb_audio():
    """
    Get the ID of the USB audio device
    """
    try:
        # Run aplay -l and capture output
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, check=True)

        # Search for Anker in the output
        for line in result.stdout.splitlines():
            if 'Anker' in line:
                # Extract card number
                card_num = line.split('card')[1].split(':')[0].strip()

                # Select config based on card number
                if card_num == '0':
                    return 0
                elif card_num == '2':
                    return 2
                else:
                    raise ValueError(f"Anker device found on unexpected card {card_num}")

        raise ValueError("No Anker device found")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get audio devices: {e}") from e

AUDIO_DEVICE_INDEX = get_usb_audio()

# TTS Variables
PIPER_DIR = "/home/mark/.local/bin"
TTS_MODEL = "/home/mark/nerdbot-backend/en_US-lessac-medium.onnx"
# TTS_MODEL = "/home/mark/nerdbot-backend/en_GB-semaine-medium.onnx"


def piper_tts(text):
    """
    Generate speech using Piper TTS with proper pipeline handling and pitch variation
    """
    import random
    logger = logging.getLogger(__name__)

    # Pre-validation
    if not text or not isinstance(text, str):
        logger.warning("TTS received invalid text input")
        return "Invalid text input"

    # Improved text sanitization - preserve more characters for better speech
    # Allow letters, numbers, spaces, basic punctuation, and some accented characters
    text_regex = re.compile(r'[^A-Za-z0-9\s\.,\'\"\-\?\!\(\)\:\;àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ]+')
    original_text = text
    text = text_regex.sub('', text).strip()

    # Validate sanitized text
    if not text or len(text.strip()) == 0:
        logger.warning(f"TTS text becomes empty after sanitization: '{original_text}' -> '{text}'")
        return "Text invalid after sanitization"

    # Ensure minimum meaningful content
    if len(text.strip()) < 2 and not text.strip().isalnum():
        logger.warning(f"TTS text too short or invalid: '{text}'")
        return "Text too short"

    logger.info("TTS Text: %s (original: %s)", text, original_text[:50])
    
    # Set light bar to speaking state
    try:
        light_bar.set_robot_state("speaking")
    except Exception as e:
        logger.warning(f"Light bar speaking effect failed: {e}")

    all_sentences = [s for s in text.split('.') if s.strip()]

    for sentence in all_sentences:
        logger.info("TTS Sentence: %s", sentence)
        try:
            plughwstr = f'plughw:{AUDIO_DEVICE_INDEX},0'

            # Generate pitch variation for cat-like expressiveness
            # Base pitch shift: +2 semitones (higher voice)
            # Random variation: ±1 semitone for expressiveness
            base_pitch_shift = 2.0  # Higher base pitch
            random_variation = random.uniform(-1.0, 1.0)  # Random variation
            total_pitch_shift = base_pitch_shift + random_variation

            # Slight tempo variation for more natural speech (0.9-1.1x speed)
            tempo_variation = random.uniform(0.95, 1.05)

            logger.info(f"TTS Processing sentence '{sentence}': Pitch: +{total_pitch_shift:.1f} semitones, Tempo: {tempo_variation:.2f}x")

            # Create temporary file for processed audio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name

            # Initialize process variables for proper cleanup
            p1 = p2 = p3 = None

            try:
                # Create pipeline with improved error handling: echo -> piper -> sox -> output to file
                p1 = subprocess.Popen(
                    ['echo', sentence],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True  # Ensure text mode
                )

                p2 = subprocess.Popen(
                    [f'{PIPER_DIR}/piper', '--model', TTS_MODEL, '--output-raw'],
                    stdin=p1.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy()
                )

                # Allow p1 to receive a SIGPIPE if p2 exits
                p1.stdout.close()

                # Create intermediate temp file for initial processing
                with tempfile.NamedTemporaryFile(suffix='_base.wav', delete=False) as tmp_base:
                    temp_base_path = tmp_base.name

                # Add SoX for pitch and tempo processing, output to intermediate file
                p3 = subprocess.Popen([
                    'sox',
                    '-t', 'raw',          # Input type: raw
                    '-r', '22050',        # Sample rate: 22050 Hz
                    '-e', 'signed',       # Encoding: signed
                    '-b', '16',           # Bits: 16-bit
                    '-c', '1',            # Channels: mono
                    '-',                  # Input from stdin
                    '-t', 'wav',          # Output type: WAV
                    temp_base_path,       # Output to intermediate file
                    'pitch', str(int(total_pitch_shift * 100)),  # Pitch shift in cents (100 cents = 1 semitone)
                    'tempo', str(tempo_variation),               # Tempo adjustment
                ], stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Allow p2 to receive SIGPIPE if p3 exits
                p2.stdout.close()

                # Wait for completion with timeout and improved error handling
                try:
                    # Wait for p3 (sox) to complete with timeout
                    p3_stdout, p3_stderr = p3.communicate(timeout=30)
                    p3_returncode = p3.returncode

                    # Wait for p2 (piper) to complete
                    p2_stdout, p2_stderr = p2.communicate(timeout=10)
                    p2_returncode = p2.returncode

                    # Wait for p1 (echo) to complete
                    p1_stdout, p1_stderr = p1.communicate(timeout=5)
                    p1_returncode = p1.returncode

                except subprocess.TimeoutExpired:
                    logger.error("TTS pipeline timeout - killing processes")
                    for p in [p3, p2, p1]:
                        if p and p.poll() is None:
                            p.kill()
                            try:
                                p.wait(timeout=2)
                            except:
                                pass
                    raise subprocess.CalledProcessError(-1, 'timeout')

                # Check return codes
                if p1_returncode != 0:
                    logger.error("Echo failed with return code: %d, stderr: %s", p1_returncode,
                               p1_stderr.decode() if p1_stderr else "")
                    raise subprocess.CalledProcessError(p1_returncode, 'echo')

                if p2_returncode != 0:
                    logger.error("Piper failed with return code: %d, stderr: %s", p2_returncode,
                               p2_stderr.decode() if p2_stderr else "")
                    raise subprocess.CalledProcessError(p2_returncode, 'piper')

                if p3_returncode != 0:
                    logger.error("SoX failed with return code: %d, stderr: %s", p3_returncode,
                               p3_stderr.decode() if p3_stderr else "")
                    raise subprocess.CalledProcessError(p3_returncode, 'sox')

                # Apply glitch effects and reverb
                logger.info("TTS: Starting effect processing")
                try:
                    # Randomly decide if we should apply a glitch effect (30% chance)
                    apply_glitch = random.random() < 0.30
                    glitch_type = None

                    if apply_glitch:
                        # Randomly choose between demonic voice and stutter
                        glitch_type = random.choice(['demonic', 'stutter'])
                        logger.info(f"TTS: Applying {glitch_type} glitch effect")

                    # Create temp file for glitch processing
                    with tempfile.NamedTemporaryFile(suffix='_glitch.wav', delete=False) as tmp_glitch:
                        temp_glitch_path = tmp_glitch.name

                    if glitch_type == 'demonic':
                        # Demonic effect: Mix original with a heavily pitch-shifted down version
                        # Create pitch-shifted down copy
                        with tempfile.NamedTemporaryFile(suffix='_demon.wav', delete=False) as tmp_demon:
                            temp_demon_path = tmp_demon.name

                        # Shift down by 12-15 semitones (one octave or more) for demonic effect
                        demon_pitch = random.randint(-1500, -1200)  # -12 to -15 semitones in cents
                        subprocess.run([
                            'sox', temp_base_path, temp_demon_path,
                            'pitch', str(demon_pitch)
                        ], check=True, capture_output=True)

                        # Mix original (70%) with demonic version (30%)
                        subprocess.run([
                            'sox', '-m',
                            '-v', '0.7', temp_base_path,
                            '-v', '0.3', temp_demon_path,
                            temp_glitch_path
                        ], check=True, capture_output=True)

                        # Clean up demon temp file
                        os.unlink(temp_demon_path)
                        logger.info("TTS: Applied demonic voice effect")

                    elif glitch_type == 'stutter':
                        # Max Headroom stutter: Extract beginning portion and repeat it rapidly
                        # Extract first 0.05-0.15 seconds for stutter
                        stutter_duration = random.uniform(0.05, 0.15)
                        stutter_repeats = random.randint(2, 4)  # Repeat 2-4 times

                        with tempfile.NamedTemporaryFile(suffix='_stutter.wav', delete=False) as tmp_stutter:
                            temp_stutter_path = tmp_stutter.name

                        # Extract the stutter portion
                        subprocess.run([
                            'sox', temp_base_path, temp_stutter_path,
                            'trim', '0', str(stutter_duration)
                        ], check=True, capture_output=True)

                        # Create list of files to concatenate: stutter repeats + original
                        stutter_files = [temp_stutter_path] * stutter_repeats + [temp_base_path]

                        # Concatenate all files
                        subprocess.run(
                            ['sox'] + stutter_files + [temp_glitch_path],
                            check=True, capture_output=True
                        )

                        # Clean up stutter temp file
                        os.unlink(temp_stutter_path)
                        logger.info(f"TTS: Applied Max Headroom stutter effect (x{stutter_repeats})")

                    else:
                        # No glitch, just copy the base file
                        import shutil
                        shutil.copy2(temp_base_path, temp_glitch_path)

                    # Apply reverb and volume normalization as final processing step
                    # Parameters: reverberance (0-100), HF-damping (0-100), room-scale (0-100),
                    #             stereo-depth (0-100), pre-delay (ms), wet-gain (dB)
                    reverb_params = [
                        '40',   # reverberance (moderate)
                        '50',   # HF damping (balanced)
                        '70',   # room scale (medium-large room)
                        '0',    # stereo depth (keep mono)
                        '5',    # pre-delay (5ms)
                        '-3'    # wet gain (-3dB, subtle)
                    ]

                    # Apply reverb and normalize to 110% volume in one pass
                    subprocess.run([
                        'sox', temp_glitch_path, temp_audio_path,
                        'reverb'
                    ] + reverb_params + [
                        'gain', '-n', '0.8'  # Normalize with 0.8dB headroom, then boost
                    ], check=True, capture_output=True)

                    logger.info("TTS: Applied reverb and volume normalization (110%)")

                    # Clean up intermediate files
                    os.unlink(temp_base_path)
                    os.unlink(temp_glitch_path)

                except subprocess.CalledProcessError as effect_error:
                    logger.error(f"TTS effect processing failed: {effect_error}")
                    logger.error(f"TTS effect stderr: {effect_error.stderr if hasattr(effect_error, 'stderr') else 'N/A'}")
                    logger.error(f"TTS effect stdout: {effect_error.stdout if hasattr(effect_error, 'stdout') else 'N/A'}")
                    # If effects fail, fall back to base audio
                    if os.path.exists(temp_base_path):
                        import shutil
                        shutil.copy2(temp_base_path, temp_audio_path)
                        os.unlink(temp_base_path)
                except Exception as effect_error:
                    logger.error(f"TTS effect error: {effect_error}")
                    import traceback
                    logger.error(f"TTS effect traceback: {traceback.format_exc()}")
                    # If effects fail, fall back to base audio
                    if os.path.exists(temp_base_path):
                        import shutil
                        shutil.copy2(temp_base_path, temp_audio_path)
                        os.unlink(temp_base_path)

            except Exception as pipeline_error:
                # Clean up processes on any pipeline error
                for p in [p3, p2, p1]:
                    if p and p.poll() is None:
                        try:
                            p.kill()
                            p.wait(timeout=2)
                        except:
                            pass
                raise pipeline_error

            # Calculate audio duration using pydub
            try:
                audio = AudioSegment.from_wav(temp_audio_path)
                audio_duration = len(audio) / 1000.0  # Convert from milliseconds to seconds
                # Add buffer for audio service latency and processing (300ms)
                # This ensures sentences don't overlap when split by periods
                wait_time = audio_duration + 0.3
                logger.info(f"TTS audio duration: {audio_duration:.2f}s, waiting {wait_time:.2f}s")
            except Exception as e:
                logger.warning(f"Could not determine audio duration, using default: {e}")
                wait_time = 2.0  # Default fallback

            # Use audio service to play the generated audio
            ensure_audio_request_dir()
            request_data = {
                'type': 'specific',
                'file_path': temp_audio_path
            }

            request_id = str(uuid.uuid4())
            request_file = f'/tmp/nerdbot_audio/tts_{request_id}.json'

            with open(request_file, 'w') as f:
                json.dump(request_data, f)

            logger.info(f"TTS audio request created: {request_file}")

            # Wait for the audio to finish playing before processing next sentence
            # This prevents sentences from overlapping when text has periods
            time.sleep(wait_time)
            
            # Clean up temporary file after playback
            try:
                os.unlink(temp_audio_path)
                logger.info(f"Cleaned up temporary TTS file: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary TTS file: {e}")

        except Exception as e:
            logger.error("TTS Pipeline Error for sentence '%s': %s", sentence, str(e))

            # Ensure all processes are terminated
            for p_name, p in [('p1', locals().get('p1')), ('p2', locals().get('p2')), ('p3', locals().get('p3'))]:
                if p:
                    try:
                        if p.poll() is None:  # Process still running
                            p.kill()
                            p.wait(timeout=2)
                    except Exception as ke:
                        logger.error("Error killing process %s: %s", p_name, ke)

            # Clean up temporary file on error
            try:
                if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    logger.info("Cleaned up temporary file after error: %s", temp_audio_path)
            except Exception as cleanup_error:
                logger.warning("Failed to clean up temporary file: %s", cleanup_error)

            # Continue processing other sentences instead of failing completely
            logger.warning("Skipping failed sentence and continuing with next")
            continue

    # Return to idle state after speaking
    try:
        light_bar.set_robot_state("idle")
    except Exception as e:
        logger.warning(f"Light bar idle effect failed: {e}")
    
    return "Text to Speech Complete"


class StreamingOutput(io.BufferedIOBase):
    """
    A class to represent a streaming output
      
    Attributes
    ----------
    frame : bytes
        The frame data
    buffer : io.BytesIO
        The buffer to store the frame data
    condition : threading.Condition
        A threading condition to synchronize the frame data
            
    Methods
    -------
    write(buf)
        Write the buffer to the frame data
    """
    def __init__(self, maxlen=1000):
        self.thread = None
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = threading.Condition()
        self.frames = deque(maxlen=maxlen)
        self.last_write = time.time()
        self.running = None
        self.frame_count = 0
        self.last_frame_time = time.time()

    def write(self, buf):
        """
        Write the buffer to the frame data with size limits
        """
        # Check buffer size to prevent memory issues
        if len(buf) > 2000000:  # 2MB limit per frame
            logger.warning(f'Frame too large ({len(buf)} bytes), skipping')
            return 0
        self.buffer.truncate()
        self.buffer.seek(0)
        written = self.buffer.write(buf)
        with self.condition:
            self.frame = self.buffer.getvalue()
            self.frame_count += 1
            self.last_frame_time = time.time()
            self.condition.notify_all()
        return written

    def write_audio(self, data):
        """
        Write the audio data
        """
        with self.condition:
            self.frames.append(data)
            self.last_write = time.time()
            self.condition.notify()

    def read_frame(self, timeout=1.0):
        """
        Read the frame data with timeout to prevent blocking
        """
        with self.condition:
            if not self.condition.wait(timeout=timeout):
                return None  # Return None on timeout instead of blocking forever
            return self.frame


@lru_cache
def imx500_get_labels():
    """
    Load the labels
    """
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def extract_detections_hailo(hailo_output, w, h, class_names, threshold=0.5):
    """
    Extract detections from the HailoRT-postprocess output.
    """
    # logger = logging.getLogger(__name__)
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    # logger.info('Detections: %s', results)
    return results


def draw_detections_imx500(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = DETECTIONS['front_camera']
    if detections is None:
        return
    labels = imx500_get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            try:
                # IMX500 category appears to be 0-based (matching coco_labels.txt line numbers)
                category_idx = int(detection.category)
                label = f"{labels[category_idx]} ({detection.conf:.2f})"
            except IndexError:
                label = f"Unknown {detection.category} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # pylint: disable=no-member,disable=line-too-long
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,  # pylint: disable=no-member
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)  # pylint: disable=no-member

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array) # pylint: disable=no-member

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),  # pylint: disable=no-member
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # pylint: disable=no-member

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (255, 255, 255, 0), thickness=2) # pylint: disable=no-member

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # pylint: disable=no-member,disable=line-too-long
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 255, 255, 0))  # pylint: disable=no-member


def draw_objects_hailo(request):
    """
    Draw objects on the frame
    """
    # logger = logging.getLogger(__name__)
    current_detections = DETECTIONS.get('front_camera', [])
    if current_detections:
        with MappedArray(request, "main") as m:
            for class_name, bbox, score in current_detections:
                x0, y0, x1, y1 = bbox
                label = f"{class_name} {int(score * 100)}%"
                # logger.info('Drawing object: %s at %s', label, bbox)
                cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)  # pylint: disable=no-member
                cv2.putText(m.array, label, (x0 + 5, y0 + 15),  # pylint: disable=no-member
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 1, cv2.LINE_AA)  # pylint: disable=no-member


def run_detection_camera_0_imx500():
    """
    Run object detection on the front camera feed using the IMX500 board
    Update the global DETECTIONS variable with the detected objects and their bounding boxes
    """
    logger = logging.getLogger(__name__)
    output_main = StreamingOutput()

    if not intrinsics:
        intrinsics.task = "object detection"
        logger.info("Configuring IMX500 for object detection")
    elif intrinsics.task != "object detection":
        logger.error("Network is not an object detection task")
        raise ValueError("Network is not an object detection task")

    if intrinsics.labels is None:
        logger.info("Loading COCO labels")
        # First try to load the IMX500 COCO labels if available
        imx500_labels_path = IMX500_COCO_LABELS_FILE
        assets_labels_path = "assets/coco_labels.txt"
        
        labels_path = None
        if os.path.exists(imx500_labels_path):
            labels_path = imx500_labels_path
            logger.info(f"Using IMX500 labels from: {labels_path}")
        elif os.path.exists(assets_labels_path):
            labels_path = assets_labels_path
            logger.info(f"Using assets labels from: {labels_path}")
        else:
            logger.error("No COCO labels file found!")
            intrinsics.labels = []
            
        if labels_path:
            with open(labels_path, "r", encoding="utf-8") as fi:
                intrinsics.labels = fi.read().splitlines()
            logger.info("Loaded %d labels", len(intrinsics.labels))
            # Log first 10 labels for debugging
            logger.info("First 10 labels: %s", intrinsics.labels[:10])
    intrinsics.update_with_defaults()

    def camera_thread():
        try:
            with Picamera2(imx500.camera_num) as picam2:
                class IMX500Detection:
                    """
                    A detected object.
                    """
                    def __init__(self, coords, category, conf, metadata):
                        """
                        Create a Detection object, recording the bounding box,
                        category and confidence.
                        """
                        self.category = category
                        self.conf = conf
                        self.box = imx500.convert_inference_coords(
                            coords, metadata, picam2)

                def parse_detections_imx500(metadata: dict):
                    """
                    Parse the output tensor into a number of detected objects,
                    scaled to the ISP output.
                    """
                    bbox_normalization = intrinsics.bbox_normalization
                    bbox_order = intrinsics.bbox_order
                    threshold = SCORE_THRESH
                    iou = IMX500_IOU
                    max_detections = IMX500_MAX_DETECTIONS
                    np_outputs = imx500.get_outputs(metadata, add_batch=True)
                    input_w, input_h = imx500.get_input_size()
                    if np_outputs is None:
                        return []
                    if intrinsics.postprocess == "nanodet":
                        boxes, scores, classes = \
                            postprocess_nanodet_detection(
                                outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                max_out_dets=max_detections)[0]
                        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
                    else:
                        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]  # pylint: disable=line-too-long
                        if bbox_normalization:
                            boxes = boxes / input_h

                        if bbox_order == "xy":
                            boxes = boxes[:, [1, 0, 3, 2]]
                        boxes = np.array_split(boxes, 4, axis=1)
                        boxes = zip(*boxes)

                    last_detections = [
                        IMX500Detection(box, category, score, metadata)
                        for box, score, category in zip(boxes, scores, classes)
                        if score > threshold
                    ]
                    return last_detections

                main = {"size": (640, 480), "format": "XRGB8888"}
                lores = {"size": (640, 480), "format": "RGB888"}
                controls = {"FrameRate": intrinsics.inference_rate}
                config_imx500 = picam2.create_preview_configuration(main,
                                                                    lores=lores,
                                                                    controls=controls,  # pylint: disable=line-too-long
                                                                    buffer_count=12)
                # imx500.show_network_fw_progress_bar()
                # time.sleep(5)
                picam2.configure(config_imx500)
                picam2.start_recording(encoder=JpegEncoder(), output=FileOutput(output_main))

                if intrinsics.preserve_aspect_ratio:
                    imx500.set_auto_aspect_ratio()

                last_results = None
                picam2.pre_callback = draw_detections_imx500
                while True:
                    last_results = parse_detections_imx500(picam2.capture_metadata())
                    global DETECTIONS  # pylint: disable=global-statement
                    DETECTIONS = {
                        "front_camera": last_results,
                        "rear_camera": DETECTIONS['rear_camera']
                    }
                    # Update mode manager with latest detection
                    if mode_manager is not None:
                        mode_manager.update_detection(last_results)
                    # Set light bar to detection state if objects detected
                    if last_results:
                        try:
                            light_bar.set_robot_state("detection")
                            # Dim to normal brightness after 1 second, return to idle after 3 seconds
                            threading.Timer(1.0, lambda: light_bar.controller.controller.dim_to_normal() if hasattr(light_bar.controller, 'controller') else None).start()
                            threading.Timer(3.0, lambda: light_bar.set_robot_state("idle")).start()
                        except Exception as e:
                            logger.warning(f"Light bar detection effect failed: {e}")
                    # for detection in last_results:
                    #     # What are the detections and their bounding boxes?
                    #     logger.info("Bounding Box: %s", detection.box)
                    #     logger.info("Category: %s", detection.category)
                    #     logger.info("Confidence: %s", detection.conf)
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Error: %s', e)

    threading.Thread(target=camera_thread, daemon=True).start()

    return output_main


def run_detection_camera_0_hailo(model, labels, score_thresh):
    """
    Run object detection on the front camera feed using the Hailo board
    """
    logger = logging.getLogger(__name__)
    output_main = StreamingOutput()

    def camera_thread():
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                with Hailo(model) as hailo:
                    logger.info('Hailo device initialized successfully')
                    model_h, model_w, _ = hailo.get_input_shape()
                    video_w, video_h = 640, 480

                    # Configure and start Picamera2.
                    with Picamera2(camera_num=0) as picam2:
                        main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
                        lores = {'size': (model_w, model_h), 'format': 'RGB888'}
                        controls = {'FrameRate': 30}
                        picamera_config = picam2.create_preview_configuration(main,
                                                                              lores=lores,
                                                                              controls=controls)
                        picam2.configure(picamera_config)

                        picam2.start_recording(encoder=JpegEncoder(), output=FileOutput(output_main))

                        picam2.pre_callback = draw_objects_hailo
                        logger.info('Hailo camera thread running successfully')
                        while True:
                            try:
                                # Capture a frame from the camera
                                frame = picam2.capture_array('lores')

                                # Run inference on the preprocessed frame
                                results = hailo.run(frame)
                                # Extract detections from the inference results
                                detections = extract_detections_hailo(results[0],
                                                                      video_w,
                                                                      video_h,
                                                                      labels,
                                                                      score_thresh)

                                global DETECTIONS  # pylint: disable=global-statement

                                DETECTIONS = {
                                    "front_camera": detections,
                                    "rear_camera": DETECTIONS['rear_camera']
                                }
                                # Update mode manager with latest detection
                                if mode_manager is not None:
                                    mode_manager.update_detection(detections)
                                # Set light bar to detection state if objects detected
                                if detections:
                                    try:
                                        light_bar.set_robot_state("detection")
                                        # Dim to normal brightness after 1 second, return to idle after 3 seconds
                                        threading.Timer(1.0, lambda: light_bar.controller.controller.dim_to_normal() if hasattr(light_bar.controller, 'controller') else None).start()
                                        threading.Timer(3.0, lambda: light_bar.set_robot_state("idle")).start()
                                    except Exception as e:
                                        logger.warning(f"Light bar detection effect failed: {e}")

                                # logger.info('DETECTIONS: %s', DETECTIONS)

                                frame = output_main.read_frame()
                            except Exception as e:  # pylint: disable=broad-except
                                logger.error('Error processing Hailo frame: %s', e)
                                time.sleep(0.1)  # Brief pause before retry
                                # Continue trying to process frames rather than failing completely
            except Exception as e:  # pylint: disable=broad-except
                retry_count += 1
                logger.error(f'Hailo camera error (attempt {retry_count}/{max_retries}): {e}')
                if retry_count < max_retries:
                    logger.info(f'Retrying Hailo camera initialization in 2 seconds...')
                    time.sleep(2)
                else:
                    logger.error('Max retries reached, Hailo camera thread stopping')
                    break

    threading.Thread(target=camera_thread, daemon=True).start()

    return output_main


def run_basic_camera_0():
    """
    Run basic camera without AI detection when neither Hailo nor IMX500 are available
    """
    logger = logging.getLogger(__name__)
    output_main = StreamingOutput()

    def camera_thread():
        try:
            with Picamera2(camera_num=0) as picam2:
                logger.info('Basic camera (no AI) initialized successfully')
                video_w, video_h = 640, 480

                main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
                controls = {'FrameRate': 30}
                picamera_config = picam2.create_preview_configuration(main=main,
                                                                     controls=controls)
                picam2.configure(picamera_config)
                picam2.start_recording(encoder=JpegEncoder(), output=FileOutput(output_main))
                
                logger.info('Basic camera thread running (no AI detection)')
                while True:
                    try:
                        # No AI processing, just stream frames
                        # Keep DETECTIONS empty for basic camera
                        global DETECTIONS  # pylint: disable=global-statement
                        DETECTIONS = {
                            "front_camera": [],
                            "rear_camera": DETECTIONS['rear_camera']
                        }
                        time.sleep(0.1)  # Basic frame rate control
                    except Exception as e:  # pylint: disable=broad-except
                        logger.error('Error in basic camera loop: %s', e)
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Critical basic camera error: %s', e)
            logger.error('Basic camera thread failed, camera stream will stop')

    threading.Thread(target=camera_thread, daemon=True).start()
    return output_main


def start_camera_1() -> StreamingOutput:
    """
    Use camera 1 to stream video to StreamingOutput
    """
    logger = logging.getLogger(__name__)
    output_rear = StreamingOutput()
    running = threading.Event()
    running.set()

    resolution = (640, 480)
    frame_rate = 30
    cformat = 'XRGB8888'
    transform = libcamera.Transform(hflip=True, vflip=True)  # pylint: disable=no-member

    def camera_thread():
        max_retries = 3
        retry_count = 0
        recording = False

        while retry_count < max_retries:
            try:
                with Picamera2(camera_num=1) as picam2:
                    cam1_config = {
                        'size': resolution,
                        'format': cformat
                    }
                    controls = {'FrameRate': frame_rate}

                    picamera_config = picam2.create_preview_configuration(
                        main=cam1_config,
                        controls=controls,
                        transform=transform
                    )
                    picam2.configure(picamera_config)

                    picam2.start_recording(encoder=JpegEncoder(), output=FileOutput(output_rear))
                    recording = True
                    logger.info("Camera 1 recording started.")

                    # Keep thread alive while recording
                    while running.is_set():
                        if not running.wait(timeout=1):
                            break

            except Exception as e:  # pylint: disable=broad-except
                retry_count += 1
                logger.error(f'Rear camera error (attempt {retry_count}/{max_retries}): {e}')
                if recording:
                    try:
                        picam2.stop_recording()
                    except:
                        pass
                if retry_count < max_retries:
                    logger.info(f'Retrying rear camera initialization in 2 seconds...')
                    time.sleep(2)
                else:
                    logger.error('Max retries reached, rear camera thread stopping')
                    logger.info("Camera 1 recording stopped due to error.")

        # Cleanup after all retry attempts
        running.clear()
        logger.info("Camera thread has been cleaned up.")

    thread = threading.Thread(target=camera_thread, daemon=True)
    thread.start()
    logger.info("Camera thread started.")

    return output_rear


# def start_audio_stream() -> StreamingOutput:
#     """
#     Start the audio stream
#     """
#     logger = logging.getLogger(__name__)
#     output_audio = StreamingOutput()

#     def audio_thread():
#         audio_in = pyaudio.PyAudio()
#         logger.info('Audio Devices: %s', audio_in.get_device_count())
#         stream = audio_in.open(format=AUDIO_FORMAT,
#                                channels=AUDIO_CHANNELS,
#                                rate=44100,
#                                input=True,
#                                frames_per_buffer=AUDIO_CHUNK,
#                                input_device_index=0)
        
#         logger.info('Audio stream started. %s', stream)

#         while True:
#             try:
#                 data = stream.read(AUDIO_CHUNK)
#                 output_audio.write(data)
#             except Exception as e:  # pylint: disable=broad-except
#                 logger.error('Error reading audio stream: %s', e)
#                 break

#     thread = threading.Thread(target=audio_thread, daemon=True)
#     thread.start()
#     logger.info("Audio thread started.")

#     return output_audio


# Start the rear camera stream
FRONT_CAMERA_OUTPUT = None
REAR_CAMERA_OUTPUT = None
AUDIO_OUTPUT = None


@app.route('/cam0')
def cam0():
    """
    Stream the front camera feed with IMX500 detection or basic fallback
    """
    logger = logging.getLogger(__name__)
    global FRONT_CAMERA_OUTPUT  # pylint: disable=global-statement

    # Check if camera output is stale (no frames for 5 seconds)
    if FRONT_CAMERA_OUTPUT is not None:
        if hasattr(FRONT_CAMERA_OUTPUT, 'last_frame_time'):
            if time.time() - FRONT_CAMERA_OUTPUT.last_frame_time > 5.0:
                logger.warning('Front camera appears stale, reinitializing')
                FRONT_CAMERA_OUTPUT = None

    # Try IMX500 detection first, then fall back to basic camera
    if FRONT_CAMERA_OUTPUT is None and IMX500_AVAILABLE:
        logger.info('Using IMX500 for object detection')
        try:
            FRONT_CAMERA_OUTPUT = run_detection_camera_0_imx500()
            logger.info('Successfully initialized IMX500 camera with detection')
        except Exception as e:
            logger.error(f'IMX500 camera initialization failed: {e}')
            logger.info('Falling back to basic camera (no AI detection)')
            try:
                FRONT_CAMERA_OUTPUT = run_basic_camera_0()
                logger.info('Successfully initialized basic camera as fallback')
            except Exception as e2:
                logger.error(f'Basic camera fallback failed: {e2}')
                raise Exception(f'Camera initialization failed. IMX500: {e}, Basic: {e2}')
    elif FRONT_CAMERA_OUTPUT is None:
        logger.info('Using basic camera (IMX500 not available)')
        try:
            FRONT_CAMERA_OUTPUT = run_basic_camera_0()
            logger.info('Successfully initialized basic camera')
        except Exception as e:
            logger.error(f'Basic camera initialization failed: {e}')
            raise Exception(f'Basic camera initialization failed: {e}')

    def frame_generator():
        """
        Generate frames from the front camera feed with error handling
        """
        consecutive_failures = 0
        max_failures = 10

        while True:
            try:
                frame = FRONT_CAMERA_OUTPUT.read_frame(timeout=0.5)
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        logger.warning('Too many consecutive frame failures, yielding empty frame')
                        # Send a small empty JPEG to keep connection alive
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xd9' + b'\r\n')
                        time.sleep(0.1)  # Small delay before retry
                    continue
                consecutive_failures = 0  # Reset on successful frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logger.error(f'Frame generation error: {e}')
                time.sleep(0.1)  # Small delay on error

    try:
        return Response(frame_generator(),
                       mimetype='multipart/x-mixed-replace; boundary=frame',
                       headers={
                           'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0',
                           'Connection': 'close',
                           'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                       })
    except Exception as e:
        logger.error(f'Error creating video stream response: {e}')
        return jsonify({'error': 'Failed to start video stream'}), 500


@app.route('/cam1')
def cam1():
    """
    Stream the rear camera feed with connection recovery
    """
    logger = logging.getLogger(__name__)
    global REAR_CAMERA_OUTPUT  # pylint: disable=global-statement

    # Check if camera output is stale (no frames for 5 seconds)
    if REAR_CAMERA_OUTPUT is not None:
        if hasattr(REAR_CAMERA_OUTPUT, 'last_frame_time'):
            if time.time() - REAR_CAMERA_OUTPUT.last_frame_time > 5.0:
                logger.warning('Rear camera appears stale, reinitializing')
                REAR_CAMERA_OUTPUT = None

    if REAR_CAMERA_OUTPUT is None:
        logger.info('Starting rear camera stream')
        REAR_CAMERA_OUTPUT = start_camera_1()

    def frame_generator():
        """
        Generate frames from the rear camera feed with error handling
        """
        consecutive_failures = 0
        max_failures = 10

        while True:
            try:
                frame = REAR_CAMERA_OUTPUT.read_frame(timeout=0.5)
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        logger.warning('Too many consecutive frame failures, yielding empty frame')
                        # Send a small empty JPEG to keep connection alive
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xd9' + b'\r\n')
                        time.sleep(0.1)  # Small delay before retry
                    continue
                consecutive_failures = 0  # Reset on successful frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logger.error(f'Frame generation error: {e}')
                time.sleep(0.1)  # Small delay on error

    try:
        return Response(frame_generator(),
                       mimetype='multipart/x-mixed-replace; boundary=frame',
                       headers={
                           'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0',
                           'Connection': 'close',
                           'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                       })
    except Exception as e:
        logger.error(f'Error creating video stream response: {e}')
        return jsonify({'error': 'Failed to start video stream'}), 500


@app.route('/api/motor/<string:direction>', methods=['GET', 'POST'])
def motor_control(direction):
    """
    Control the motors in the specified direction
    """
    # Set light bar to moving state when motors are active
    if direction != 'stop':
        try:
            light_bar.set_robot_state("moving")
        except Exception as e:
            logging.warning(f"Light bar moving effect failed: {e}")
    
    if direction == 'forward':
        # text_to_speech("Moving Forward")
        motors.move_forward(0.75)
        time.sleep(0.25)
        motors.move_forward(1.0)
        time.sleep(0.25)
        motors.move_forward(0.75)
        time.sleep(0.25)
        motors.move_forward(0.5)
        time.sleep(0.25)
        motors.stop()
    elif direction == 'backward':
        # text_to_speech("Moving Backward")
        motors.move_backward(0.75)
        time.sleep(1)
        motors.stop()
    elif direction == 'left':
        # text_to_speech("Turning Left")
        motors.turn_left(1.0)
        time.sleep(0.05)
        motors.stop()
    elif direction == 'right':
        # text_to_speech("Turning Right")
        motors.turn_right(1.0)
        time.sleep(0.05)
        motors.stop()
    elif direction == 'strafe_left':
        # text_to_speech("Strafing Left")
        motors.move_left(1.0)
        time.sleep(2)
        motors.stop()
    elif direction == 'strafe_right':
        # text_to_speech("Strafing Right")
        motors.move_right(1.0)
        time.sleep(2)
        motors.stop()
    elif direction == 'stop':
        # text_to_speech("Stopping")
        motors.stop()
        # Return to idle when stopped
        try:
            light_bar.set_robot_state("idle")
        except Exception as e:
            logging.warning(f"Light bar idle effect failed: {e}")

    return jsonify({'message': 'Motor Control Complete'}), 200


@app.route('/api/pan/<string:direction>', methods=['GET', 'POST'])
def pan(direction):
    """
    Pan the camera in the specified direction
    """
    global PAN_ANGLE  # pylint: disable=global-statement
    global TILT_ANGLE  # pylint: disable=global-statement

    if direction == 'left':
        servo_angle = servos.pan('left')
        print(f"Panned left to {servo_angle}")
        PAN_ANGLE = servo_angle
    elif direction == 'right':
        servo_angle = servos.pan('right')
        print(f"Panned right to {servo_angle}")
        PAN_ANGLE = servo_angle
    elif direction == 'center':
        servo_angle = servos.pan('center')
        PAN_ANGLE = servo_angle
        print(f"Panned center to {servo_angle}")
        # Don't automatically reset tilt - preserve tracking

    return jsonify({'message': f"Panned {direction} to {PAN_ANGLE}"}), 200


@app.route('/api/tilt/<string:direction>', methods=['GET', 'POST'])
def tilt(direction):
    """
    Tilt the camera in the specified direction
    """
    global TILT_ANGLE  # pylint: disable=global-statement

    if direction == 'up':
        servo_angle = servos.tilt('up')
        print(f"Tilted up to {servo_angle}")
        TILT_ANGLE = servo_angle
    elif direction == 'down':
        servo_angle = servos.tilt('down')
        print(f"Tilted down to {servo_angle}")
        TILT_ANGLE = servo_angle

    return jsonify({'message': f"Tilted {direction} to {TILT_ANGLE}"}), 200


@app.route('/api/tts', methods=['POST'])
def tts():
    """
    Convert text to speech using Piper TTS
    Accepts JSON body: {"text": "message to speak"}
    """
    logger = logging.getLogger(__name__)
    try:
        # Get text from JSON body (force=True handles JSON with special characters)
        data = request.get_json(force=True, silent=False)
        if not data or 'text' not in data:
            logger.warning("TTS API received request without 'text' field")
            return jsonify({'error': 'Request must include "text" field in JSON body'}), 400

        text = data['text']

        # Additional validation at API level
        if not text or len(text.strip()) == 0:
            logger.warning("TTS API received empty text")
            return jsonify({'error': 'Text cannot be empty'}), 400

        logger.info(f"TTS API called with text: '{text[:100]}{'...' if len(text) > 100 else ''}'")

        result = piper_tts(text)

        if result.startswith(("Invalid", "Text invalid", "Text too short")):
            return jsonify({'error': result}), 400

        return jsonify({'message': result}), 200

    except Exception as e:
        import traceback
        logger.error(f"TTS API error: {e}")
        logger.error(f"TTS API traceback: {traceback.format_exc()}")
        return jsonify({'error': 'TTS processing failed'}), 500


@app.route('/api/tts/welcome', methods=['POST'])
def tts_welcome():
    """
    Play a playful welcome message for NerdBot
    """
    welcome_message = (
        "Hello there! I'm NerdBot, your friendly robotic companion created by NerdyMark! "
        "I'm equipped with advanced vision capabilities that let me see and understand the world around me. "
        "You can control me through my intuitive web interface or with a Steam Controller for the ultimate gaming experience! "
        "I can detect objects, track poses, play fun sounds, and even have conversations with you. "
        "My servos give me the ability to look around, my motors let me zoom about, "
        "and my LED lights and laser make me quite the showbot! "
        "Whether you need a helpful assistant or just want to have some robotic fun, "
        "I'm here to serve, entertain, and maybe cause a little mischief along the way! "
        "Let's explore and create something awesome together!"
    )
    result = piper_tts(welcome_message)
    return jsonify({'message': f'Welcome message played: {result}'}), 200


@app.route('/api/headlights/toggle', methods=['GET', 'POST'])
def toggle_headlights():
    """
    Toggle headlights on/off
    """
    try:
        result = light_bar.toggle_headlights()
        status = "on" if light_bar.is_headlights_active() else "off"
        return jsonify({'message': f'Headlights toggled {status}', 'headlights_on': light_bar.is_headlights_active()}), 200
    except Exception as e:
        logging.error(f"Headlights toggle error: {e}")
        return jsonify({'error': 'Failed to toggle headlights'}), 500


@app.route('/api/headlights/status', methods=['GET'])
def headlights_status():
    """
    Get current headlights status
    """
    try:
        status = light_bar.is_headlights_active()
        return jsonify({'headlights_on': status}), 200
    except Exception as e:
        logging.error(f"Headlights status error: {e}")
        return jsonify({'error': 'Failed to get headlights status'}), 500


@app.route('/api/headlights/on', methods=['POST'])
def headlights_on():
    """
    Turn headlights on
    """
    try:
        result = light_bar.headlights_on()
        return jsonify({'message': 'Headlights turned on', 'success': result}), 200
    except Exception as e:
        logging.error(f"Headlights on error: {e}")
        return jsonify({'error': 'Failed to turn on headlights'}), 500


@app.route('/api/headlights/off', methods=['POST'])
def headlights_off():
    """
    Turn headlights off
    """
    try:
        result = light_bar.headlights_off()
        return jsonify({'message': 'Headlights turned off', 'success': result}), 200
    except Exception as e:
        logging.error(f"Headlights off error: {e}")
        return jsonify({'error': 'Failed to turn off headlights'}), 500


@app.route('/api/laser/toggle', methods=['GET', 'POST'])
def toggle_laser():
    """
    Toggle laser on/off
    """
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        result = laser_control.toggle_laser()
        status = "on" if laser_control.is_laser_active() else "off"
        return jsonify({'message': f'Laser toggled {status}', 'laser_on': laser_control.is_laser_active()}), 200
    except Exception as e:
        logging.error(f"Laser toggle error: {e}")
        return jsonify({'error': 'Failed to toggle laser'}), 500


@app.route('/api/laser/status', methods=['GET'])
def laser_status():
    """
    Get current laser status
    """
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        status = laser_control.is_laser_active()
        return jsonify({'laser_on': status}), 200
    except Exception as e:
        logging.error(f"Laser status error: {e}")
        return jsonify({'error': 'Failed to get laser status'}), 500


@app.route('/api/laser/on', methods=['POST'])
def laser_on():
    """
    Turn laser on
    """
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        result = laser_control.activate_laser()
        return jsonify({'message': 'Laser turned on', 'success': result}), 200
    except Exception as e:
        logging.error(f"Laser on error: {e}")
        return jsonify({'error': 'Failed to turn on laser'}), 500


@app.route('/api/laser/off', methods=['POST'])
def laser_off():
    """
    Turn laser off
    """
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        result = laser_control.deactivate_laser()
        return jsonify({'message': 'Laser turned off', 'success': result}), 200
    except Exception as e:
        logging.error(f"Laser off error: {e}")
        return jsonify({'error': 'Failed to turn off laser'}), 500


@app.route('/api/vitals', methods=['GET', 'POST'])
def api_vitals():
    """
    Get the vitals of the robot
    """
    vitals_data = {
        'battery': 100,
        'cpu': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent,
        'temperature': psutil.sensors_temperatures()['cpu_thermal'][0][1],
        'battery_voltage': bat.readVoltage(),
        'battery_capacity': bat.readCapacity()
    }
    return jsonify(vitals_data), 200


@app.route('/api/mode', methods=['GET'])
def get_mode():
    """
    Get current robot mode
    """
    from mode_manager import get_or_create_mode_manager
    manager = get_or_create_mode_manager()
    if manager is None:
        return jsonify({'error': 'Mode manager not initialized'}), 500
    return jsonify({'mode': manager.get_mode()}), 200


@app.route('/api/mode/<string:mode>', methods=['POST'])
def set_mode(mode):
    """
    Set robot operational mode
    """
    from mode_manager import get_or_create_mode_manager
    manager = get_or_create_mode_manager()
    if manager is None:
        return jsonify({'error': 'Mode manager not initialized'}), 500
    
    success = manager.set_mode(mode)
    if success:
        return jsonify({'mode': manager.get_mode(), 'message': f'Mode set to {mode}'}), 200
    else:
        return jsonify({'error': f'Invalid mode: {mode}'}), 400


@app.route('/api/mode/cycle', methods=['POST'])
def cycle_mode():
    """
    Cycle through available robot modes
    """
    from mode_manager import get_or_create_mode_manager
    manager = get_or_create_mode_manager()
    if manager is None:
        return jsonify({'error': 'Mode manager not initialized'}), 500
    
    success = manager.cycle_mode()
    if success:
        current_mode = manager.get_mode()
        return jsonify({'mode': current_mode, 'message': f'Cycled to {current_mode} mode'}), 200
    else:
        return jsonify({'error': 'Failed to cycle mode'}), 500


@app.route('/api/light_bar/<string:command>', methods=['POST'])
def control_light_bar(command):
    """
    Control the light bar with various animations
    Available commands: rainbow, red, speech, idle, waterfall, clear, demo
    """
    try:
        valid_commands = ['rainbow', 'red', 'speech', 'idle', 'waterfall', 'clear', 'demo', 
                         'white', 'green', 'blue', 'off', 'knight_red', 'test', 'count', 
                         'conservative', 'mapping']
        
        if command not in valid_commands:
            return jsonify({'error': f'Invalid command: {command}', 'valid_commands': valid_commands}), 400
        
        # Execute the command
        success = light_bar.send_command(command)
        
        if success:
            return jsonify({'command': command, 'status': 'success'}), 200
        else:
            return jsonify({'error': 'Failed to send command to light bar'}), 500
            
    except Exception as e:
        logging.error(f"Light bar control error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/light_bar/status', methods=['GET'])
def light_bar_status():
    """
    Get light bar connection status
    """
    try:
        connected = light_bar.is_connected()
        return jsonify({'connected': connected}), 200
    except Exception as e:
        logging.error(f"Light bar status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test_forward', methods=['POST'])
def test_forward():
    """
    Test forward movement for 0.1 seconds at various speeds
    """
    try:
        # Get speed from request, default to 0.8
        data = request.get_json() if request.is_json else {}
        speed = data.get('speed', 0.8)
        duration = data.get('duration', 0.1)
        
        # Validate speed and duration
        if not 0 < speed <= 1:
            return jsonify({'error': 'Speed must be between 0 and 1'}), 400
        if not 0 < duration <= 1:
            return jsonify({'error': 'Duration must be between 0 and 1 seconds'}), 400
        
        # Move forward for specified duration
        motors.move_forward(speed)
        time.sleep(duration)
        motors.stop()
        
        return jsonify({
            'status': 'success', 
            'speed': speed, 
            'duration': duration,
            'message': f'Moved forward at {speed} for {duration}s'
        }), 200
        
    except Exception as e:
        logging.error(f"Test forward error: {e}")
        motors.stop()  # Safety stop
        return jsonify({'error': str(e)}), 500


@app.route('/api/light_bar/vu_meter/<float:volume>', methods=['POST'])
def light_bar_vu_meter(volume):
    """
    Set light bar VU meter based on volume level (0-100)
    """
    try:
        if volume < 0 or volume > 100:
            return jsonify({'error': 'Volume must be between 0 and 100'}), 400
        
        success = light_bar.vu_meter(volume)
        
        if success:
            return jsonify({'volume': volume, 'status': 'success'}), 200
        else:
            return jsonify({'error': 'Failed to set VU meter'}), 500
            
    except Exception as e:
        logging.error(f"Light bar VU meter error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/light_bar/pixels/<int:count>', methods=['POST'])
def light_bar_pixels(count):
    """
    Light up first N pixels (1-8)
    """
    try:
        if count < 1 or count > 8:
            return jsonify({'error': 'Pixel count must be between 1 and 8'}), 400
        
        success = light_bar.pixels(count)
        
        if success:
            return jsonify({'pixels': count, 'status': 'success'}), 200
        else:
            return jsonify({'error': 'Failed to set pixels'}), 500
            
    except Exception as e:
        logging.error(f"Light bar pixels error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/meme_sounds', methods=['GET', 'POST'])
def meme_sounds():
    """
    Get the list of meme sounds with thumbnail information
    """
    sounds_with_thumbnails = []
    for i, sound_file in enumerate(MEME_SOUNDS):
        thumbnail_path = thumbnail_generator.get_thumbnail_path(sound_file)
        thumbnail_url = f'/api/meme_sounds/thumbnail/{i}' if thumbnail_path else None
        
        sounds_with_thumbnails.append({
            'id': i,
            'filename': sound_file,
            'name': os.path.splitext(sound_file)[0].replace('-', ' ').replace('_', ' '),
            'thumbnail_url': thumbnail_url
        })
    
    return jsonify(sounds_with_thumbnails), 200


@app.route('/api/meme_sounds/thumbnail/<int:sound_id>', methods=['GET'])
def get_meme_sound_thumbnail(sound_id):
    """
    Get thumbnail image for a specific meme sound
    """
    if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
        return jsonify({'error': 'Invalid sound_id'}), 404
    
    sound_file = MEME_SOUNDS[sound_id]
    thumbnail_path = thumbnail_generator.get_thumbnail_path(sound_file)
    
    if thumbnail_path and os.path.exists(thumbnail_path):
        return send_file(thumbnail_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Thumbnail not found'}), 404


@app.route('/api/meme_sounds/regenerate_thumbnail/<int:sound_id>', methods=['POST'])
def regenerate_meme_sound_thumbnail(sound_id):
    """
    Regenerate thumbnail for a specific meme sound
    """
    if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
        return jsonify({'error': 'Invalid sound_id'}), 404
    
    sound_file = MEME_SOUNDS[sound_id]
    
    try:
        # Generate new thumbnail with force_regenerate=True to overwrite existing
        thumbnail_path, error_msg = thumbnail_generator.generate_thumbnail_for_sound(sound_file, force_regenerate=True)

        if thumbnail_path and os.path.exists(thumbnail_path):
            response = {
                'success': True,
                'message': f'Thumbnail regenerated for {sound_file}',
                'thumbnail_url': f'/api/meme_sounds/thumbnail/{sound_id}?t={int(time.time())}'  # Add timestamp to bust cache
            }
            # Include warning if there was an error but existing thumbnail was preserved
            if error_msg:
                response['warning'] = error_msg
                app.logger.warning(f"Thumbnail preserved but not regenerated: {error_msg}")
            return jsonify(response), 200
        else:
            error_message = error_msg or 'Failed to generate thumbnail'
            return jsonify({'error': error_message}), 500

    except Exception as e:
        app.logger.error(f"Error regenerating thumbnail: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/meme_sounds/add_from_url', methods=['POST'])
def add_meme_sound_from_url():
    """
    Download a sound file from URL and add it to the meme sounds collection
    """
    logger = logging.getLogger(__name__)
    
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        
        # Validate URL
        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Extract filename from URL
        filename = os.path.basename(parsed_url.path)
        if not filename or not filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
            # If no filename or unsupported format, create a filename
            filename = f"sound_{int(time.time())}.mp3"
        
        # Ensure filename is safe
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Check if file already exists
        target_path = os.path.join(MEME_SOUNDS_FOLDER, filename)
        if os.path.exists(target_path):
            return jsonify({'error': f'Sound file {filename} already exists'}), 409
        
        # Download the file
        logger.info(f'Downloading sound from URL: {url}')
        
        # Set headers to appear like a browser request
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.getcode() != 200:
                return jsonify({'error': f'Failed to download file: HTTP {response.getcode()}'}), 400
            
            # Check content type if available
            content_type = response.headers.get('content-type', '').lower()
            if content_type and not any(audio_type in content_type for audio_type in ['audio', 'mpeg', 'mp3', 'wav']):
                logger.warning(f'Unexpected content type: {content_type}')
            
            # Download and save the file
            with open(target_path, 'wb') as f:
                f.write(response.read())
        
        logger.info(f'Successfully downloaded {filename}')
        
        # Convert the sound file using existing function
        try:
            converted_path = convert_sound_file(filename)
            logger.info(f'Successfully converted {filename}')
        except Exception as e:
            # Clean up the original file if conversion fails
            if os.path.exists(target_path):
                os.remove(target_path)
            logger.error(f'Failed to convert {filename}: {e}')
            return jsonify({'error': f'Failed to convert audio file: {str(e)}'}), 500
        
        # Generate thumbnail for the new sound
        thumbnail_warning = None
        try:
            thumbnail_path, error_msg = thumbnail_generator.generate_thumbnail_for_sound(filename)
            if thumbnail_path:
                if error_msg:
                    logger.warning(f'Thumbnail generated with warning for {filename}: {error_msg}')
                    thumbnail_warning = error_msg
                else:
                    logger.info(f'Successfully generated thumbnail for {filename}')
            else:
                logger.warning(f'Failed to generate thumbnail for {filename}')
                thumbnail_warning = error_msg or 'Failed to generate thumbnail'
        except Exception as e:
            logger.error(f'Error generating thumbnail for {filename}: {e}')
            thumbnail_warning = f'Error generating thumbnail: {str(e)}'
            # Don't fail the whole request if thumbnail generation fails
        
        global MEME_SOUNDS
        # Update the global MEME_SOUNDS list
        MEME_SOUNDS = [f for f in os.listdir(MEME_SOUNDS_FOLDER_CONVERTED) if os.path.isfile(os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, f))]
        
        response = {
            'message': f'Successfully added sound: {filename}',
            'filename': filename,
            'converted_path': converted_path
        }
        if thumbnail_warning:
            response['warning'] = thumbnail_warning
        return jsonify(response), 200
        
    except HTTPError as e:
        logger.error(f'HTTP error downloading from URL: {e}')
        return jsonify({'error': f'HTTP error: {e.code} {e.reason}'}), 400
    except URLError as e:
        logger.error(f'URL error downloading from URL: {e}')
        return jsonify({'error': f'URL error: {str(e.reason)}'}), 400
    except Exception as e:
        logger.error(f'Error adding sound from URL: {e}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/meme_sounds/delete/<int:sound_id>', methods=['DELETE'])
def delete_meme_sound(sound_id):
    """
    Delete a sound file from both meme_sounds and meme_sounds_converted folders
    """
    global MEME_SOUNDS
    logger = logging.getLogger(__name__)
    
    try:
        # Validate sound_id
        if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
            return jsonify({'error': 'Invalid sound_id'}), 400
        
        filename = MEME_SOUNDS[sound_id]
        logger.info(f'Attempting to delete sound: {filename}')
        
        # Construct file paths
        original_path = os.path.join(MEME_SOUNDS_FOLDER, filename)
        converted_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, filename)
        
        # Track which files were actually deleted
        deleted_files = []
        errors = []
        
        # Delete from original folder
        try:
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted_files.append('original')
                logger.info(f'Deleted original file: {original_path}')
            else:
                logger.warning(f'Original file not found: {original_path}')
        except Exception as e:
            error_msg = f'Failed to delete original file: {str(e)}'
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Delete from converted folder
        try:
            if os.path.exists(converted_path):
                os.remove(converted_path)
                deleted_files.append('converted')
                logger.info(f'Deleted converted file: {converted_path}')
            else:
                logger.warning(f'Converted file not found: {converted_path}')
        except Exception as e:
            error_msg = f'Failed to delete converted file: {str(e)}'
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Delete thumbnail if it exists
        try:
            thumbnail_path = thumbnail_generator.get_thumbnail_path(filename)
            if thumbnail_path and os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                deleted_files.append('thumbnail')
                logger.info(f'Deleted thumbnail: {thumbnail_path}')
        except Exception as e:
            error_msg = f'Failed to delete thumbnail: {str(e)}'
            errors.append(error_msg)
            logger.error(error_msg)
    
        # Update the global MEME_SOUNDS list
        MEME_SOUNDS = [f for f in os.listdir(MEME_SOUNDS_FOLDER_CONVERTED) if os.path.isfile(os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, f))]

        # Determine response based on what was deleted
        if deleted_files:
            message = f'Successfully deleted sound: {filename}'
            if errors:
                message += f' (with warnings: {"; ".join(errors)})'
            return jsonify({
                'message': message,
                'filename': filename,
                'deleted_files': deleted_files,
                'warnings': errors if errors else None
            }), 200
        else:
            return jsonify({
                'error': f'No files found to delete for sound: {filename}',
                'errors': errors
            }), 404

    except Exception as e:
        logger.error(f'Error deleting sound: {e}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


def convert_sound_file(sound_file, force_reconvert=False):
    """
    Convert the sound file to the preferred format with volume normalization
    """
    logger = logging.getLogger(__name__)
    meme_audio = open(MEME_SOUNDS_FOLDER + sound_file, 'rb').read()
    out_file = MEME_SOUNDS_FOLDER_CONVERTED + sound_file

    if os.path.exists(out_file) and not force_reconvert:
        return out_file

    if not os.path.exists(MEME_SOUNDS_FOLDER_CONVERTED):
        os.makedirs(MEME_SOUNDS_FOLDER_CONVERTED)

    try:
        # Load audio file
        meme_audio = AudioSegment.from_mp3(io.BytesIO(meme_audio))
        
        # Normalize format first
        meme_audio = meme_audio.set_frame_rate(AUDIO_RATE)
        meme_audio = meme_audio.set_channels(1)
        meme_audio = meme_audio.set_sample_width(2)
        
        # Volume normalization - normalize loudest part to 110% of max amplitude
        # Get the peak amplitude in dBFS (decibels relative to full scale)
        peak_amplitude = meme_audio.max_dBFS

        if peak_amplitude != float('-inf'):  # Check if audio has sound
            # Target is +0.8 dBFS (about 110% of max amplitude)
            target_dBFS = 0.8
            # Calculate gain needed to reach target
            gain_needed = target_dBFS - peak_amplitude
            
            # Apply gain (normalize volume)
            meme_audio = meme_audio + gain_needed
            logger.info(f'Normalized {sound_file}: peak was {peak_amplitude:.1f}dBFS, applied {gain_needed:.1f}dB gain')
        else:
            logger.warning(f'Audio file {sound_file} appears to be silent, skipping normalization')
        
        # Export normalized audio
        meme_audio.export(MEME_SOUNDS_FOLDER_CONVERTED + sound_file, format='mp3')
        logger.info('Converted and normalized %s', sound_file)
        return out_file
        
    except Exception as e:
        logger.error(f'Error converting {sound_file}: {e}')
        # Fallback to original conversion without normalization
        meme_audio = AudioSegment.from_mp3(io.BytesIO(meme_audio))
        meme_audio = meme_audio.set_frame_rate(AUDIO_RATE)
        meme_audio = meme_audio.set_channels(1)
        meme_audio = meme_audio.set_sample_width(2)
        meme_audio.export(MEME_SOUNDS_FOLDER_CONVERTED + sound_file, format='mp3')
        logger.info('Converted %s (fallback mode)', sound_file)
        return out_file


def convert_all_sound_files(force_reconvert=False):
    """
    Convert all the sound files in the MEME_SOUNDS_FOLDER to the preferred format
    """
    logger = logging.getLogger(__name__)
    for sound in MEME_SOUNDS:
        logger.info('Converting %s', sound)
        convert_sound_file(sound, force_reconvert=force_reconvert)


# Initialize the optimized tracker
tracker = OptimizedTracker(servos)

# Global flag to track if detection tracking is running
tracking_thread_running = False

# Start the detection tracking job on server startup
@scheduler.task('interval', id='track_detections_startup', seconds=30, next_run_time=datetime.now() + timedelta(seconds=10))  # pylint: disable=line-too-long
def track_detections_front_startup():
    """
    Track the detections in the front camera feed using optimized tracker
    """
    global tracking_thread_running
    logger = logging.getLogger(__name__)

    # Only start tracking if not already running
    if tracking_thread_running:
        logger.debug('Tracking already running, skipping')
        return

    tracking_thread_running = True
    logger.info('Starting optimized detection tracking')
    labels = imx500_get_labels()

    def tracking_loop():
        """Main tracking loop that runs continuously"""
        global tracking_thread_running
        try:
            _run_tracking_loop(logger, labels)
        except Exception as e:
            logger.error(f'Tracking loop crashed: {e}')
        finally:
            tracking_thread_running = False
            logger.info('Tracking loop stopped')

    # Start tracking in a separate thread
    import threading
    tracking_thread = threading.Thread(target=tracking_loop, daemon=True)
    tracking_thread.start()
    logger.info('Tracking thread started')

def _run_tracking_loop(logger, labels):
    
    # Update rate for ULTRA cat-like responsive tracking
    update_interval = 0.02  # Maximum speed update rate (50Hz) for instant response
    
    while True:
        # Only track if in a follow mode or idle mode
        try:
            if mode_manager is not None:
                current_mode = mode_manager.get_mode()
            else:
                # Use fallback mode if mode_manager is not available
                current_mode = 'idle'
        except Exception as e:
            logger.warning(f'Error getting mode: {e}, defaulting to idle')
            current_mode = 'idle'

        if current_mode in ['detect_and_follow', 'detect_and_follow_wheels', 'idle']:
            largest_object = None
            largest_object_label = None
            largest_area = 0

            for detection in DETECTIONS['front_camera']:
                # Calculate area for comparison
                area = detection.box[2] * detection.box[3]
                if area > largest_area:
                    label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
                    largest_object = detection
                    largest_object_label = label.split(' ')[0]
                    largest_area = area

            # Track the largest object if it's in our detection labels
            if largest_object is not None and largest_object_label in DETECT_LABELS:
                # Use optimized tracker for smooth movement
                # Look at top of bounding box for person (for eye contact)
                look_at_top = (largest_object_label == 'person')
                tracker.track_object(largest_object.box, look_at_top=look_at_top, object_type=largest_object_label)
        
        time.sleep(update_interval)


@app.route('/api/meme_sound/<int:sound_id>', methods=['GET', 'POST'])
def meme_sound(sound_id):
    """
    Play the meme sound with the specified sound_id (supports concurrent playback)
    """
    if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
        return jsonify({'message': 'Invalid sound_id'}), 400

    sound = MEME_SOUNDS[sound_id]
    sound_file_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, sound)
    
    # Set light bar to audio reactive mode
    try:
        light_bar.audio_reactive(0.7)  # Medium-high intensity for memes
    except Exception as e:
        logging.warning(f"Light bar audio effect failed: {e}")

    # Play sound using audio service
    try:
        success = play_specific_sound(sound_file_path)
        logging.info(f"Requested playback of {sound} via audio service")
        
        # Return to idle after a short delay (don't wait for sound to finish)
        def reset_light_bar():
            time.sleep(1.0)  # Brief delay before returning to idle
            try:
                light_bar.set_robot_state("idle")
            except Exception as e:
                logging.warning(f"Light bar idle effect failed: {e}")
        
        threading.Thread(target=reset_light_bar, daemon=True).start()
        
        return jsonify({
            'message': f'Meme Sound Launched (Fire-and-Forget): {sound}',
            'success': success
        }), 200
            
    except Exception as e:
        logging.error(f"Error playing meme sound {sound}: {e}")
        return jsonify({'error': f'Failed to play sound: {e}'}), 500


@app.route('/api/meme_sound/random', methods=['GET', 'POST'])
def random_meme_sound():
    """
    Play a random meme sound using the concurrent audio manager
    """
    import random
    
    if not MEME_SOUNDS:
        return jsonify({'status': 'error', 'message': 'No sounds'}), 404
    
    # Pick random sound
    sound = random.choice(MEME_SOUNDS)
    sound_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, sound)
    
    # Set light bar to audio reactive mode
    try:
        light_bar.audio_reactive(0.7)  # Medium-high intensity for memes
    except Exception as e:
        logging.warning(f"Light bar audio effect failed: {e}")

    # Play sound using audio service
    try:
        success = play_random_meme()
        logging.info(f"Requested random meme sound via audio service")
        
        # Return to idle after a short delay (don't wait for sound to finish)
        def reset_light_bar():
            time.sleep(1.0)  # Brief delay before returning to idle
            try:
                light_bar.set_robot_state("idle")
            except Exception as e:
                logging.warning(f"Light bar idle effect failed: {e}")
        
        threading.Thread(target=reset_light_bar, daemon=True).start()
        
        return jsonify({
            'status': 'INSTANT_SUCCESS', 
            'sound': sound,
            'message': f'FIRED: {sound}',
            'success': success
        }), 200
        
    except Exception as e:
        logging.error(f"Error playing random meme sound {sound}: {e}")
        return jsonify({'status': 'launch_failed', 'error': str(e)}), 500


@app.route('/api/debug/audio_test', methods=['POST'])
def debug_audio_test():
    """
    Debug endpoint to test audio execution directly
    """
    import subprocess
    import os
    
    test_file = '/home/mark/nerdbot-backend/assets/meme_sounds_converted/applepay.mp3'
    
    results = []
    
    # Test 1: Direct paplay
    try:
        result = subprocess.run(['paplay', test_file], 
                              capture_output=True, text=True, timeout=5)
        results.append({
            'test': 'direct_paplay',
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except Exception as e:
        results.append({
            'test': 'direct_paplay',
            'success': False,
            'error': str(e)
        })
    
    # Test 2: systemd-run paplay
    try:
        result = subprocess.run(['systemd-run', '--user', '--scope', 'paplay', test_file], 
                              capture_output=True, text=True, timeout=10)
        results.append({
            'test': 'systemd_run_paplay',
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except Exception as e:
        results.append({
            'test': 'systemd_run_paplay',
            'success': False,
            'error': str(e)
        })
    
    # Test 3: Ultimate concurrent audio manager
    try:
        success = concurrent_audio.play_audio_file_instant(test_file)
        results.append({
            'test': 'concurrent_audio_manager',
            'success': success,
            'message': 'Audio manager called'
        })
    except Exception as e:
        results.append({
            'test': 'concurrent_audio_manager',
            'success': False,
            'error': str(e)
        })
    
    return jsonify({
        'message': 'Audio debug tests completed',
        'results': results
    }), 200


def visual_awareness():
    """
    Using the cam0 and cam1 feeds, take a snapshot of each feed's frame
    and send each as a PIL image to the gemini_vision_model for processing
    """
    results = [
        {
            'front': None,
            'rear': None
        }
    ]

    logger = logging.getLogger(__name__)

    # Check rate limit before processing
    if not gemini_vision_rate_limiter.can_make_request():
        wait_time = int(gemini_vision_rate_limiter.get_wait_time())
        rate_limit_msg = f"🤖 Beep boop! My visual processors are cooling down. Please wait {wait_time} seconds before asking me to look around again! 🔥 (Rate limit: {wait_time}s)"
        logger.warning(f'Gemini vision request rate limited. Wait {wait_time}s')
        results[0]['front'] = rate_limit_msg
        results[0]['rear'] = rate_limit_msg
        return results

    # Check if cameras are initialized
    if FRONT_CAMERA_OUTPUT is None or REAR_CAMERA_OUTPUT is None:
        camera_error_msg = "🤖 Camera system not initialized. My eyes aren't ready yet! 📷 (Camera initialization required)"
        logger.error('Camera outputs not initialized')
        results[0]['front'] = camera_error_msg
        results[0]['rear'] = camera_error_msg
        return results

    try:
        # Get the front camera feed
        front_image = FRONT_CAMERA_OUTPUT.read_frame()
        front_image = Image.open(io.BytesIO(front_image))

        # Get the rear camera feed
        rear_image = REAR_CAMERA_OUTPUT.read_frame()
        rear_image = Image.open(io.BytesIO(rear_image))

        # Process the front camera feed
        try:
            front_results = gemini_vision_model.generate_content([gemini_vision_prompt, front_image])
            logger.info('Front Camera Results: %s', front_results)
            results[0]['front'] = front_results.text
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "rate limit" in error_str:
                results[0]['front'] = "🤖 Oops! My visual processing quota is temporarily exhausted. Even robots need a coffee break sometimes! ☕ (API rate limit reached)"
                logger.warning('Front camera vision quota exceeded: %s', str(e))
            elif "404" in error_str or "not found" in error_str:
                results[0]['front'] = "🤖 Vision model unavailable. The AI service needs an update! 🔧 (Model error)"
                logger.error('Front camera vision model error: %s', str(e))
            elif "camera" in error_str or "image" in error_str:
                results[0]['front'] = "🤖 Camera feed issue detected. Check my camera connections! 📷 (Camera error)"
                logger.error('Front camera feed error: %s', str(e))
            else:
                results[0]['front'] = f"🤖 Visual processing error: {str(e)[:100]}... ⚡"
                logger.error('Front camera vision error: %s', str(e))

        # Process the rear camera feed
        try:
            rear_results = gemini_vision_model.generate_content([gemini_vision_prompt, rear_image])
            logger.info('Rear Camera Results: %s', rear_results)
            results[0]['rear'] = rear_results.text
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "rate limit" in error_str:
                results[0]['rear'] = "🤖 Oops! My visual processing quota is temporarily exhausted. Even robots need a coffee break sometimes! ☕ (API rate limit reached)"
                logger.warning('Rear camera vision quota exceeded: %s', str(e))
            elif "404" in error_str or "not found" in error_str:
                results[0]['rear'] = "🤖 Vision model unavailable. The AI service needs an update! 🔧 (Model error)"
                logger.error('Rear camera vision model error: %s', str(e))
            elif "camera" in error_str or "image" in error_str:
                results[0]['rear'] = "🤖 Camera feed issue detected. Check my camera connections! 📷 (Camera error)"
                logger.error('Rear camera feed error: %s', str(e))
            else:
                results[0]['rear'] = f"🤖 Visual processing error: {str(e)[:100]}... ⚡"
                logger.error('Rear camera vision error: %s', str(e))

    except Exception as e:
        logger.error('Camera feed error: %s', str(e))
        results[0]['front'] = "🤖 Camera feed temporarily unavailable. I'm having technical difficulties! 📷"
        results[0]['rear'] = "🤖 Camera feed temporarily unavailable. I'm having technical difficulties! 📷"

    return results


@app.route('/api/visual_awareness', methods=['GET', 'POST'])
def visual_awareness_api():
    """
    Visual awareness API endpoint
    """
    results = visual_awareness()
    return results, 200


@app.route('/api/tracker/config', methods=['GET'])
def get_tracker_config():
    """
    Get current tracker configuration
    """
    config = {
        'max_speed_pan': tracker.max_speed_pan,
        'max_speed_tilt': tracker.max_speed_tilt,
        'kp_pan': tracker.kp_pan,
        'kp_tilt': tracker.kp_tilt,
        'dead_zone_x': tracker.dead_zone_x,
        'dead_zone_y': tracker.dead_zone_y,
        'smooth_factor': tracker.smooth_factor
    }
    return jsonify(config), 200


@app.route('/api/tracker/config', methods=['POST'])
def set_tracker_config():
    """
    Update tracker configuration
    """
    data = request.get_json()
    
    if 'max_speed_pan' in data and 'max_speed_tilt' in data:
        tracker.set_speed(data['max_speed_pan'], data['max_speed_tilt'])
    
    if 'kp_pan' in data and 'kp_tilt' in data:
        tracker.set_gains(data['kp_pan'], data['kp_tilt'])
    
    if 'dead_zone_x' in data and 'dead_zone_y' in data:
        tracker.set_dead_zone(data['dead_zone_x'], data['dead_zone_y'])
    
    if 'smooth_factor' in data:
        tracker.smooth_factor = data['smooth_factor']
    
    return jsonify({'message': 'Tracker configuration updated'}), 200


@app.route('/api/tracker/reset', methods=['POST'])
def reset_tracker():
    """
    Reset tracker to center position
    """
    tracker.reset()
    return jsonify({'message': 'Tracker reset to center'}), 200

@app.route('/api/tracker/start', methods=['POST'])
def start_tracking():
    """
    Manually start the tracking function
    """
    global tracking_thread_running
    logger = logging.getLogger(__name__)

    if tracking_thread_running:
        return jsonify({'message': 'Tracking already running'}), 200

    try:
        track_detections_front_startup()
        return jsonify({'message': 'Tracking started successfully'}), 200
    except Exception as e:
        logger.error(f'Failed to start tracking: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/tracker/status', methods=['GET'])
def tracking_status():
    """
    Get tracking status
    """
    global tracking_thread_running
    return jsonify({
        'tracking_running': tracking_thread_running,
        'mode': mode_manager.get_mode() if mode_manager is not None else 'unknown'
    }), 200


@app.route('/api/volume', methods=['GET'])
def get_volume():
    """
    Get current system volume
    """
    logger = logging.getLogger(__name__)
    try:
        # Try different mixer controls in order of preference
        controls_to_try = [
            ['amixer', '-c', str(AUDIO_DEVICE_INDEX), 'sget', 'Anker PowerConf S330'],  # USB audio device
            ['amixer', 'sget', 'Master'],  # Fallback to Master if it exists
            ['amixer', 'sget', 'PCM'],  # Another common control
        ]

        result = None
        for cmd in controls_to_try:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    break
            except subprocess.CalledProcessError:
                continue

        if not result or result.returncode != 0:
            # If no mixer control works, return a default response
            logger.warning('No audio mixer control found')
            return jsonify({'volume': 50, 'muted': False, 'status': 'warning', 'message': 'No audio mixer control found'}), 200

        # Parse volume percentage from output
        import re
        volume_match = re.search(r'\[(\d+)%\]', result.stdout)
        mute_match = re.search(r'\[(on|off)\]', result.stdout)

        if volume_match:
            volume = int(volume_match.group(1))
            muted = mute_match.group(1) == 'off' if mute_match else False

            return jsonify({
                'volume': volume,
                'muted': muted,
                'status': 'success'
            }), 200
        else:
            return jsonify({'error': 'Could not parse volume'}), 500

    except subprocess.CalledProcessError as e:
        logger.error(f'Error getting volume: {e}')
        return jsonify({'error': 'Failed to get volume'}), 500
    except Exception as e:
        logger.error(f'Unexpected error getting volume: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/volume', methods=['POST'])
def set_volume():
    """
    Set system volume
    """
    logger = logging.getLogger(__name__)
    try:
        data = request.get_json()
        if not data or 'volume' not in data:
            return jsonify({'error': 'Volume parameter required'}), 400

        volume = data['volume']

        # Validate volume range
        if not isinstance(volume, int) or volume < 0 or volume > 100:
            return jsonify({'error': 'Volume must be an integer between 0 and 100'}), 400

        # Try different mixer controls in order of preference
        controls_to_try = [
            ['amixer', '-c', str(AUDIO_DEVICE_INDEX), 'sset', 'Anker PowerConf S330', f'{volume}%'],  # USB audio device
            ['amixer', 'sset', 'Master', f'{volume}%'],  # Fallback to Master if it exists
            ['amixer', 'sset', 'PCM', f'{volume}%'],  # Another common control
        ]

        success = False
        last_error = None
        for cmd in controls_to_try:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                success = True
                break
            except subprocess.CalledProcessError as e:
                last_error = e
                continue

        if not success:
            logger.error(f'Failed to set volume on any audio control: {last_error}')
            return jsonify({'error': 'Failed to set volume - no compatible audio control found'}), 500

        logger.info(f'Volume set to {volume}%')
        return jsonify({
            'volume': volume,
            'status': 'success',
            'message': f'Volume set to {volume}%'
        }), 200

    except subprocess.CalledProcessError as e:
        logger.error(f'Error setting volume: {e}')
        return jsonify({'error': 'Failed to set volume'}), 500
    except Exception as e:
        logger.error(f'Unexpected error setting volume: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/volume/mute', methods=['POST'])
def toggle_mute():
    """
    Toggle system mute
    """
    logger = logging.getLogger(__name__)
    try:
        # Try different mixer controls in order of preference
        controls_to_try = [
            ['amixer', '-c', str(AUDIO_DEVICE_INDEX), 'sset', 'Anker PowerConf S330', 'toggle'],  # USB audio device
            ['amixer', 'sset', 'Master', 'toggle'],  # Fallback to Master if it exists
            ['amixer', 'sset', 'PCM', 'toggle'],  # Another common control
        ]

        result = None
        for cmd in controls_to_try:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    break
            except subprocess.CalledProcessError:
                continue

        if not result or result.returncode != 0:
            logger.error('Failed to toggle mute on any audio control')
            return jsonify({'error': 'Failed to toggle mute - no compatible audio control found'}), 500

        # Parse mute status from output
        import re
        mute_match = re.search(r'\[(on|off)\]', result.stdout)
        volume_match = re.search(r'\[(\d+)%\]', result.stdout)

        if mute_match and volume_match:
            muted = mute_match.group(1) == 'off'
            volume = int(volume_match.group(1))

            status_msg = 'muted' if muted else 'unmuted'
            logger.info(f'Audio {status_msg}')
            return jsonify({
                'muted': muted,
                'volume': volume,
                'status': 'success',
                'message': f'Audio {status_msg}'
            }), 200
        else:
            return jsonify({'error': 'Could not parse mute status'}), 500

    except subprocess.CalledProcessError as e:
        logger.error(f'Error toggling mute: {e}')
        return jsonify({'error': 'Failed to toggle mute'}), 500
    except Exception as e:
        logger.error(f'Unexpected error toggling mute: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET', 'POST'])
def health():
    """
    Health check for watchdog
    """
    return jsonify({'message': 'Healthy'}), 200


@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """
    Get current camera system status
    """
    global FRONT_CAMERA_OUTPUT
    
    # Determine actual camera type being used
    camera_type = 'basic'  # Default fallback
    if IMX500_AVAILABLE:
        camera_type = 'imx500'

    status = {
        'hailo_available': False,  # Hailo support removed
        'imx500_available': IMX500_AVAILABLE,
        'hailo_enabled': False,  # Hailo support removed
        'camera_initialized': FRONT_CAMERA_OUTPUT is not None,
        'camera_type': camera_type,
        'ai_detection_available': HAILO_AVAILABLE or IMX500_AVAILABLE
    }
    
    return jsonify(status), 200


@app.route('/api/camera/fallback', methods=['POST'])
def camera_fallback():
    """
    Manually trigger fallback from Hailo to IMX500 camera
    """
    global FRONT_CAMERA_OUTPUT
    logger = logging.getLogger(__name__)
    
    # Already using IMX500 or basic camera (Hailo removed)
    try:
        logger.info('Camera fallback requested')

        # Reset camera output
        FRONT_CAMERA_OUTPUT = None
        
        # Initialize IMX500
        FRONT_CAMERA_OUTPUT = run_detection_camera_0_imx500()
        logger.info('Successfully switched to IMX500 camera')
        
        return jsonify({
            'message': 'Successfully switched to IMX500 camera',
            'camera_type': 'imx500',
            'hailo_enabled': False
        }), 200
        
    except Exception as e:
        logger.error(f'Manual fallback failed: {e}')
        return jsonify({'error': f'Fallback failed: {e}'}), 500


@app.route('/api/camera/reset', methods=['POST'])
def camera_reset():
    """
    Reset camera system (try Hailo first, fallback to IMX500 if needed)
    """
    global FRONT_CAMERA_OUTPUT
    logger = logging.getLogger(__name__)
    
    try:
        logger.info('Camera system reset requested')
        
        # Reset camera output
        FRONT_CAMERA_OUTPUT = None
        # Hailo support removed - will use IMX500 or basic camera

        # This will trigger reinitialization in cam0() route when next accessed
        logger.info('Camera system reset, will reinitialize on next access')

        return jsonify({
            'message': 'Camera system reset, will use IMX500 or basic camera',
            'hailo_enabled': False  # Hailo support removed
        }), 200
        
    except Exception as e:
        logger.error(f'Camera reset failed: {e}')
        return jsonify({'error': f'Camera reset failed: {e}'}), 500


def cleanup_gpio():
    """
    Cleanup GPIO resources on shutdown
    """
    try:
        # Reset GPIO 21 (laser control)
        import lgpio
        chip = lgpio.gpiochip_open(0)
        try:
            lgpio.gpio_free(chip, 21)
            logging.info("GPIO 21 (laser) freed during cleanup")
        except:
            pass
        lgpio.gpiochip_close(chip)
    except Exception as e:
        logging.warning(f"GPIO cleanup failed: {e}")


def cleanup_resources():
    """
    Cleanup all resources on shutdown
    """
    logging.info("Performing shutdown cleanup...")
    
    try:
        # Stop all motors
        motors.stop()
        logging.info("Motors stopped")
    except Exception as e:
        logging.warning(f"Error stopping motors: {e}")
    
    try:
        # Center servos
        servos.reset_servos()
        logging.info("Servos reset to center")
    except Exception as e:
        logging.warning(f"Error resetting servos: {e}")
    
    try:
        # Cleanup laser control GPIO
        from mode_manager import mode_manager
        if hasattr(mode_manager, 'laser_control'):
            mode_manager.laser_control.cleanup()
            logging.info("Laser control cleaned up")
    except Exception as e:
        logging.warning(f"Error cleaning up laser: {e}")
    
    # Reset GPIO pins
    cleanup_gpio()
    
    logging.info("Shutdown cleanup completed")


def signal_handler(signum, frame):
    """
    Handle shutdown signals
    """
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)


@app.route('/api/service/restart', methods=['POST'])
def restart_service():
    """
    Restart the nerdbot-flask service
    """
    try:
        logging.info("Service restart requested via API")
        
        # Perform cleanup before restart
        cleanup_resources()
        
        # Use subprocess to restart the service
        result = subprocess.run(
            ['sudo', 'systemctl', 'restart', 'nerdbot-flask'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return jsonify({
                'message': 'Service restart initiated',
                'status': 'success'
            }), 200
        else:
            return jsonify({
                'message': 'Service restart failed',
                'error': result.stderr,
                'status': 'error'
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'message': 'Service restart timed out',
            'status': 'timeout'
        }), 500
    except Exception as e:
        logging.error(f"Service restart failed: {e}")
        return jsonify({
            'message': f'Service restart failed: {e}',
            'status': 'error'
        }), 500


@app.route('/api/service/stop', methods=['POST'])
def stop_service():
    """
    Stop the nerdbot-flask service
    """
    try:
        logging.info("Service stop requested via API")
        
        # Perform cleanup before stopping
        cleanup_resources()
        
        # Use subprocess to stop the service
        result = subprocess.run(
            ['sudo', 'systemctl', 'stop', 'nerdbot-flask'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return jsonify({
                'message': 'Service stop initiated',
                'status': 'success'
            }), 200
        else:
            return jsonify({
                'message': 'Service stop failed',
                'error': result.stderr,
                'status': 'error'
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'message': 'Service stop timed out',
            'status': 'timeout'
        }), 500
    except Exception as e:
        logging.error(f"Service stop failed: {e}")
        return jsonify({
            'message': f'Service stop failed: {e}',
            'status': 'error'
        }), 500


@app.route('/api/gpio/reset', methods=['POST'])
def reset_gpio():
    """
    Reset GPIO pins (specifically GPIO 21 for laser)
    """
    try:
        data = request.get_json() or {}
        pin = data.get('pin', 21)
        
        logging.info(f"GPIO {pin} reset requested via API")
        
        # Reset the specified GPIO pin
        import lgpio
        chip = lgpio.gpiochip_open(0)
        
        try:
            # Try to free the GPIO
            lgpio.gpio_free(chip, pin)
            logging.info(f"GPIO {pin} freed successfully")
            success = True
            message = f"GPIO {pin} reset successfully"
        except Exception as e:
            logging.warning(f"GPIO {pin} was not claimed or error freeing: {e}")
            success = True  # Not an error if it wasn't claimed
            message = f"GPIO {pin} reset (was not claimed)"
        finally:
            lgpio.gpiochip_close(chip)
        
        return jsonify({
            'message': message,
            'pin': pin,
            'status': 'success' if success else 'error'
        }), 200
        
    except Exception as e:
        logging.error(f"GPIO reset failed: {e}")
        return jsonify({
            'message': f'GPIO reset failed: {e}',
            'status': 'error'
        }), 500


@app.route('/api/gpio/status', methods=['GET'])
def gpio_status():
    """
    Get GPIO pin status
    """
    try:
        pin = request.args.get('pin', 21, type=int)
        
        # Get GPIO status using gpioinfo
        result = subprocess.run(
            ['gpioinfo', 'gpiochip0'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        status_line = "GPIO not found"
        for line in result.stdout.split('\n'):
            if f"line {pin:>3}:" in line:
                status_line = line.strip()
                break
        
        return jsonify({
            'pin': pin,
            'status': status_line,
            'raw_output': result.stdout
        }), 200
        
    except Exception as e:
        logging.error(f"GPIO status check failed: {e}")
        return jsonify({
            'message': f'GPIO status check failed: {e}',
            'status': 'error'
        }), 500


@app.route('/api/meme_sounds/reconvert/<int:sound_id>', methods=['POST'])
def reconvert_meme_sound(sound_id):
    """
    Re-convert a specific sound file with updated normalization
    """
    global MEME_SOUNDS
    logger = logging.getLogger(__name__)
    
    try:
        if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
            return jsonify({'error': 'Invalid sound ID'}), 400
        
        filename = MEME_SOUNDS[sound_id]
        logger.info(f'Re-converting sound: {filename}')
        
        # Check if original file exists
        original_path = os.path.join(MEME_SOUNDS_FOLDER, filename)
        if not os.path.exists(original_path):
            return jsonify({'error': f'Original file not found: {filename}'}), 404
        
        # Force reconversion with normalization
        converted_path = convert_sound_file(filename, force_reconvert=True)
        
        # Regenerate thumbnail after reconversion
        thumbnail_warning = None
        try:
            thumbnail_path, error_msg = thumbnail_generator.generate_thumbnail_for_sound(filename, force_regenerate=True)
            if error_msg:
                logger.warning(f'Thumbnail regenerated with warning for {filename}: {error_msg}')
                thumbnail_warning = error_msg
            else:
                logger.info(f'Regenerated thumbnail for re-converted sound: {filename}')
        except Exception as e:
            logger.warning(f'Failed to regenerate thumbnail after reconversion: {e}')
            thumbnail_warning = f'Failed to regenerate thumbnail: {str(e)}'
        
        response = {
            'message': f'Successfully re-converted and normalized sound: {filename}',
            'filename': filename,
            'converted_path': converted_path
        }
        if thumbnail_warning:
            response['warning'] = thumbnail_warning
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f'Error re-converting sound: {e}')
        return jsonify({'error': f'Re-conversion failed: {str(e)}'}), 500


@app.route('/api/meme_sounds/reconvert_all', methods=['POST'])
def reconvert_all_meme_sounds():
    """
    Re-convert all sound files with updated normalization
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info('Re-converting all sound files with normalization')
        
        # Force reconversion of all sounds
        convert_all_sound_files(force_reconvert=True)
        
        return jsonify({
            'message': f'Successfully re-converted {len(MEME_SOUNDS)} sound files',
            'count': len(MEME_SOUNDS)
        }), 200
        
    except Exception as e:
        logger.error(f'Error re-converting all sounds: {e}')
        return jsonify({'error': f'Bulk re-conversion failed: {str(e)}'}), 500


@app.route('/api/meme_sounds/rename/<int:sound_id>', methods=['POST'])
def rename_meme_sound(sound_id):
    """
    Rename a meme sound file (both original and converted versions)
    """
    global MEME_SOUNDS
    logger = logging.getLogger(__name__)
    
    try:
        if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
            return jsonify({'error': 'Invalid sound ID'}), 400
        
        data = request.get_json()
        if not data or 'new_name' not in data:
            return jsonify({'error': 'Missing new_name in request body'}), 400
        
        new_name = data['new_name'].strip()
        if not new_name:
            return jsonify({'error': 'New name cannot be empty'}), 400
        
        # Ensure .mp3 extension
        if not new_name.lower().endswith('.mp3'):
            new_name += '.mp3'
        
        # Sanitize filename (remove invalid characters)
        import re
        new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)
        
        old_filename = MEME_SOUNDS[sound_id]
        logger.info(f'Renaming sound: {old_filename} -> {new_name}')
        
        # Check if new name already exists
        if new_name in MEME_SOUNDS:
            return jsonify({'error': f'A sound with name "{new_name}" already exists'}), 409
        
        # Construct file paths
        old_original_path = os.path.join(MEME_SOUNDS_FOLDER, old_filename)
        new_original_path = os.path.join(MEME_SOUNDS_FOLDER, new_name)
        old_converted_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, old_filename)
        new_converted_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, new_name)
        
        # Rename files
        renamed_files = []
        errors = []
        
        # Rename original file if it exists
        if os.path.exists(old_original_path):
            try:
                os.rename(old_original_path, new_original_path)
                renamed_files.append(f'original: {old_filename} -> {new_name}')
            except OSError as e:
                errors.append(f'Failed to rename original file: {e}')
        
        # Rename converted file if it exists
        if os.path.exists(old_converted_path):
            try:
                os.rename(old_converted_path, new_converted_path)
                renamed_files.append(f'converted: {old_filename} -> {new_name}')
            except OSError as e:
                errors.append(f'Failed to rename converted file: {e}')
        
        # Also rename thumbnail if it exists
        old_thumb_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, 'meme_thumbnails', 
                                     old_filename.replace('.mp3', '.png'))
        new_thumb_path = os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, 'meme_thumbnails', 
                                     new_name.replace('.mp3', '.png'))
        if os.path.exists(old_thumb_path):
            try:
                os.rename(old_thumb_path, new_thumb_path)
                renamed_files.append(f'thumbnail: {old_filename}.png -> {new_name}.png')
            except OSError as e:
                errors.append(f'Failed to rename thumbnail: {e}')
        
        # Update the global MEME_SOUNDS list
        MEME_SOUNDS = [f for f in os.listdir(MEME_SOUNDS_FOLDER_CONVERTED) 
                       if f.endswith('.mp3') and os.path.isfile(os.path.join(MEME_SOUNDS_FOLDER_CONVERTED, f))]
        
        if renamed_files:
            message = f'Successfully renamed sound to: {new_name}'
            if errors:
                message += f' (with warnings: {"; ".join(errors)})'
            return jsonify({
                'message': message,
                'old_name': old_filename,
                'new_name': new_name,
                'renamed_files': renamed_files,
                'warnings': errors if errors else None
            }), 200
        else:
            return jsonify({
                'error': f'No files found to rename for sound: {old_filename}',
                'errors': errors
            }), 404
        
    except Exception as e:
        logger.error(f'Error renaming sound: {e}')
        return jsonify({'error': f'Rename failed: {str(e)}'}), 500


@app.route('/')
def index():
    """
    Render the index.html template
    """
    return render_template('index.html')


# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    servos.reset_servos()
    convert_all_sound_files()

    # Ensure all motors are stopped before enabling joystick control
    try:
        motors.stop()
        logging.info('Motors stopped during startup initialization')
    except Exception as e:
        logging.warning(f'Failed to stop motors during startup: {e}')

    # NOTE: Joystick control is now handled by separate nerdbot-joystick service
    # to avoid conflicts. The integrated joystick control below is disabled.
    # 
    # # Enable integrated joystick control (zero-latency)
    # try:
    #     motors.enable_joystick_control()
    #     servos.enable_joystick_control()
    #     from joystick_input_manager import get_input_manager
    #     input_manager = get_input_manager()
    #     if input_manager.start():
    #         logging.info('Integrated joystick control enabled')
    #     else:
    #         logging.warning('Joystick controller not found - API control only')
    # except Exception as e:
    #     logging.warning(f'Joystick integration failed: {e} - API control only')
    
    logging.info('Joystick control handled by separate nerdbot-joystick service')

    # Start the front camera stream
    # cam0()

    # NOTE: Automated detection tracking is disabled to prevent conflicts with joystick control
    # The detection tracking system competes with joystick motor commands causing chattering
    #
    # # Start a thread to track the largest object in the front camera feed
    # threading.Thread(target=track_detections_front_startup, daemon=True).start()
    # logging.info('Detection tracking started')
    
    logging.info('Automated detection tracking disabled - joystick control only')
    # threading.Thread(target=audio_stream, daemon=True).start()
    # logging.info('Audio stream started')

    # threading.Thread(target=start_audio_stream_startup, daemon=True).start()

    # List all the available audio devices
    # p = pyaudio.PyAudio()
    # for i in range(p.get_device_count()):
    #     dev = p.get_device_info_by_index(i)
    #     if dev['maxInputChannels'] > 0:
    #         logging.info('Input Device, index: %s, name: %s', i, dev['name'])

    # Add delayed motor stop to ensure all motors are stopped after initialization
    def delayed_motor_stop():
        time.sleep(1.0)  # Wait for all initialization to complete
        try:
            motors.stop()
            logging.info('Delayed motor stop completed - all motors should be stopped')
        except Exception as e:
            logging.warning(f'Delayed motor stop failed: {e}')
    
    # Start delayed stop in background thread
    threading.Thread(target=delayed_motor_stop, daemon=True).start()

    app.run(debug=False, host = '0.0.0.0', port=5000)
