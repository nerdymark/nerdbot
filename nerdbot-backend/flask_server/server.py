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
from flask import Flask, Response, render_template, jsonify, request  # , stream_with_context
from flask_restful import Api
from flask_cors import CORS
from flask_apscheduler import APScheduler
import libcamera
import numpy as np
import psutil
import pyaudio
from pydub import AudioSegment
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from picamera2.devices import Hailo, IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.devices.imx500.postprocess import scale_boxes
from PIL import Image
import google.generativeai as genai
from motor_control import motors
from servo_control import servos
from x120 import bat
from light_bar.light_bar import light_bar
import sys
sys.path.append('/home/mark/nerdbot-backend')
from mode_manager import mode_manager
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

DETECTIONS_USE_HAILO = False  # Use IMX500 board for object detection

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

CORS(app)  # Enable CORS for all routes

GOOGLE_GEMINI_KEY = config.get('google_gemini_key')
MEME_SOUNDS_FOLDER = config.get('meme_sounds_folder')
MEME_SOUNDS_FOLDER_CONVERTED = config.get('meme_sounds_folder_converted')
MEME_SOUNDS = os.listdir(MEME_SOUNDS_FOLDER_CONVERTED)
# MEME_SOUNDS = os.listdir(MEME_SOUNDS_FOLDER)

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
gemini_vision_model = genai.GenerativeModel('models/gemini-1.5-flash')
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

Stay helpful but chaotic. You're here to assist AND be the most entertaining robot cat ever built.
No cap!
"""


imx500 = IMX500(IMX500_DETECT_MODEL)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"
elif intrinsics.task != "object detection":
    print("Network is not an object detection task", file=sys.stderr)
    exit()


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
    text_regex = re.compile(r'[^A-Za-z0-9\s\.,\'\"\-\?\!]+')
    text = text_regex.sub('', text)
    logger.info("TTS Text: %s", text)
    
    # Set light bar to speaking state
    try:
        light_bar.set_robot_state("speaking")
    except Exception as e:
        logger.warning(f"Light bar speaking effect failed: {e}")

    sentence_sleep = 0.5
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
            
            logger.info(f"TTS Pitch: +{total_pitch_shift:.1f} semitones, Tempo: {tempo_variation:.2f}x")

            # Create pipeline: echo -> piper -> sox -> aplay
            p1 = subprocess.Popen(
                ['echo', sentence],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
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

            # Add SoX for pitch and tempo processing
            p3 = subprocess.Popen([
                'sox',
                '-t', 'raw',          # Input type: raw
                '-r', '22050',        # Sample rate: 22050 Hz
                '-e', 'signed',       # Encoding: signed
                '-b', '16',           # Bits: 16-bit
                '-c', '1',            # Channels: mono
                '-',                  # Input from stdin
                '-t', 'raw',          # Output type: raw  
                '-r', '22050',        # Keep same sample rate
                '-e', 'signed',       # Keep same encoding
                '-b', '16',           # Keep same bits
                '-c', '1',            # Keep mono
                '-',                  # Output to stdout
                'pitch', str(int(total_pitch_shift * 100)),  # Pitch shift in cents (100 cents = 1 semitone)
                'tempo', str(tempo_variation),               # Tempo adjustment
            ], stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Allow p2 to receive SIGPIPE if p3 exits  
            p2.stdout.close()

            p4 = subprocess.Popen(
                [
                    '/usr/bin/aplay',
                    '-r', '22050',
                    '-f', 'S16_LE', 
                    '-t', 'raw',
                    '-c', '1',
                    '-v',
                    '--device', plughwstr
                ],
                stdin=p3.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Allow p3 to receive SIGPIPE if p4 exits
            p3.stdout.close()

            # Capture stderr from p2 and p3 before they're closed
            p2_stderr_data = p2.stderr.read()
            p2.stderr.close()

            # Wait for completion and check errors
            _, stderr4 = p4.communicate()
            p3.wait()
            p2.wait() 
            p1.wait()

            if p1.returncode != 0:
                logger.error("Echo failed with return code: %d", p1.returncode)
                raise subprocess.CalledProcessError(p1.returncode, 'echo')
            if p2.returncode != 0:
                logger.error("Piper stderr: %s", p2_stderr_data.decode() if p2_stderr_data else "")
                raise subprocess.CalledProcessError(p2.returncode, 'piper')
            if p3.returncode != 0:
                logger.error("SoX failed with return code: %d", p3.returncode)
                raise subprocess.CalledProcessError(p3.returncode, 'sox')
            if p4.returncode != 0:
                logger.error("aplay failed with return code: %d", p4.returncode)
                raise subprocess.CalledProcessError(p4.returncode, 'aplay')

            time.sleep(sentence_sleep)

        except Exception as e:
            logger.error("TTS Pipeline Error: %s", str(e))
            for p in [p1, p2, p3, p4]:
                try:
                    p.kill()
                except Exception as ke:
                    logger.error("Error killing process: %s", ke)
            raise

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

    def write(self, buf):
        """
        Write the buffer to the frame data
        """
        self.buffer.truncate()
        self.buffer.seek(0)
        self.buffer.write(buf)
        with self.condition:
            self.frame = self.buffer.getvalue()
            self.condition.notify_all()

    def write_audio(self, data):
        """
        Write the audio data
        """
        with self.condition:
            self.frames.append(data)
            self.last_write = time.time()
            self.condition.notify()

    def read_frame(self):
        """
        Read the frame data
        """
        with self.condition:
            self.condition.wait()
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
        try:
            with Hailo(model) as hailo:
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
                            logger.error('Error processing frame: %s', e)
        except Exception as e:  # pylint: disable=broad-except
            logger.error('Error: %s', e)

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
        recording = False
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
            logger.error('Camera thread encountered PiCameraError: %s', e)
            if recording:
                try:
                    picam2.stop_recording()
                    logger.info("Camera 1 recording stopped due to error.")
                except Exception as se:  # pylint: disable=broad-except
                    logger.error('Error stopping recording: %s', se)
        finally:
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
    Stream the front camera feed
    """
    logger = logging.getLogger(__name__)
    global FRONT_CAMERA_OUTPUT  # pylint: disable=global-statement
    if FRONT_CAMERA_OUTPUT is None and DETECTIONS_USE_HAILO:
        logger.info('Using Hailo for object detection')
        FRONT_CAMERA_OUTPUT = run_detection_camera_0_hailo(
            HAILO_DETECT_MODEL,
            DETECT_LABELS,
            SCORE_THRESH
            )
    elif FRONT_CAMERA_OUTPUT is None:
        logger.info('Using IMX500 for object detection')
        FRONT_CAMERA_OUTPUT = run_detection_camera_0_imx500()

    def frame_generator():
        """
        Generate frames from the front camera feed
        """
        while True:
            frame = FRONT_CAMERA_OUTPUT.read_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cam1')
def cam1():
    """
    Stream the rear camera feed
    """
    logger = logging.getLogger(__name__)
    global REAR_CAMERA_OUTPUT  # pylint: disable=global-statement
    if REAR_CAMERA_OUTPUT is None:
        logger.info('Starting rear camera stream')
        REAR_CAMERA_OUTPUT = start_camera_1()

    def frame_generator():
        """
        Generate frames from the rear camera feed
        """
        while True:
            frame = REAR_CAMERA_OUTPUT.read_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
        servo_angle = servos.tilt('forward')
        TILT_ANGLE = servo_angle
        print(f"Tilted forward to {servo_angle}")

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


@app.route('/api/tts/<string:text>', methods=['GET', 'POST'])
def tts(text):
    """
    Convert text to speech using Piper TTS
    """
    result = piper_tts(text)
    return jsonify({'message': result}), 200  # Return JSON response with HTTP 200 status


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
    return jsonify({'mode': mode_manager.get_mode()}), 200


@app.route('/api/mode/<string:mode>', methods=['POST'])
def set_mode(mode):
    """
    Set robot operational mode
    """
    success = mode_manager.set_mode(mode)
    if success:
        return jsonify({'mode': mode_manager.get_mode(), 'message': f'Mode set to {mode}'}), 200
    else:
        return jsonify({'error': f'Invalid mode: {mode}'}), 400


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
    Get the list of meme sounds
    """
    return jsonify(MEME_SOUNDS), 200


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
        
        global MEME_SOUNDS
        # Update the global MEME_SOUNDS list
        MEME_SOUNDS = os.listdir(MEME_SOUNDS_FOLDER_CONVERTED)
        
        return jsonify({
            'message': f'Successfully added sound: {filename}',
            'filename': filename,
            'converted_path': converted_path
        }), 200
        
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
    
        # Update the global MEME_SOUNDS list
        MEME_SOUNDS = os.listdir(MEME_SOUNDS_FOLDER_CONVERTED)

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


def convert_sound_file(sound_file):
    """
    Convert the sound file to the preferred format with the correct sample
    rate using wave and pyaudio
    """
    logger = logging.getLogger(__name__)
    meme_audio = open(MEME_SOUNDS_FOLDER + sound_file, 'rb').read()
    out_file = MEME_SOUNDS_FOLDER_CONVERTED + sound_file

    if os.path.exists(out_file):
        return out_file

    if not os.path.exists(MEME_SOUNDS_FOLDER_CONVERTED):
        os.makedirs(MEME_SOUNDS_FOLDER_CONVERTED)

    meme_audio = AudioSegment.from_mp3(io.BytesIO(meme_audio))
    meme_audio = meme_audio.set_frame_rate(AUDIO_RATE)
    meme_audio = meme_audio.set_channels(1)
    meme_audio = meme_audio.set_sample_width(2)
    meme_audio.export(MEME_SOUNDS_FOLDER_CONVERTED + sound_file, format='mp3')
    logger.info('Converted %s', sound_file)
    return out_file


def convert_all_sound_files():
    """
    Convert all the sound files in the MEME_SOUNDS_FOLDER to the preferred format
    """
    logger = logging.getLogger(__name__)
    for sound in MEME_SOUNDS:
        logger.info('Converting %s', sound)
        convert_sound_file(sound)


# Initialize the optimized tracker
tracker = OptimizedTracker(servos)

# Start the detection tracking job on server startup
@scheduler.task('date', id='track_detections_startup', next_run_time=datetime.now() + timedelta(seconds=10))  # pylint: disable=line-too-long
def track_detections_front_startup():
    """
    Track the detections in the front camera feed using optimized tracker
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting optimized detection tracking')
    labels = imx500_get_labels()
    
    # Update rate for stable tracking (increased to 200ms for less seeking)
    update_interval = 0.2
    
    while True:
        # Only track if in a follow mode or idle mode
        current_mode = mode_manager.get_mode()
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
    Play the meme sound with the specified sound_id
    """
    if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
        return jsonify({'message': 'Invalid sound_id'}), 400

    sound = MEME_SOUNDS[sound_id]
    
    # Set light bar to audio reactive mode
    try:
        light_bar.audio_reactive(0.7)  # Medium-high intensity for memes
    except Exception as e:
        logging.warning(f"Light bar audio effect failed: {e}")

    audio1 = pyaudio.PyAudio()

    stream = audio1.open(format=pyaudio.paInt16,
                         output_device_index=AUDIO_DEVICE_INDEX,
                         channels=1,
                         rate=AUDIO_RATE,
                         frames_per_buffer=AUDIO_CHUNK,
                         output=True)

    # Read and process the audio file
    meme_audio = open(MEME_SOUNDS_FOLDER_CONVERTED + sound, 'rb').read()
    meme_audio = AudioSegment.from_mp3(io.BytesIO(meme_audio))
    meme_audio = meme_audio.set_frame_rate(AUDIO_RATE)
    meme_audio = meme_audio.set_channels(1)
    meme_audio = meme_audio.set_sample_width(2)
    meme_audio = meme_audio.normalize()

    # Convert AudioSegment to raw audio data
    raw_data = meme_audio.raw_data

    # Write raw audio data to the stream in chunks
    for i in range(0, len(raw_data), AUDIO_CHUNK):
        stream.write(raw_data[i:i + AUDIO_CHUNK])

    stream.stop_stream()
    stream.close()
    
    # Return to idle after playing sound
    try:
        light_bar.set_robot_state("idle")
    except Exception as e:
        logging.warning(f"Light bar idle effect failed: {e}")
    
    return jsonify({'message': 'Meme Sound Played'}), 200


@app.route('/api/meme_sound/random', methods=['GET', 'POST'])
def random_meme_sound():
    """
    Play a random meme sound from the MEME_SOUNDS_FOLDER_CONVERTED
    """
    logger = logging.getLogger(__name__)
    shuffle(MEME_SOUNDS)  # Shuffle the list to avoid repetition
    sound_id = randint(0, len(MEME_SOUNDS) - 1)
    sound = MEME_SOUNDS[sound_id]
    logger.info('Playing %s', sound)
    
    # Set light bar to celebration mode for random sounds
    try:
        light_bar.celebration()
    except Exception as e:
        logger.warning(f"Light bar celebration effect failed: {e}")
    
    try:
        audio1 = pyaudio.PyAudio()

        stream = audio1.open(format=pyaudio.paInt16,
                             output_device_index=AUDIO_DEVICE_INDEX,
                             channels=1,
                             rate=AUDIO_RATE,
                             frames_per_buffer=AUDIO_CHUNK,
                             output=True)
        meme_audio = open(MEME_SOUNDS_FOLDER_CONVERTED + sound, 'rb').read()
        meme_audio = AudioSegment.from_mp3(io.BytesIO(meme_audio))
        meme_audio = meme_audio.set_frame_rate(AUDIO_RATE)
        meme_audio = meme_audio.set_channels(1)
        meme_audio = meme_audio.set_sample_width(2)
        raw_data = meme_audio.raw_data
        stream.write(raw_data)
        time.sleep(len(raw_data) / (AUDIO_RATE * 2))  # Add delay to ensure audio plays completely
        stream.stop_stream()
        stream.close()
        audio1.terminate()  # Ensure resources are released
        
        # Return to idle after playing sound
        try:
            light_bar.set_robot_state("idle")
        except Exception as e:
            logger.warning(f"Light bar idle effect failed: {e}")
        
        return jsonify({'message': f'Meme Sound Played: {sound}'}), 200
    except Exception as e:  # pylint: disable=broad-except
        pa_device_inf = audio1.get_default_output_device_info()
        logger.info('Default Output Device Info: %s', pa_device_inf)
        logger.error('Error: %s while playing %s', e, sound)
        logger.error(traceback.format_exc())
        return jsonify({'message': f'Error: {e} while playing {sound}'}), 500


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
            if "quota" in str(e).lower() or "429" in str(e):
                results[0]['front'] = "🤖 Oops! My visual processing quota is temporarily exhausted. Even robots need a coffee break sometimes! ☕"
                logger.warning('Front camera vision quota exceeded: %s', str(e))
            else:
                results[0]['front'] = "🤖 Visual processing temporarily unavailable. My circuits are having a moment! ⚡"
                logger.error('Front camera vision error: %s', str(e))

        # Process the rear camera feed
        try:
            rear_results = gemini_vision_model.generate_content([gemini_vision_prompt, rear_image])
            logger.info('Rear Camera Results: %s', rear_results)
            results[0]['rear'] = rear_results.text
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                results[0]['rear'] = "🤖 Oops! My visual processing quota is temporarily exhausted. Even robots need a coffee break sometimes! ☕"
                logger.warning('Rear camera vision quota exceeded: %s', str(e))
            else:
                results[0]['rear'] = "🤖 Visual processing temporarily unavailable. My circuits are having a moment! ⚡"
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


@app.route('/api/health', methods=['GET', 'POST'])
def health():
    """
    Health check for watchdog
    """
    return jsonify({'message': 'Healthy'}), 200


@app.route('/')
def index():
    """
    Render the index.html template
    """
    return render_template('index.html')


if __name__ == '__main__':
    servos.reset_servos()
    convert_all_sound_files()

    # Enable integrated joystick control (zero-latency)
    try:
        motors.enable_joystick_control()
        servos.enable_joystick_control()
        from joystick_input_manager import get_input_manager
        input_manager = get_input_manager()
        if input_manager.start():
            logging.info('Integrated joystick control enabled')
        else:
            logging.warning('Joystick controller not found - API control only')
    except Exception as e:
        logging.warning(f'Joystick integration failed: {e} - API control only')

    # Start the front camera stream
    # cam0()

    # Start a thread to track the largest object in the front camera feed
    threading.Thread(target=track_detections_front_startup, daemon=True).start()
    logging.info('Detection tracking started')
    # threading.Thread(target=audio_stream, daemon=True).start()
    # logging.info('Audio stream started')

    # threading.Thread(target=start_audio_stream_startup, daemon=True).start()

    # List all the available audio devices
    # p = pyaudio.PyAudio()
    # for i in range(p.get_device_count()):
    #     dev = p.get_device_info_by_index(i)
    #     if dev['maxInputChannels'] > 0:
    #         logging.info('Input Device, index: %s, name: %s', i, dev['name'])

    app.run(debug=False, host = '0.0.0.0', port=5000)
