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
# import wave
# import audioop
from collections import deque
from functools import lru_cache
import cv2
from flask import Flask, Response, render_template, jsonify  # , stream_with_context
from flask_restful import Api
from flask_cors import CORS
from flask_apscheduler import APScheduler
import libcamera
import numpy as np
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
# from light_bar import light_bar


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
# lightbar = light_bar.LightBar()

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
gemini_vision_prompt = """You are a sophisticated robo-cat assistant with keen
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
        raise RuntimeError(f"Failed to get audio devices: {e}")

AUDIO_DEVICE_INDEX = get_usb_audio()

# TTS Variables
PIPER_DIR = "/home/mark/.local/bin"
TTS_MODEL = "/home/mark/nerdbot-backend/en_US-lessac-medium.onnx"


def piper_tts(text, mode=None):
    """
    Generate speech using Piper TTS with proper pipeline handling
    """

    modes = [
        'slow',
        'fast',
        'yelling',
        'whispering',
        'robotic',
        'echo',
        'reverse',
    ]

    # Sanitize input text
    text_regex = re.compile(r'[^A-Za-z0-9 ]+')
    text = text_regex.sub('', text)

    try:
        # Create pipeline using Popen with error capture
        p1 = subprocess.Popen(['echo', text], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(
            [f'{PIPER_DIR}/piper', '--model', TTS_MODEL, '--output-raw'],
            stdin=p1.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        plughwstr = f'plughw:{AUDIO_DEVICE_INDEX},0'
        p3 = subprocess.Popen(
            ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw', '-c', '1', '-v', '--device', plughwstr],
            stdin=p2.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Close previous pipes
        p1.stdout.close()
        p2.stdout.close()

        # Wait and capture output
        stdout, stderr = p3.communicate()  # pylint: disable=unused-variable

        # Check each process
        if p1.returncode and p1.returncode != 0:
            logging.error("Echo failed: %s", p1.stderr.read().decode())
        if p2.returncode and p2.returncode != 0:
            logging.error("Piper failed: %s", p2.stderr.read().decode())
        if p3.returncode != 0:
            logging.error("Aplay failed: %s", stderr.decode())
            raise subprocess.CalledProcessError(p3.returncode, 'aplay', stderr.decode())

        return "Text to Speech Complete"

    except Exception as e:
        logging.error("TTS Pipeline Error: %s", str(e))
        # Cleanup
        for p in [p1, p2, p3]:
            try:
                p.kill()
            except Exception as ke:  # pylint: disable=broad-except
                logging.error("Error killing process: %s", ke)
        raise


# def create_wav_header(sample_rate, bits_per_sample, channels):
#     datasize = 2000*10**6  # Some large number to act as a placeholder
#     o = bytes("RIFF", 'ascii')                                               # (4byte) Marks file as RIFF
#     o += struct.pack('<I', datasize + 36)                                     # (4byte) File size in bytes excluding this and RIFF marker
#     o += bytes("WAVE", 'ascii')                                               # (4byte) File type
#     o += bytes("fmt ", 'ascii')                                               # (4byte) Format Chunk Marker
#     o += struct.pack('<I', 16)                                                # (4byte) Length of above format data
#     o += struct.pack('<H', 1)                                                 # (2byte) Format type (1 - PCM)
#     o += struct.pack('<H', channels)                                          # (2byte)
#     o += struct.pack('<I', sample_rate)                                       # (4byte)
#     o += struct.pack('<I', sample_rate * channels * bits_per_sample // 8)     # (4byte)
#     o += struct.pack('<H', channels * bits_per_sample // 8)                   # (2byte)
#     o += struct.pack('<H', bits_per_sample)                                   # (2byte)
#     o += bytes("data", 'ascii')                                               # (4byte) Data Chunk Marker
#     o += struct.pack('<I', datasize)                                          # (4byte) Data size in bytes
#     return o


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
        self.buffer.truncate()
        self.buffer.seek(0)
        self.buffer.write(buf)
        with self.condition:
            self.frame = self.buffer.getvalue()
            self.condition.notify_all()

    def write_audio(self, data):
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
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
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
        with open("assets/coco_labels.txt", "r", encoding="utf-8") as fi:
            intrinsics.labels = fi.read().splitlines()
        logger.info("Labels: %s", intrinsics.labels)
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


# def genHeader(sampleRate, bitsPerSample, channels):
#     """
#     Generate the WAV header
#     """
#     datasize = 2000*10**6
#     o = bytes("RIFF",'ascii')
#     o += (datasize + 36).to_bytes(4,'little')
#     o += bytes("WAVE",'ascii')
#     o += bytes("fmt ",'ascii')
#     o += (16).to_bytes(4,'little')
#     o += (1).to_bytes(2,'little')
#     o += (channels).to_bytes(2,'little')
#     o += (sampleRate).to_bytes(4,'little')
#     o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')
#     o += (channels * bitsPerSample // 8).to_bytes(2,'little')
#     o += (bitsPerSample).to_bytes(2,'little')
#     o += bytes("data",'ascii')
#     o += (datasize).to_bytes(4,'little')
#     return o


# @app.route('/audio')
# def audio():
#     """
#     Stream the audio feed
#     """
#     logger = logging.getLogger(__name__)

#     def audio_generator():
#         """
#         Generate audio frames
#         """
#         while True:
#             frame = AUDIO_OUTPUT.read_frame()
#             logger.info('Frame: %s', frame)
#             if frame is None:
#                 continue
#             yield (b'--frame\r\n'
#                    b'Content-Type: audio/wav\r\n\r\n' + frame + b'\r\n')
            
#     return Response(audio_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


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


# @scheduler.task('date', id='start_audio_stream_startup', next_run_time=datetime.now() + timedelta(seconds=10))
# def start_audio_stream_startup():
#     """
#     Start the audio stream
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('Starting audio stream')
#     global AUDIO_OUTPUT  # pylint: disable=global-statement

#     AUDIO_OUTPUT = start_audio_stream()


# Start the detection tracking job on server startup
@scheduler.task('date', id='track_detections_startup', next_run_time=datetime.now() + timedelta(seconds=10))
def track_detections_front_startup():
    """
    Track the detections in the front camera feed
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting detection tracking')
    labels = imx500_get_labels()
    while True:
        largest_object = None
        largest_object_label = None
        label = None

        for detection in DETECTIONS['front_camera']:
            # Check the box size
            if largest_object is None or detection.box[2] * detection.box[3] > largest_object.box[2] * largest_object.box[3]:  # pylint: disable=line-too-long
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
                # logger.info(dir(detection))
                largest_object = detection
                largest_object_label = label.split(' ')[0]

        # logger.info('Largest Object: %s', largest_object_label)

        if largest_object is not None and largest_object_label in DETECT_LABELS:
            # logger.info('Tracking Object: %s', label)
            x, y, w, h = largest_object.box  # pylint: disable=unused-variable
            # Move the servos to track the object across a 640x480 frame
            if x + (w / 2) < 280:
                try:
                    servos.pan('left')
                except ValueError:
                    pass
            elif x + (w / 2) > 360:
                try:
                    servos.pan('right')
                except ValueError:
                    pass
            if y - (h / 2) < 200:
                try:
                    servos.tilt('up')
                    # Tilt up a couple more times to get to a person's face
                    servos.tilt('up')
                    servos.tilt('up')
                except ValueError:
                    pass
            elif y - (h / 2) > 280:
                try:
                    servos.tilt('down')
                except ValueError:
                    pass

            # If the bounding box has an edge near the image edge, pan / tilt towards it
            if x < 50:
                try:
                    servos.pan('left')
                except ValueError:
                    pass
            elif x + w > 590:
                try:
                    servos.pan('right')
                except ValueError:
                    pass
            if y < 50:
                try:
                    servos.tilt('up')
                except ValueError:
                    pass
            elif y + h > 430:
                try:
                    servos.tilt('down')
                except ValueError:
                    pass

        time.sleep(0.5)


@app.route('/api/meme_sound/<int:sound_id>', methods=['GET', 'POST'])
def meme_sound(sound_id):
    """
    Play the meme sound with the specified sound_id
    """
    if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
        return jsonify({'message': 'Invalid sound_id'}), 400

    sound = MEME_SOUNDS[sound_id]

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
        return jsonify({'message': f'Meme Sound Played: {sound}'}), 200
    except Exception as e:  # pylint: disable=broad-except
        pa_device_inf = audio1.get_default_output_device_info()
        logger.info('Default Output Device Info: %s', pa_device_inf)
        logger.error('Error: %s while playing %s', e, sound)
        logger.error(traceback.format_exc())
        return jsonify({'message': f'Error: {e} while playing {sound}'}), 500


# audio_output = StreamingOutput()


# @app.route('/audio')
# def audio_feed():
#     """Stream WAV audio"""
#     def generate():
#         FRAMES_PER_CHUNK = 10
#         SAMPLE_WIDTH = 2  # 16-bit audio
#         SAMPLE_RATE = 44100
#         CHANNELS = 1
        
#         # Write WAV header
#         header = (
#             b'RIFF' +
#             (36 + SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS).to_bytes(4, 'little') +
#             b'WAVE' +
#             b'fmt ' +
#             (16).to_bytes(4, 'little') +
#             (1).to_bytes(2, 'little') +
#             CHANNELS.to_bytes(2, 'little') +
#             SAMPLE_RATE.to_bytes(4, 'little') +
#             (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS).to_bytes(4, 'little') +
#             (SAMPLE_WIDTH * CHANNELS).to_bytes(2, 'little') +
#             (SAMPLE_WIDTH * 8).to_bytes(2, 'little') +
#             b'data' +
#             (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS).to_bytes(4, 'little')
#         )
#         yield header

#         # Stream audio data
#         while True:
#             frames = []
#             with audio_output.condition:
#                 for _ in range(FRAMES_PER_CHUNK):
#                     if not audio_output.frames:
#                         if not audio_output.condition.wait(timeout=0.1):
#                             break
#                     if audio_output.frames:
#                         frames.append(audio_output.frames.popleft())

#                 if frames:
#                     chunk = b''.join(frames)
#                     yield chunk

#     return Response(
#         generate(),
#         mimetype='audio/wav',
#         headers={
#             'Cache-Control': 'no-cache',
#             'Connection': 'keep-alive',
#             'Transfer-Encoding': 'chunked'
#         }
#     )


# def audio_stream():
#     """Capture audio feed"""
#     logger = logging.getLogger(__name__)
#     p = pyaudio.PyAudio()

#     # Log and find working input device
#     device_count = p.get_device_count()
#     logger.info("Found %d audio devices", device_count)
    
#     input_device = 1

#     stream = p.open(
#         format=pyaudio.paInt16,
#         channels=2,
#         rate=44100,
#         input=True,
#         frames_per_buffer=1024,
#         input_device_index=input_device
#     )

#     if not stream.is_active():
#         raise RuntimeError("Stream failed to start")

#     logger.info("Audio stream started")

#     try:
#         # start_time = time.time()
#         # frames_read = 0
        
#         GAIN = 2.0

#         while True:
#             # logger.info('Reading audio frame %d', frames_read)
#             data = stream.read(2048, exception_on_overflow=False)
            
#             # Calculate audio levels
#             # rms = audioop.rms(data, 2)
#             # peak = audioop.max(data, 2)
#             # logger.info('Audio frame %d - RMS: %d, Peak: %d', 
#             #             frames_read, rms, peak)
            
#             # Track timing
#             # frames_read += 1
#             # if frames_read % 100 == 0:
#             #     elapsed = time.time() - start_time
#             #     fps = frames_read / elapsed
#             #     logger.info('FPS: %.2f, Total Frames: %d', fps, frames_read)
            
#             # audio_output.write_audio(data)

#             # Convert to numpy array for processing
#             samples = np.frombuffer(data, dtype=np.int16)
            
#             # Calculate RMS level
#             # rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
#             # if rms > 0:
#             #     logger.info('Input RMS level: %.2f', rms)
            
#             # Apply gain
#             samples = np.clip(samples * GAIN, -32768, 32767).astype(np.int16)
            
#             # Monitor output levels
#             # output_rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
#             # if output_rms > 0:
#             #     logger.info('Output RMS level: %.2f', output_rms)
            
#             audio_output.write_audio(samples.tobytes())
            
#     except Exception as e:
#         logger.error('Stream error: %s', str(e))
#         logger.error('Error details: %s', traceback.format_exc())
#     finally:
#         stream.stop_stream()
#         stream.close() 
#         p.terminate()


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
    # Get the front camera feed
    front_image = FRONT_CAMERA_OUTPUT.read_frame()
    front_image = Image.open(io.BytesIO(front_image))

    # Get the rear camera feed
    rear_image = REAR_CAMERA_OUTPUT.read_frame()
    rear_image = Image.open(io.BytesIO(rear_image))

    # Process the front camera feed
    front_results = gemini_vision_model.generate_content([gemini_vision_prompt, front_image])
    logger.info('Front Camera Results: %s', front_results)
    results[0]['front'] = front_results.text

    # Process the rear camera feed
    rear_results = gemini_vision_model.generate_content([gemini_vision_prompt, rear_image])
    logger.info('Rear Camera Results: %s', rear_results)
    results[0]['rear'] = rear_results.text

    return results


@app.route('/api/visual_awareness', methods=['GET', 'POST'])
def visual_awareness_api():
    """
    Visual awareness API endpoint
    """
    results = visual_awareness()
    return results, 200


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
