import re
import time
import io
import threading
import queue
import subprocess
import traceback
import logging
import json
from random import randint
import wave
import os
import sys
from flask import Flask, Response, render_template, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
import atexit
from datetime import datetime
from threading import Condition
# import picamera2
import numpy as np
import requests
import pyaudio
from pydub import AudioSegment
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder, H264Encoder
from picamera2.outputs import FileOutput, FfmpegOutput
from picamera2.devices import Hailo, IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from functools import lru_cache
# from piper.voice import PiperVoice
import sounddevice as sd
import pyrubberband as pyrb
import cv2
from motor_control import motors
from retry import retry
from servo_control import servos
# from light_bar import light_bar


# COQUI_URL = "http://[::1]:5002"
# COQUI_ENDPOINT = "/api/tts?text="
TILT_ANGLE = None
PAN_ANGLE = None
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 48000
# AUDIO_RATE = 44100
AUDIO_CHUNK = 1024
AUDIO_RECORD_SECONDS = 5
AUDIO_DEVICE_INDEX = 0
DETECTIONS = {
    "front_camera": [],
    "rear_camera": []
}

DETECTIONS_USE_HAILO = False  # Use IMX500 board for object detection

CONFIG_FILE = '/home/mark/nerdbot-scripts/config.json'
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

LOG_FILE = config.get('log_file')
logging.basicConfig(level=logging.INFO, filename=LOG_FILE)
logging.info("Logging to file %s", LOG_FILE)
logging.getLogger('flask_cors').level = logging.DEBUG

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)
audio1 = pyaudio.PyAudio()
# lightbar = light_bar.LightBar()

CORS(app)  # Enable CORS for all routes

MEME_SOUNDS_FOLDER = config.get('meme_sounds_folder')
MEME_SOUNDS_FOLDER_CONVERTED = config.get('meme_sounds_folder_converted')
MEME_SOUNDS = os.listdir(MEME_SOUNDS_FOLDER_CONVERTED)

DETECT_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "bird",
    "cat",
    "dog",
    "sports ball",
    "cell phone"
    ]

HAILO_DETECT_MODEL = "/usr/share/hailo-models/yolov8s_h8l.hef"
IMX500_DETECT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
IMX500_COCO_LABELS_FILE = "/usr/share/imx500-models/coco_labels.txt"
IMX500_IOU = 0.65
IMX500_MAX_DETECTIONS = 10
SCORE_THRESH = 0.5


imx500 = IMX500(IMX500_DETECT_MODEL)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"
elif intrinsics.task != "object detection":
    print("Network is not an object detection task", file=sys.stderr)
    exit()

# TTS Variables
PIPER_DIR = "/home/mark/.local/bin"
TTS_MODEL = "/home/mark/nerdbot-backend/en_US-lessac-medium.onnx"
# TTS_VOICE = PiperVoice.load(TTS_MODEL)


def piper_tts(text):
    """
    Generate speech using Piper TTS with proper pipeline handling
    """

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
        p3 = subprocess.Popen(
            ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw', '-c', '1', '-v'],
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
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        self.buffer.truncate()
        self.buffer.seek(0)
        self.buffer.write(buf)
        with self.condition:
            self.frame = self.buffer.getvalue()
            self.condition.notify_all()

    def read_frame(self):
        """
        Read the frame data
        """
        with self.condition:
            self.condition.wait()
            return self.frame


class IMX500Detection:
    def __init__(self, coords, category, conf, metadata):
        """
        Create a Detection object, recording the bounding box, category and confidence.
        """
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)



def parse_detections_imx500(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global DETECTIONS
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = SCORE_THRESH
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
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


@lru_cache
def imx500_get_labels():
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
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


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


def run_detection_camera_0_hailo(model, labels, score_thresh):
    """
    Run object detection on the front camera feed
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


def start_camera_1():
    """
    Use camera 1 to stream video to StreamOutput
    """
    with Picamera2(camera_num=1) as picam2:
        main = {'size': (640, 480), 'format': 'XRGB8888'}
        lores = {'size': (640, 480), 'format': 'RGB888'}
        controls = {'FrameRate': 30}
        picamera_config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
        picam2.configure(picamera_config)

        output = StreamingOutput()
        picam2.start_recording(JpegEncoder(), FileOutput(output))

        while True:
            frame = output.read_frame()
            if frame is None:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Start the rear camera stream
FRONT_CAMERA_OUTPUT = None
REAR_CAMERA_OUTPUT = start_camera_1()


@app.route('/cam0')
def cam0():
    """
    Stream the front camera feed
    """
    global FRONT_CAMERA_OUTPUT  # pylint: disable=global-statement
    if FRONT_CAMERA_OUTPUT is None and DETECTIONS_USE_HAILO:
        FRONT_CAMERA_OUTPUT = run_detection_camera_0_hailo(
            HAILO_DETECT_MODEL,
            DETECT_LABELS,
            SCORE_THRESH
            )

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
    return Response(REAR_CAMERA_OUTPUT,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# @retry(delay=2, backoff=3, max_delay=10)
# def coqui_tts(text):
#     """
#     Send the text to the Coqui TTS server and return the audio
#     """
#     response = requests.get(COQUI_URL + COQUI_ENDPOINT + text)
#     audio = response.content
#     return audio


# @retry(delay=2, backoff=3, max_delay=10)
# def play_audio(audio):
#     """
#     Speed up the audio with pydub
#     Play the audio using PyAudio
#     """
#     audio = AudioSegment.from_wav(io.BytesIO(audio))
#     print('Audio Length:', len(audio))
#     audio = audio.speedup(playback_speed=1.15)
#     print('new Audio Length:', len(audio))
#     # Make sure the audio is in the correct format
#     audio = audio.set_frame_rate(AUDIO_RATE)
#     audio = audio.set_channels(1)
#     audio = audio.set_sample_width(2)  # 2 bytes for paInt16

#     # Convert AudioSegment to raw audio data
#     raw_data = audio.raw_data

#     # Open a stream
#     stream = audio1.open(format=pyaudio.paInt16,
#                          output_device_index=AUDIO_DEVICE_INDEX,
#                          channels=1,
#                          rate=AUDIO_RATE,
#                          frames_per_buffer=AUDIO_CHUNK,
#                          output=True)

#     # Write raw audio data to the stream
#     stream.write(raw_data)
#     stream.stop_stream()
#     stream.close()

#     return "Audio Played"


# def text_to_speech(text):
#     """
#     Convert text to speech using Coqui TTS
#     """
#     audio = coqui_tts(text)
#     play_audio(audio)
#     return "Text to Speech Complete"

"""
api.add_resource(video_feed_0, '/cam0')
api.add_resource(video_feed_1, '/cam1')
"""

@app.route('/api/motor/<string:direction>', methods=['GET', 'POST'])
def motor_control(direction):
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
    
    return jsonify({'message': 'Motor Control Complete'}), 200  # Return JSON response with HTTP 200 status


@app.route('/api/pan/<string:direction>', methods=['GET', 'POST'])
def pan(direction):
    global PAN_ANGLE
    global TILT_ANGLE
    max_left = None
    max_right = None
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

    return jsonify({'message': f"Panned {direction} to {PAN_ANGLE}"}), 200  # Return JSON response with HTTP 200 status


@app.route('/api/tilt/<string:direction>', methods=['GET', 'POST'])
def tilt(direction):
    global TILT_ANGLE
    max_up = None
    max_down = None
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
    result = piper_tts(text)
    return jsonify({'message': result}), 200  # Return JSON response with HTTP 200 status


def stream_audio():
    # start Recording
    def sound():
        sampleRate = AUDIO_RATE
        bitsPerSample = 16
        channels = 2
        wav_header = genHeader(sampleRate, bitsPerSample, channels)

        stream = audio1.open(output_device_index=AUDIO_DEVICE_INDEX,
                             format=AUDIO_FORMAT,
                             channels=AUDIO_CHANNELS,
                             rate=AUDIO_RATE,
                             input=True,
                             frames_per_buffer=AUDIO_CHUNK)
        print("recording...")
        first_run = True
        while True:
            if first_run:
                data = wav_header + stream.read(AUDIO_CHUNK)
                first_run = False
            else:
                data = stream.read(AUDIO_CHUNK)
            yield data

    return Response(sound(), mimetype="audio/wav")


def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000*10**6  # Some large number to indicate streaming
    o = bytes("RIFF", 'ascii')                                   # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4, 'little')                   # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE", 'ascii')                                  # (4byte) File type
    o += bytes("fmt ", 'ascii')                                  # (4byte) Format Chunk Marker
    o += (16).to_bytes(4, 'little')                              # (4byte) Length of above format data
    o += (1).to_bytes(2, 'little')                               # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2, 'little')                        # (2byte)
    o += (sampleRate).to_bytes(4, 'little')                      # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4, 'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2, 'little')   # (2byte)
    o += (bitsPerSample).to_bytes(2, 'little')                   # (2byte)
    o += bytes("data", 'ascii')                                  # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4, 'little')                        # (4byte) Data size in bytes
    return o


def convert_sound_file(sound_file):
    """
    Convert the sound file to the preferred format with the correct sample rate using wave and pyaudio
    """
    logger = logging.getLogger(__name__)
    audio = open(MEME_SOUNDS_FOLDER + sound_file, 'rb').read()
    out_file = MEME_SOUNDS_FOLDER_CONVERTED + sound_file
    # Skip if the file already exists
    if os.path.exists(out_file):
        return out_file
    # Create the folder if it doesn't exist
    if not os.path.exists(MEME_SOUNDS_FOLDER_CONVERTED):
        os.makedirs(MEME_SOUNDS_FOLDER_CONVERTED)

    # Convert the audio file to the preferred format
    audio = AudioSegment.from_mp3(io.BytesIO(audio))
    audio = audio.set_frame_rate(AUDIO_RATE)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio.export(MEME_SOUNDS_FOLDER_CONVERTED + sound_file, format='wav')
    logger.info(f'Converted {sound_file}')
    return out_file


def convert_all_sound_files():
    """
    Convert all the sound files in the MEME_SOUNDS_FOLDER to the preferred format
    """
    logger = logging.getLogger(__name__)
    for sound in MEME_SOUNDS:
        logger.info(f'Converting {sound}')
        convert_sound_file(sound)
    

@app.route('/api/meme_sound/<int:sound_id>', methods=['GET', 'POST'])
def meme_sound(sound_id):
    if sound_id < 0 or sound_id >= len(MEME_SOUNDS):
        return jsonify({'message': 'Invalid sound_id'}), 400
    
    sound = MEME_SOUNDS[sound_id]
    stream = audio1.open(format=pyaudio.paInt16,
                         output_device_index=AUDIO_DEVICE_INDEX,
                         channels=1,
                         rate=AUDIO_RATE,
                         frames_per_buffer=AUDIO_CHUNK,
                         output=True)
    
    # Read and process the audio file
    audio = open(MEME_SOUNDS_FOLDER_CONVERTED + sound, 'rb').read()
    audio = AudioSegment.from_mp3(io.BytesIO(audio))
    audio = audio.set_frame_rate(AUDIO_RATE)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio = audio.normalize()
    
    # Convert AudioSegment to raw audio data
    raw_data = audio.raw_data
    
    # Write raw audio data to the stream in chunks
    for i in range(0, len(raw_data), AUDIO_CHUNK):
        stream.write(raw_data[i:i + AUDIO_CHUNK])
    
    stream.stop_stream()
    stream.close()
    return jsonify({'message': 'Meme Sound Played'}), 200


@app.route('/api/meme_sound/random', methods=['GET', 'POST'])
def random_meme_sound():
    sound_id = randint(0, len(MEME_SOUNDS) - 1)
    sound = MEME_SOUNDS[sound_id]
    try:
        stream = audio1.open(format=pyaudio.paInt16,
                       output_device_index=AUDIO_DEVICE_INDEX,
                       channels=1,
                       rate=AUDIO_RATE,
                       frames_per_buffer=AUDIO_CHUNK,
                       output=True)
        audio = open(MEME_SOUNDS_FOLDER_CONVERTED + sound, 'rb').read()
        audio = AudioSegment.from_mp3(io.BytesIO(audio))
        audio = audio.set_frame_rate(AUDIO_RATE)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)
        raw_data = audio.raw_data
        stream.write(raw_data)
        stream.stop_stream()
        stream.close()
        return jsonify({'message': f'Meme Sound Played: {sound}'}), 200
    except Exception as e:  # pylint: disable=broad-except
        pa_device_inf = audio1.get_default_output_device_info()
        logging.info('Default Output Device Info: %s', pa_device_inf)
        logging.error('Error: %s while playing %s', e, sound)
        logging.error(traceback.format_exc())
        return jsonify({'message': f'Error: {e} while playing {sound}'}), 500


@app.route('/api/health', methods=['GET', 'POST'])
def health():
    """
    Health check for watchdog
    """
    return jsonify({'message': 'Healthy'}), 200


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    servos.reset_servos()
    convert_all_sound_files()
    # lightbar.turn_on()
    servos.laser_on()
    time.sleep(1)
    servos.laser_off()

    app.run(debug=False, host = '0.0.0.0', port=5000)