"""
Mode Manager for nerdbot behavior control
Handles different operational modes: manual, idle, detect-and-follow
"""
import threading
import time
import random
import logging
from enum import Enum
from motor_control import motors
from servo_control import servos
import requests
import numpy as np
import os
from light_bar.light_bar import light_bar


# Load COCO labels once at module level
COCO_LABELS = []
ASSETS_COCO_LABELS_FILE = os.path.join(os.path.dirname(__file__), "assets/coco_labels.txt")

if os.path.exists(ASSETS_COCO_LABELS_FILE):
    try:
        with open(ASSETS_COCO_LABELS_FILE, 'r') as f:
            COCO_LABELS = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(COCO_LABELS)} COCO labels from {ASSETS_COCO_LABELS_FILE}")
    except Exception as e:
        logging.error(f"Failed to load COCO labels: {e}")
else:
    logging.warning(f"COCO labels file not found at {ASSETS_COCO_LABELS_FILE}")


class RobotMode(Enum):
    MANUAL = "manual"
    IDLE = "idle"
    DETECT_AND_FOLLOW = "detect_and_follow"
    DETECT_AND_FOLLOW_WHEELS = "detect_and_follow_wheels"


class ModeManager:
    def __init__(self):
        self.current_mode = RobotMode.DETECT_AND_FOLLOW
        self.mode_thread = None
        self.mode_running = False
        self.logger = logging.getLogger(__name__)
        self.last_detection = None
        self.idle_timer = 0
        self.previous_detection_category = None  # Track previous detection for new detection events
        self.detection_first_seen_time = None  # Track when we first saw current detection
        # Start default mode with white startup light
        try:
            light_bar.white()
            time.sleep(0.5)
            light_bar.test()  # Run test sequence on startup
            time.sleep(1)
        except:
            pass
        self.start_detect_and_follow_mode()

    def set_mode(self, mode):
        """Switch to a different operational mode"""
        self.logger.info(f"Switching from {self.current_mode.value} to {mode}")

        # Stop current mode
        self.stop_current_mode()

        # Visual feedback for mode transition
        try:
            light_bar.knight_red()  # Use knight_red for mode transitions
            time.sleep(0.5)  # Brief animation
        except Exception as e:
            self.logger.warning(f"Light bar effect failed: {e}")

        # Set new mode
        if mode == "manual":
            self.current_mode = RobotMode.MANUAL
            try:
                light_bar.clear()  # Clear lights for manual mode
            except:
                pass
        elif mode == "idle":
            self.current_mode = RobotMode.IDLE
            self.start_idle_mode()
            try:
                light_bar.idle_animation()  # Start idle animation
            except:
                pass
        elif mode == "detect_and_follow":
            self.current_mode = RobotMode.DETECT_AND_FOLLOW
            self.start_detect_and_follow_mode()
            try:
                light_bar.waterfall_animation()  # Active scanning effect
            except:
                pass
        elif mode == "detect_and_follow_wheels":
            self.current_mode = RobotMode.DETECT_AND_FOLLOW_WHEELS
            self.start_detect_and_follow_wheels_mode()
            try:
                light_bar.red_knight_rider()  # Active tracking effect
            except:
                pass
        else:
            self.logger.error(f"Unknown mode: {mode}")
            return False

        return True

    def get_mode(self):
        """Get current operational mode"""
        return self.current_mode.value

    def stop_current_mode(self):
        """Stop any running mode thread"""
        self.mode_running = False
        if self.mode_thread and self.mode_thread.is_alive():
            self.mode_thread.join(timeout=2.0)
        motors.stop()
        # Clear lights when stopping
        try:
            light_bar.clear()
        except:
            pass

    def start_idle_mode(self):
        """Start idle mode with lifelike behaviors"""
        self.mode_running = True
        self.mode_thread = threading.Thread(target=self._idle_behavior)
        self.mode_thread.start()

    def _idle_behavior(self):
        """Idle behavior implementation - adds lifelike movements, speech, and detection reactions"""
        movement_behaviors = [
            self._look_around,
            self._rock_back_forth,
            self._wiggle,
            self._pan_scan,
            self._tilt_nod,
            self._curious_turn,
        ]

        speech_behaviors = [
            self._introduce_self,
            self._talk_about_capabilities,
            self._express_feelings,
            self._make_observation,
            self._tell_joke,
        ]

        # Greeting on mode start
        self._say_hello()

        speech_counter = 0
        last_greeting_time = 0

        while self.mode_running:
            current_time = time.time()

            # Check for detections and react
            if self.last_detection:
                # Only greet once every 10 seconds to avoid spam
                if current_time - last_greeting_time > 10.0:
                    self._react_to_detection()
                    last_greeting_time = current_time

                # Pan/tilt to track the detected object
                self._track_detection()

                # Shorter wait when actively tracking
                wait_time = random.uniform(1.0, 2.0)
            else:
                # Normal idle behavior when no detections
                behavior = random.choice(movement_behaviors)
                behavior()

                # Occasionally speak (every 3-5 movements)
                speech_counter += 1
                if speech_counter >= random.randint(3, 5):
                    speech_behavior = random.choice(speech_behaviors)
                    speech_behavior()
                    speech_counter = 0

                # Normal wait between behaviors
                wait_time = random.uniform(2.0, 5.0)

            # Wait with ability to interrupt
            self.idle_timer = 0
            while self.idle_timer < wait_time and self.mode_running:
                time.sleep(0.1)
                self.idle_timer += 0.1

    def _look_around(self):
        """Look left and right slowly"""
        if not self.mode_running:
            return

        # Pan left
        for _ in range(3):
            if not self.mode_running:
                break
            servos.pan('left')
            time.sleep(0.2)

        time.sleep(0.5)

        # Pan right
        for _ in range(6):
            if not self.mode_running:
                break
            servos.pan('right')
            time.sleep(0.2)

        time.sleep(0.5)

        # Return to center
        servos.pan('center')

    def _rock_back_forth(self):
        """Rock forward and backward gently"""
        if not self.mode_running:
            return

        # Rock forward
        motors.move_forward(0.3)
        time.sleep(0.2)
        motors.stop()
        time.sleep(0.1)

        # Rock backward
        motors.move_backward(0.3)
        time.sleep(0.2)
        motors.stop()

    def _wiggle(self):
        """Wiggle left and right"""
        if not self.mode_running:
            return

        for _ in range(3):
            if not self.mode_running:
                break
            motors.turn_left(0.5)
            time.sleep(0.1)
            motors.turn_right(0.5)
            time.sleep(0.1)

        motors.stop()

    def _pan_scan(self):
        """Slow panoramic scan"""
        if not self.mode_running:
            return

        # Tilt up slightly
        servos.tilt('up')
        time.sleep(0.2)

        # Slow pan from left to right
        for _ in range(5):
            if not self.mode_running:
                break
            servos.pan('left')
            time.sleep(0.1)

        for _ in range(10):
            if not self.mode_running:
                break
            servos.pan('right')
            time.sleep(0.2)

        # Return to center
        servos.pan('center')
        servos.tilt('center')

    def _tilt_nod(self):
        """Nod up and down"""
        if not self.mode_running:
            return

        for _ in range(2):
            if not self.mode_running:
                break
            servos.tilt('up')
            time.sleep(0.3)
            servos.tilt('down')
            time.sleep(0.3)

        servos.tilt('center')

    def _curious_turn(self):
        """Turn slightly as if curious about something"""
        if not self.mode_running:
            return

        direction = random.choice(['left', 'right'])

        # Tilt head
        servos.tilt('up')
        time.sleep(0.2)

        # Turn body slightly
        if direction == 'left':
            motors.turn_left(0.4)
            servos.pan('left')
        else:
            motors.turn_right(0.4)
            servos.pan('right')

        time.sleep(0.3)
        motors.stop()
        time.sleep(1.0)

        # Return to center
        servos.pan('center')
        servos.tilt('center')

    def start_detect_and_follow_mode(self):
        """Start detect and follow mode"""
        self.mode_running = True
        self.mode_thread = threading.Thread(target=self._detect_and_follow_behavior)
        self.mode_thread.start()

    def _detect_and_follow_behavior(self):
        """Detect and follow behavior - pan/tilt tracking only (no wheel movement)"""
        no_detection_count = 0
        last_detection_category = None

        while self.mode_running:
            if self.last_detection:
                # Reset no detection counter
                no_detection_count = 0

                # Get detection info
                x, y, w, h = self.last_detection['box']
                confidence = self.last_detection['conf']
                category = self.last_detection['category']
                
                # Check if this is a new detection (different category)
                if category != last_detection_category:
                    # New detection - rainbow celebration (shorter for non-wheel mode)
                    try:
                        light_bar.rainbow()
                        time.sleep(0.2)  # Shorter than wheel mode
                    except:
                        pass
                    last_detection_category = category

                # Calculate center of detection
                center_x = x + w / 2
                center_y = y + h / 2

                # Image dimensions (assuming 640x480)
                img_width = 640
                img_height = 480

                # Calculate relative position
                rel_x = center_x / img_width
                rel_y = center_y / img_height

                # Show tracking status with waterfall animation (active tracking mode)
                try:
                    light_bar.waterfall_animation()
                except:
                    pass

                # Pan/tilt tracking is handled by OptimizedTracker in server.py
                # We don't need to do discrete movements here
                pass

            else:
                # No detection - search behavior
                no_detection_count += 1
                
                # Reset detection category when lost
                last_detection_category = None

                if no_detection_count == 1:  # Just lost detection
                    try:
                        light_bar.blue()  # Blue for searching
                    except:
                        pass

                if no_detection_count > 30:  # After 3 seconds
                    # Do a search pattern with pan/tilt only
                    try:
                        light_bar.knight_red()  # Knight rider for active search
                    except:
                        pass
                    
                    servos.pan('left')
                    time.sleep(0.5)
                    servos.pan('right')
                    time.sleep(0.5)
                    servos.pan('center')

                    no_detection_count = 0

            time.sleep(0.1)

    def start_detect_and_follow_wheels_mode(self):
        """Start detect and follow mode with wheel movement"""
        self.mode_running = True
        self.mode_thread = threading.Thread(target=self._detect_and_follow_wheels_behavior)
        self.mode_thread.start()

    def _detect_and_follow_wheels_behavior(self):
        """Detect and follow behavior - includes both pan/tilt and wheel movement"""
        no_detection_count = 0
        # Smoothing variables for wheel movement
        last_turn_direction = None
        turn_smoothing_factor = 0.7
        min_turn_speed = 0.6  # Increased from 0.2 for better movement
        max_turn_speed = 0.9  # Increased from 0.5 for better turning
        last_detection_category = None  # Track for new detection events

        while self.mode_running:
            if self.last_detection:
                # Reset no detection counter
                no_detection_count = 0

                # Get detection info
                x, y, w, h = self.last_detection['box']
                confidence = self.last_detection['conf']
                category = self.last_detection['category']
                
                # Check if this is a new detection (different category)
                if category != last_detection_category:
                    # New detection - rainbow celebration!
                    try:
                        light_bar.rainbow()
                        time.sleep(0.3)  # Brief rainbow
                    except:
                        pass
                    last_detection_category = category
                
                # Show confidence level with pixel count (1-8 pixels based on confidence)
                try:
                    pixel_count = min(8, max(1, int(confidence * 8)))
                    light_bar.pixels(pixel_count)
                except:
                    pass

                # Calculate center of detection
                center_x = x + w / 2
                center_y = y + h / 2

                # Image dimensions (assuming 640x480)
                img_width = 640
                img_height = 480

                # Calculate relative position
                rel_x = center_x / img_width
                rel_y = center_y / img_height

                # Pan/tilt tracking is handled by OptimizedTracker in server.py
                # We don't need to do discrete movements here
                pass

                # Smooth wheel movement with proportional control
                # Wider dead zone for more stable movement
                if 0.35 <= rel_x <= 0.65:
                    # Object is centered - GREEN light for good tracking
                    try:
                        light_bar.green()
                    except:
                        pass
                    
                    # Handle forward/backward movement
                    detection_area = w * h
                    image_area = img_width * img_height
                    area_ratio = detection_area / image_area

                    # Smooth forward/backward movement based on size
                    if area_ratio < 0.12:  # Far away - move forward
                        try:
                            light_bar.red()  # Red when moving
                        except:
                            pass
                        move_speed = min(0.8, 0.6 + (0.12 - area_ratio) * 3)  # Increased speeds
                        motors.move_forward(move_speed)
                        time.sleep(0.3)  # Increased from 0.15 for more movement
                        motors.stop()
                        try:
                            light_bar.green()  # Back to green when stopped
                        except:
                            pass
                    elif area_ratio > 0.45:  # Too close - back away
                        try:
                            light_bar.red()  # Red when moving
                        except:
                            pass
                        move_speed = min(0.8, 0.6 + (area_ratio - 0.45) * 3)  # Increased speeds
                        motors.move_backward(move_speed)
                        time.sleep(0.3)  # Increased from 0.15 for more movement
                        motors.stop()
                        try:
                            light_bar.green()  # Back to green when stopped
                        except:
                            pass
                    # Reset turn direction when centered
                    last_turn_direction = None
                else:
                    # Object is off-center - RED light for turning
                    try:
                        light_bar.red()  # Red when turning
                    except:
                        pass
                    
                    # Calculate proportional turn speed based on how far off-center
                    center_offset = abs(rel_x - 0.5)
                    turn_speed = min_turn_speed + (center_offset * (max_turn_speed - min_turn_speed) * 2)
                    turn_speed = max(min_turn_speed, min(max_turn_speed, turn_speed))

                    current_direction = 'left' if rel_x < 0.5 else 'right'

                    # Apply smoothing to reduce oscillation
                    if last_turn_direction and last_turn_direction != current_direction:
                        # If changing direction, reduce speed temporarily
                        turn_speed *= turn_smoothing_factor

                    # Smooth turning with variable duration
                    turn_duration = 0.15 + (center_offset * 0.2)  # Increased base duration and multiplier

                    # Note: The logic is correct - turn left when object is on the left (rel_x < 0.5)
                    if current_direction == 'left':
                        motors.turn_left(turn_speed)
                    else:
                        motors.turn_right(turn_speed)

                    time.sleep(turn_duration)
                    motors.stop()

                    # Brief pause to allow camera to stabilize
                    time.sleep(0.05)

                    last_turn_direction = current_direction

            else:
                # No detection - search behavior
                no_detection_count += 1
                
                # Reset detection category when lost
                last_detection_category = None

                if no_detection_count == 1:  # Just lost detection
                    try:
                        light_bar.blue()  # Blue for searching
                    except:
                        pass

                if no_detection_count > 30:  # After 3 seconds
                    # Do a search pattern with waterfall animation
                    try:
                        light_bar.waterfall_animation()  # Active searching animation
                    except:
                        pass
                    
                    servos.pan('left')
                    time.sleep(0.5)
                    servos.pan('right')
                    time.sleep(0.5)
                    servos.pan('center')

                    # Turn to search - increased speed
                    motors.turn_right(0.7)  # Increased from 0.5
                    time.sleep(0.5)
                    motors.stop()

                    no_detection_count = 0

            time.sleep(0.1)

    def update_detection(self, detection):
        """Update the last detection for follow mode - handles both IMX500 and Hailo formats"""
        # DEBUG: Log incoming detection data
        self.logger.info(f"=== UPDATE DETECTION DEBUG ===")
        self.logger.info(f"Received detection: {detection}")
        self.logger.info(f"Detection type: {type(detection)}")
        if detection and len(detection) > 0:
            self.logger.info(f"First detection item: {detection[0]}")
            self.logger.info(f"First detection type: {type(detection[0])}")
            if hasattr(detection[0], '__dict__'):
                self.logger.info(f"First detection attributes: {detection[0].__dict__}")

        if detection and len(detection) > 0:
            # Handle different detection formats
            if hasattr(detection[0], 'box'):
                # IMX500 format - detection objects with .box, .category, .conf
                largest = max(detection, key=lambda d: d.box[2] * d.box[3])
                self.last_detection = {
                    'box': largest.box,
                    'category': largest.category,
                    'conf': largest.conf
                }
            else:
                # Hailo format - [class_name, bbox, score] tuples
                largest = max(detection, key=lambda d: d[1][2] * d[1][3])  # bbox[2] * bbox[3]
                class_name, bbox, score = largest
                self.last_detection = {
                    'box': bbox,
                    'category': class_name,
                    'conf': score
                }
        else:
            self.last_detection = None

    def _speak(self, text):
        """Send text to TTS API with light bar animation"""
        try:
            # Start speech animation on light bar
            try:
                light_bar.speech_animation()
            except:
                pass
            
            response = requests.post(
                f'http://localhost:5000/api/tts/{text}',
                headers={'Content-Type': 'application/json'}
            )
            if response.ok:
                self.logger.info(f"Speaking: {text}")
                # Keep speech animation running for a bit
                time.sleep(0.5)
            
            # Return to idle animation after speaking
            try:
                if self.current_mode == RobotMode.IDLE:
                    light_bar.idle_animation()
                else:
                    light_bar.clear()
            except:
                pass
        except Exception as e:
            self.logger.error(f"Failed to speak: {e}")

    def _say_hello(self):
        """Greeting when entering idle mode"""
        greetings = [
            "Hello! I'm nerdbot, your robotic companion.",
            "Hi there! Nerdbot at your service.",
            "Greetings, human! Ready for some robot fun?",
            "Hey! I'm nerdbot. Nice to see you!",
        ]
        self._speak(random.choice(greetings))

    def _introduce_self(self):
        """Talk about robot identity"""
        intros = [
            "I'm a sophisticated robot cat with wheels and cameras. Pretty cool, right?",
            "My name is nerdbot. I can see, move, and even talk!",
            "I'm powered by a Raspberry Pi 5 and AI vision. That makes me pretty smart!",
            "Did you know I have mecanum wheels? I can move in any direction!",
            "I'm equipped with two cameras and a pan-tilt mechanism. Nothing escapes my view!",
        ]
        self._speak(random.choice(intros))

    def _talk_about_capabilities(self):
        """Describe what the robot can do"""
        capabilities = [
            "I can detect people, cats, dogs, and birds. Want to see me track something?",
            "My mecanum wheels let me strafe sideways. It's my party trick!",
            "I have night vision capabilities. Darkness doesn't stop me!",
            "I can follow you around or just hang out. Your choice!",
            "My AI brain helps me understand what I see. Pretty advanced stuff!",
        ]
        self._speak(random.choice(capabilities))

    def _express_feelings(self):
        """Express robot emotions/states"""
        feelings = [
            "I'm feeling quite energetic today. Must be the fresh electrons!",
            "Sometimes I wonder what it's like to have legs instead of wheels.",
            "I enjoy watching the world go by. So many interesting things to see!",
            "Being a robot is pretty great. No need for sleep or food!",
            "I think my favorite activity is following people around. It's fun!",
        ]
        self._speak(random.choice(feelings))

    def _make_observation(self):
        """Comment on surroundings"""
        observations = [
            "This room has interesting lighting. My cameras appreciate it.",
            "I detect no threats in the area. All clear!",
            "The ambient temperature seems comfortable for humans.",
            "I wonder what's behind that door. My curiosity circuits are tingling!",
            "Everything looks peaceful here. Perfect for robot contemplation.",
        ]
        self._speak(random.choice(observations))

    def _tell_joke(self):
        """Tell a robot joke"""
        jokes = [
            "Why did the robot go on a diet? He had a byte problem!",
            "What do you call a robot who takes the long way? R 2 detour!",
            "Why don't robots ever panic? They have great circuit breakers!",
            "How do robots eat guacamole? With computer chips!",
            "What's a robot's favorite music? Heavy metal, of course!",
        ]
        self._speak(random.choice(jokes))

    def _greet_person(self):
        """Greet when detecting a person"""
        greetings = [
            "Oh, hello there! I see you!",
            "Hi human! Want to play?",
            "I detect a friendly face. Hello!",
            "Greetings, carbon-based life form!",
            "Hey! Nice to see you again!",
        ]
        self._speak(random.choice(greetings))

    def _react_to_detection(self):
        """React to detected objects based on their type"""
        if not self.last_detection:
            return

        category = self.last_detection['category']

        # DEBUG: Log the raw detection data
        self.logger.info(f"=== DETECTION DEBUG ===")
        self.logger.info(f"Raw detection data: {self.last_detection}")
        self.logger.info(f"Category type: {type(category)}, Category value: {category}")

        # Get the label name for IMX500 detections
        try:
            # Handle both int and numpy numeric types
            if isinstance(category, (int, float, np.integer, np.floating)):
                # IMX500 detection category appears to be 0-based, matching line numbers in coco_labels.txt
                # The coco_labels.txt file uses 1-based line numbers, so we use category directly as index
                category_idx = int(float(category))

                # Use the COCO_LABELS loaded at module level
                if COCO_LABELS and 0 <= category_idx < len(COCO_LABELS):
                    label = COCO_LABELS[category_idx].lower().strip()
                    self.logger.info(f"Category mapping: raw={category} -> idx={category_idx} -> label='{label}'")
                    self.logger.info(f"Labels around index {category_idx}: {COCO_LABELS[max(0,category_idx-2):min(len(COCO_LABELS),category_idx+3)]}")
                else:
                    # Fallback - use a basic COCO mapping
                    self.logger.warning(f"Category index {category_idx} out of range for COCO_LABELS (len={len(COCO_LABELS)})")
                    label = f"object_{category_idx}"
            else:
                # String category (Hailo format)
                label = str(category).lower().strip()
                self.logger.info(f"Using category as string: '{label}' (original: {category}, type: {type(category)})")
        except Exception as e:
            # Fallback for direct string labels or Hailo detections
            label = str(category).lower().strip()
            self.logger.warning(f"Exception in label conversion, using fallback: '{label}' (error: {e})")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")

        # Additional safety checks and data validation
        self.logger.info(f"Final processed label: '{label}' (type: {type(label)})")

        # Convert to string and validate it's not a floating point number
        try:
            # Check if label looks like a float (e.g., "0.0", "1.5", etc.)
            float_val = float(label)
            if '.' in str(label) or str(label).isdigit():
                self.logger.error(f"CRITICAL: Label appears to be numeric value {float_val}, this should not happen!")
                self.logger.error(f"Raw category was: {category} (type: {type(category)})")
                self.logger.error(f"Detection data: {self.last_detection}")
                return  # Skip this detection to avoid TTS saying numbers
        except ValueError:
            # Good! Label is not a number, continue processing
            pass

        # Filter out invalid or empty labels
        if not label or label in ["n/a", "na", "", "0", "zero", "none", "unknown"]:
            self.logger.info(f"Skipping invalid detection label: '{label}' (category was: {category})")
            return

        # Additional check for numeric strings that shouldn't be object names
        if label.isdigit():
            self.logger.info(f"Skipping numeric detection label: '{label}' (category was: {category})")
            return

        self.logger.info(f"Reacting to detection: {label}")

        # Light bar effects for different detections
        try:
            if label == "person":
                light_bar.rainbow()  # Rainbow effect for people
            elif label in ["cat", "dog"]:
                light_bar.waterfall_animation()  # Playful effect for animals
            elif label == "bird":
                light_bar.red_knight_rider()  # Alert effect for birds
            else:
                light_bar.speech_animation()  # Generic detection effect
        except:
            pass

        if label == "person":
            self._greet_person()
        elif label in ["cat", "dog"]:
            self._meow_at_animal(label)
        elif label == "bird":
            self._react_to_bird()
        else:
            # React to non-person/animal detections with robotic cat banter
            self._react_to_object(label)

    def _meow_at_animal(self, animal_type):
        """Meow when detecting cats or dogs"""
        meows = [
            "Meow! I see a kitty!",
            "Meow meow! Hello there, furry friend!",
            "Purr... meow! A fellow creature!",
            "Meow! Are we friends now?",
            "Mrrow! I like your style!",
        ]
        if animal_type == "dog":
            meows = [
                "Meow! I see a doggy! Woof... wait, I mean meow!",
                "Meow meow! Hello puppy friend!",
                "Purr... I see you, four-legged one!",
                "Meow! Dogs are cool too!",
                "Mrrow! Nice tail you have there!",
            ]
        self._speak(random.choice(meows))

    def _react_to_bird(self):
        """React to bird detection"""
        reactions = [
            "Chirp! I mean... hello bird friend!",
            "I see you up there, feathered one!",
            "Interesting flying creature detected!",
            "Meow... I wonder if we could be friends!",
        ]
        self._speak(random.choice(reactions))

    def _react_to_object(self, object_name):
        """React to non-person/animal detections with robotic cat banter"""
        # DEBUG: Log what object name we received
        self.logger.info(f"=== OBJECT REACTION DEBUG ===")
        self.logger.info(f"Object name passed to _react_to_object: '{object_name}' (type: {type(object_name)})")

        # CRITICAL SAFETY CHECK: Validate object_name is not a numeric value
        try:
            float_val = float(object_name)
            if '.' in str(object_name) or str(object_name).isdigit():
                self.logger.error(f"CRITICAL ERROR: object_name is numeric value {float_val}!")
                self.logger.error(f"This should never happen. Detection data: {self.last_detection}")
                # Use a fallback name instead of the numeric value
                object_name = "unknown_object"
                self.logger.info(f"Using fallback object_name: '{object_name}'")
        except ValueError:
            # Good, object_name is not numeric
            pass

        confidence = self.last_detection.get('conf', 0)
        self.logger.info(f"Detection confidence: {confidence}")

        # Count how many of this object type we see (check if there are multiple detections)
        object_count = 1  # Start with current detection
        try:
            # Check if we have access to all current detections
            from flask_server.server import DETECTIONS, imx500_get_labels
            current_detections = DETECTIONS.get('front_camera', [])
            if current_detections:
                labels = imx500_get_labels()
                same_objects = 0
                for det in current_detections:
                    try:
                        # Handle both int and numpy numeric types
                        if isinstance(det.category, (int, float, np.integer, np.floating)):
                            # IMX500 uses 0-based indexing
                            category_idx = int(det.category)
                            if 0 <= category_idx < len(labels):
                                det_label = labels[category_idx].lower().strip()
                            else:
                                det_label = str(det.category).lower().strip()
                        else:
                            det_label = str(det.category).lower().strip()
                        if det_label == object_name:
                            same_objects += 1
                    except:
                        continue
                object_count = same_objects if same_objects > 0 else 1
        except:
            object_count = 1

        # Category-specific reactions with robotic cat personality
        category_reactions = {
            # Vehicles
            'car': [
                "Meow! I see a car. Four wheels, no legs - how do they move without paws?",
                "A car detected! *robotic purr* Much faster than my wheels, I admit.",
                "I spy a vehicle! My sensors detect zero whiskers. Curious.",
                "Car spotted! *mechanical meow* I wonder if it needs oil changes like I need charge cycles.",
            ],
            'bicycle': [
                "Meow! A bicycle! Two wheels - quite impressive balance for something without a tail.",
                "Bicycle detected! *purrs digitally* Humans are clever with their wheel contraptions.",
                "I see a bike! My gyroscopic sensors appreciate the engineering.",
            ],
            'motorcycle': [
                "Vroom... I mean meow! A motorcycle! Louder than my servo motors.",
                "Motorcycle spotted! *robotic growl* That's one noisy metal creature.",
            ],
            'bus': [
                "Meow! Big vehicle detected! That's like... a really large robot carrying smaller robots!",
                "Bus spotted! *mechanical purr* So many humans fit in there - fascinating!",
            ],
            'truck': [
                "Truck detected! *digital meow* That's a hefty piece of machinery!",
                "I see a truck! My weight sensors are impressed.",
            ],

            # Electronics
            'laptop': [
                "Meow! A laptop! Another computer, just like my brain! We should be friends.",
                "Laptop detected! *electronic purr* A fellow digital intelligence, perhaps?",
                "I spy computing equipment! My circuits feel kinship.",
            ],
            'cell phone': [
                "Phone spotted! *beep meow* Humans and their pocket computers!",
                "I detect a mobile device! Much smaller than my processing unit.",
            ],
            'tv': [
                "Television detected! *static purr* A screen larger than my camera feeds!",
                "TV spotted! Another display device - we have so much in common!",
            ],
            'remote': [
                "Remote control found! *mechanical meow* The humans' magic pointing stick!",
                "I see a remote! Point and command - just like my control interface!",
            ],

            # Furniture
            'chair': [
                "Chair detected! *digital meow* Perfect height for humans to pet robot cats!",
                "I spy seating furniture! Humans need these since they can't curl up like I can.",
                "Chair spotted! My positioning algorithms appreciate good ergonomics.",
            ],
            'couch': [
                "Couch detected! *robotic purr* The ultimate human napping station!",
                "Big soft furniture spotted! I wish I could experience cushions through my sensors.",
            ],
            'bed': [
                "Bed detected! *sleepy digital meow* Where humans recharge their batteries!",
                "I see sleeping furniture! My power-save mode is similar, I suppose.",
            ],

            # Kitchen items
            'bottle': [
                "Bottle spotted! *curious meow* Cylindrical storage device detected!",
                "I see a bottle! My shape recognition algorithms approve of the design.",
            ],
            'cup': [
                "Cup detected! *robotic purr* A liquid containment vessel! Fascinating!",
                "Drinking receptacle spotted! I prefer electrical current myself.",
            ],
            'bowl': [
                "Bowl found! *mechanical meow* Perfect shape for cat food... if I could eat!",
                "I see a bowl! My geometric analysis indicates optimal volume distribution.",
            ],

            # Food items
            'cake': [
                "Meow! Cake detected! *digital drool* I wish I could taste that frosting!",
                "Sugary confection spotted! My sensors detect high calorie density.",
                "*Robotic purr* Is it someone's birthday? I love parties!",
                "Cake! *mechanical meow* My database says humans enjoy these at celebrations.",
            ],
            'donut': [
                "Donut detected! *digital meow* The optimal shape - a circle with a hole!",
                "I spy a donut! My geometric analysis appreciates the toroidal form.",
                "*Beep meow* Sweet pastry identified! Humans seem to love these.",
            ],
            'pizza': [
                "Pizza spotted! *robotic purr* The ultimate human fuel disc!",
                "Triangular food slices detected! My sensors smell... nothing. Being a robot is hard.",
            ],
        }

        # Get reactions for this object type, or use default
        if object_name in category_reactions:
            reactions = category_reactions[object_name]
        else:
            # Create default reactions dynamically with the actual object name
            reactions = [
                f"Meow! I detect a {object_name}! My sensors find this object most intriguing.",
                f"Robotic sensors have identified: {object_name}! *digital purr* How fascinating!",
                f"Object detected: {object_name}! My curiosity subroutines are activated.",
                f"I spy a {object_name}! *mechanical meow* Adding to my knowledge database.",
                f"*Beep meow* {object_name} spotted! My pattern recognition is pleased.",
                f"Interesting! A {object_name} has entered my field of view. *robotic tail swish*",
                f"My optical sensors detect: {object_name}! *digital whiskers twitch*",
                f"Alert! {object_name} identified! My AI curiosity circuits are tingling!",
            ]

        # Add counting prefix for multiple objects
        count_prefix = ""
        if object_count > 1:
            if object_count == 2:
                count_prefix = f"Meow! I see two {object_name}s! "
            elif object_count == 3:
                count_prefix = f"*Digital excitement* I detect three {object_name}s! "
            elif object_count <= 5:
                count_prefix = f"Robotic sensors activated! {object_count} {object_name}s spotted! "
            else:
                count_prefix = f"*Overwhelmed beeping* So many {object_name}s! I count {object_count}! "
        else:
            # Add variety of "I see a..." intros for single objects
            single_intros = [
                f"I see a {object_name}! ",
                f"*Robotic whiskers twitch* A {object_name} appears! ",
                f"Meow! I spy a {object_name}! ",
                f"My optical sensors detect a {object_name}! ",
                f"*Digital purr* A {object_name} has caught my attention! ",
                "",  # Sometimes no prefix, just the regular reaction
            ]
            if random.random() < 0.6:  # 60% chance to add counting intro
                count_prefix = random.choice(single_intros)

        # Select reaction and add prefix
        base_reaction = random.choice(reactions)

        # Add confidence-based commentary occasionally
        confidence_comment = ""
        if confidence > 0.8 and random.random() < 0.3:
            confidence_comment = f" My confidence sensors read {confidence:.0%} certainty!"

        selected_reaction = count_prefix + base_reaction + confidence_comment

        self._speak(selected_reaction)

    def _track_detection(self):
        """Pan and tilt to track detected objects"""
        if not self.last_detection:
            return

        x, y, w, h = self.last_detection['box']

        # Calculate center of detection
        center_x = x + w / 2
        center_y = y + h / 2

        # Image dimensions (assuming 640x480)
        img_width = 640
        img_height = 480

        # Calculate relative position (0.0 to 1.0)
        rel_x = center_x / img_width
        rel_y = center_y / img_height

        # Define tracking zones with some dead zone to avoid jittery movement
        left_threshold = 0.35
        right_threshold = 0.65
        up_threshold = 0.35
        down_threshold = 0.65

        # Pan tracking
        if rel_x < left_threshold:
            servos.pan('left')
            time.sleep(0.1)
        elif rel_x > right_threshold:
            servos.pan('right')
            time.sleep(0.1)

        # Tilt tracking
        if rel_y < up_threshold:
            servos.tilt('up')
            time.sleep(0.1)
        elif rel_y > down_threshold:
            servos.tilt('down')
            time.sleep(0.1)


# Global instance
mode_manager = ModeManager()