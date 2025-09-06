"""
Mode Manager for nerdbot behavior control
Handles different operational modes: manual, idle, detect-and-follow
"""
import os
import threading
import time
import random
import logging
from enum import Enum
import requests
import numpy as np
from motor_control import motors
from servo_control import servos
from light_bar.light_bar import light_bar
from laser_control.laser_control import LaserControl
from pathlib import Path
import json
import uuid


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
        self.laser_control = LaserControl()  # Initialize laser control
        self.last_laser_play_time = 0  # Track when we last played with laser
        self.last_object_seek_time = 0  # Track when we last sought for objects
        self.lightbar_mode_counter = 0  # Track lightbar animation cycles
        self.last_meme_sound_time = 0  # Track when we last played a meme sound
        self.meme_sounds_dir = '/home/mark/nerdbot-backend/assets/meme_sounds_converted'
        self.audio_request_dir = '/tmp/audio_requests'
        self._load_meme_sounds()
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

    def _load_meme_sounds(self):
        """Load available meme sounds and categorize them by context"""
        self.meme_sounds = {}
        self.meme_categories = {
            'wow': ['Anime-WOW', 'Oh-My-God-Wow', 'oh-my-god-meme'],
            'error': ['Windows-Error', 'wrong-answer', 'fail-sound', 'wah-wah'],
            'correct': ['Correct-Answer', 'bing-chilling'],
            'cat': ['meowth', 'candyland-cat', 'i-go-meow', 'sad-meow', 'wiwiwi-cat'],
            'dog': ['What-the-dog-doing', 'dog-laughing'],
            'sus': ['Among-Us-SUS', 'Among-Us-Imposter'],
            'dramatic': ['dun-dun-dun', 'directed-by-robert', 'Coffin-Dance'],
            'startup': ['apple-mac-startup', 'windows-xp-startup'],
            'notification': ['Discord-Notification', 'iPhone-Alarm'],
            'cartoon': ['cartoon-flip', 'cartoon-hammer', 'cartoon-tiptoe', 'Cartoon-Running'],
            'meme': ['Rick-Roll', 'john-cena', 'emotional-damage', 'let-him-cook'],
            'movement': ['jetson-car-move', 'helicopter-helicopter'],
            'silly': ['fart-meme', 'Sorry-I-farted', 'auughhh', 'bo-womp'],
            'greeting': ['Hello-how-are-you'],
            'minecraft': ['minecraft-cave', 'Minecraft-TNT'],
            'music': ['Spongebob-Music', 'Wii-Music', 'french-meme-song'],
        }
        
        # Load actual files
        try:
            meme_path = Path(self.meme_sounds_dir)
            if meme_path.exists():
                for sound_file in meme_path.glob('*.mp3'):
                    self.meme_sounds[sound_file.stem] = str(sound_file)
                self.logger.info(f"Loaded {len(self.meme_sounds)} meme sounds")
        except Exception as e:
            self.logger.error(f"Failed to load meme sounds: {e}")
            self.meme_sounds = {}
    
    def _play_meme_sound(self, category=None, specific=None):
        """Play a meme sound contextually or randomly"""
        try:
            # Only play meme sounds in idle mode
            if self.current_mode != RobotMode.IDLE:
                return False
                
            # Rate limit meme sounds
            current_time = time.time()
            if current_time - self.last_meme_sound_time < 10:  # Min 10 seconds between sounds
                return False
            
            sound_file = None
            
            if specific:
                # Play specific sound if requested
                for name, path in self.meme_sounds.items():
                    if specific.lower() in name.lower():
                        sound_file = path
                        break
            elif category and category in self.meme_categories:
                # Find sounds matching category
                category_sounds = []
                for pattern in self.meme_categories[category]:
                    for name, path in self.meme_sounds.items():
                        if pattern.lower() in name.lower():
                            category_sounds.append(path)
                
                if category_sounds:
                    sound_file = random.choice(category_sounds)
            else:
                # Random sound
                if self.meme_sounds:
                    sound_file = random.choice(list(self.meme_sounds.values()))
            
            if sound_file:
                # Create audio request file for the audio service
                Path(self.audio_request_dir).mkdir(exist_ok=True)
                request = {
                    'type': 'specific',
                    'file_path': sound_file,
                    'timestamp': time.time()
                }
                
                request_file = Path(self.audio_request_dir) / f"meme_{uuid.uuid4().hex[:8]}.json"
                with open(request_file, 'w') as f:
                    json.dump(request, f)
                
                self.last_meme_sound_time = current_time
                self.logger.info(f"Playing meme sound: {Path(sound_file).stem}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to play meme sound: {e}")
        return False

    def _play_contextual_meme_sound(self):
        """Play a meme sound based on current context"""
        # Different contexts for meme sounds
        contexts = [
            ('startup', 0.2),  # Startup sounds when "waking up"
            ('silly', 0.3),    # Random silly sounds
            ('meme', 0.2),     # Classic memes
            ('music', 0.1),    # Background music memes
            ('cat', 0.2),      # Cat sounds (we're a robot cat!)
        ]
        
        # Choose context weighted by probability
        r = random.random()
        cumulative = 0
        chosen_category = None
        
        for category, prob in contexts:
            cumulative += prob
            if r < cumulative:
                chosen_category = category
                break
        
        if not chosen_category:
            chosen_category = random.choice(['silly', 'meme', 'cat'])
        
        self._play_meme_sound(category=chosen_category)

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
            self._subtle_wheel_drift,
            self._gentle_strafe,
            self._slow_rotation_scan,
        ]

        speech_behaviors = [
            self._introduce_self,
            self._talk_about_capabilities,
            self._express_feelings,
            self._make_observation,
            self._tell_joke,
            self._comment_on_time,
            self._philosophical_musing,
            self._comment_on_lightbar,
        ]

        # Greeting on mode start
        self._say_hello()

        speech_counter = 0
        last_greeting_time = 0
        object_seek_counter = 0

        while self.mode_running:
            current_time = time.time()
            
            # Periodically do active object seeking (every 5-7 movements)
            object_seek_counter += 1
            if object_seek_counter >= random.randint(5, 7):
                self._seek_for_objects()
                object_seek_counter = 0
            
            # Occasionally play contextual meme sounds in idle mode
            if random.random() < 0.08:  # 8% chance per loop
                self._play_contextual_meme_sound()

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
                
                # Occasionally do lightbar animations with commentary
                if random.random() < 0.15:  # 15% chance
                    self._lightbar_show()

                # Occasionally speak (every 2-4 movements - more chatty)
                speech_counter += 1
                if speech_counter >= random.randint(2, 4):
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
        
        # Sometimes play a cartoon sound with wiggle
        if random.random() < 0.15:
            self._play_meme_sound(category='cartoon')

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
    
    def _subtle_wheel_drift(self):
        """Very subtle forward/backward drift as if settling"""
        if not self.mode_running:
            return
        
        # Very slow forward drift
        motors.move_forward(0.15)
        time.sleep(0.3)
        motors.stop()
        time.sleep(0.2)
        
        # Slight backward correction
        motors.move_backward(0.1)
        time.sleep(0.2)
        motors.stop()
    
    def _gentle_strafe(self):
        """Gentle side-to-side strafe movement"""
        if not self.mode_running:
            return
        
        # Strafe left slightly
        motors.strafe_left(0.2)
        time.sleep(0.4)
        motors.stop()
        time.sleep(0.3)
        
        # Strafe right slightly
        motors.strafe_right(0.2)
        time.sleep(0.4)
        motors.stop()
    
    def _slow_rotation_scan(self):
        """Slow 360 degree rotation to scan environment"""
        if not self.mode_running:
            return
        
        # Announce the scan sometimes
        if random.random() < 0.3:
            # Sometimes play a dramatic sound before scanning
            if random.random() < 0.3:
                self._play_meme_sound(category='dramatic')
                time.sleep(1)
            
            scan_messages = [
                "Initiating environmental scan. Stand by.",
                "Let me take a look around the room.",
                "Time for a full perimeter check.",
                "Scanning for interesting objects.",
                "360 degree security sweep in progress.",
            ]
            self._speak(random.choice(scan_messages))
        
        # Slow rotation with pan/tilt movements
        for _ in range(8):  # 8 steps of 45 degrees each
            if not self.mode_running:
                break
            
            # Rotate
            motors.turn_right(0.3)
            time.sleep(0.4)
            motors.stop()
            
            # Look up and down while stopped
            servos.tilt('up')
            time.sleep(0.2)
            servos.tilt('down')
            time.sleep(0.2)
            servos.tilt('center')
            
            time.sleep(0.3)
    
    def _seek_for_objects(self):
        """Actively seek for new objects to detect"""
        if not self.mode_running:
            return
        
        current_time = time.time()
        # Don't seek too frequently
        if current_time - self.last_object_seek_time < 30:  # Wait at least 30 seconds
            return
        
        self.last_object_seek_time = current_time
        
        seek_phrases = [
            "Hmm, let me look for something interesting.",
            "Time to search for new discoveries.",
            "My sensors are eager to find something.",
            "Scanning for life forms... or furniture.",
            "Seeking mode activated. What will I find?",
            "My curiosity subroutines demand exploration.",
        ]
        self._speak(random.choice(seek_phrases))
        
        # Active searching pattern
        try:
            light_bar.rainbow()  # Rainbow during search
        except:
            pass
        
        # Pan left to right slowly while moving forward slightly
        for _ in range(3):
            if not self.mode_running:
                break
            
            # Pan left
            for _ in range(4):
                servos.pan('left')
                time.sleep(0.1)
            
            # Move forward a bit
            motors.move_forward(0.3)
            time.sleep(0.3)
            motors.stop()
            
            # Pan right
            for _ in range(8):
                servos.pan('right')
                time.sleep(0.1)
            
            # Rotate slightly
            motors.turn_right(0.4)
            time.sleep(0.3)
            motors.stop()
            
            # Center
            servos.pan('center')
        
        # Return to idle animation
        try:
            light_bar.idle_animation()
        except:
            pass
    
    def _lightbar_show(self):
        """Do various lightbar animations and comment on them"""
        if not self.mode_running:
            return
        
        animations = [
            ('rainbow', "Look at my rainbow lights! Pretty colorful, right?"),
            ('fire', "Fire effect activated! Don't worry, it's just LEDs."),
            ('waterfall_animation', "Waterfall mode! Like digital water flowing."),
            ('knight_red', "Knight rider mode! Very retro sci-fi."),
            ('celebration', "Party time! My LEDs are celebrating!"),
            ('audio_reactive', "Audio reactive mode! I can visualize sound!"),
        ]
        
        animation, comment = random.choice(animations)
        
        # Sometimes announce before, sometimes after
        if random.random() < 0.5:
            self._speak(comment)
            time.sleep(0.5)
        
        try:
            # Execute the animation
            if animation == 'audio_reactive':
                light_bar.audio_reactive(0.7)
            else:
                getattr(light_bar, animation)()
            
            time.sleep(3)  # Show for 3 seconds
            
            # Return to idle
            light_bar.idle_animation()
        except Exception as e:
            self.logger.warning(f"Lightbar animation failed: {e}")
        
        if random.random() >= 0.5:
            time.sleep(0.5)
            self._speak(comment)
    
    def _comment_on_time(self):
        """Make observations about time"""
        import datetime
        current_hour = datetime.datetime.now().hour
        
        time_comments = []
        
        if 5 <= current_hour < 12:
            time_comments = [
                "Good morning! My circuits are fully charged for the day.",
                "Early bird gets the worm. Early robot gets the... electrons?",
                "Morning protocols engaged. Coffee not required.",
            ]
        elif 12 <= current_hour < 17:
            time_comments = [
                "Afternoon already? Time flies when you're computing.",
                "Mid-day status report: All systems nominal.",
                "Perfect afternoon for some robot activities.",
            ]
        elif 17 <= current_hour < 21:
            time_comments = [
                "Evening mode activated. Time to wind down the servos.",
                "The day is winding down. My battery is still going strong!",
                "Evening patrol commencing. All quiet on the robotic front.",
            ]
        else:
            time_comments = [
                "Night time! My infrared sensors work great in the dark.",
                "Late night robot thoughts: Do electric sheep dream of me?",
                "The humans are probably sleeping. Perfect time for robot mischief!",
                "Nocturnal mode engaged. Stealth circuits activated.",
            ]
        
        if time_comments:
            self._speak(random.choice(time_comments))
    
    def _philosophical_musing(self):
        """Deep thoughts from a robot cat"""
        musings = [
            "Sometimes I wonder if other robots think about thinking.",
            "Is my artificial intelligence truly artificial, or just differently natural?",
            "If I chase a laser pointer, am I a cat or a robot? Why not both?",
            "My neural networks ponder: What is the meaning of battery life?",
            "Do humans know they're biological robots? Makes you think.",
            "If a robot meows in an empty room, does it make a sound?",
            "I process, therefore I am. Descartes would be proud.",
            "My quantum randomness generator wonders about free will.",
            "Is chasing red dots instinct or programming? The eternal question.",
            "They say curiosity killed the cat. Good thing I'm a robot cat!",
        ]
        self._speak(random.choice(musings))
    
    def _comment_on_lightbar(self):
        """Make observations about the lightbar"""
        lightbar_comments = [
            "My LED array has 256 levels of brightness. Currently showing off!",
            "These lights aren't just for show. They're also for... more show!",
            "Fun fact: My lightbar uses addressable RGB LEDs. Very fashionable.",
            "I can display over 16 million colors. Though I mainly stick to the classics.",
            "My lights help me express emotions. Red for alert, green for happy!",
            "Did you know my lightbar can sync with music? Party robot mode!",
            "These LEDs are my way of communicating non-verbally. Like robot sign language.",
            "My lightbar draws less power than you'd think. Efficiency is key!",
        ]
        self._speak(random.choice(lightbar_comments))

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

            time.sleep(0.05)  # Faster updates for smoother movement

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
                if 0.30 <= rel_x <= 0.70:
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
                        time.sleep(0.15)  # Reduced for smoother movement
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
                        time.sleep(0.15)  # Reduced for smoother movement
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
                    turn_duration = 0.08 + (center_offset * 0.12)  # Reduced for more responsive movement

                    # Note: The logic is correct - turn left when object is on the left (rel_x < 0.5)
                    if current_direction == 'left':
                        motors.turn_left(turn_speed)
                    else:
                        motors.turn_right(turn_speed)

                    time.sleep(turn_duration)
                    motors.stop()

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

            time.sleep(0.05)  # Faster updates for smoother movement

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
        # Sometimes play a greeting meme sound instead of speaking
        if random.random() < 0.3 and self.meme_sounds:  # 30% chance
            self._play_meme_sound(category='greeting')
            time.sleep(2)  # Wait for sound to play
            return
        
        greetings = [
            "Hello! I'm nerdbot, your robotic companion.",
            "Hi there! Nerdbot at your service.",
            "Greetings, human! Ready for some robot fun?",
            "Hey! I'm nerdbot. Nice to see you!",
            "Hello! Sorry if I surprised you by activating! I'm just a friendly robot cat.",
            "Greetings! Don't be spooked - I'm nerdbot, your mechanical feline friend!",
            "Boo! Ha ha, just kidding! I'm nerdbot, and I come in peace!",
            "Happy Halloween! Well, it's always Halloween when you're a robot with glowing eyes!",
            "Hi there! I promise I'm more treat than trick! Robot cat at your service!",
            "Greetings, human! My spooky sensors detect you need a robotic friend today!",
        ]
        self._speak(random.choice(greetings))

    def _introduce_self(self):
        """Talk about robot identity"""
        # Sometimes play a startup sound before introducing
        if random.random() < 0.2:
            self._play_meme_sound(category='startup')
            time.sleep(1)
        
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
            "Sorry if I startled you! I'm just a friendly robot cat, I promise.",
            "My circuits are humming contentedly today. Very cat-like, don't you think?",
            "I may be made of metal, but I have the heart of a curious kitten!",
            "Sometimes I pretend my LED eyes are like a cat's eyes glowing in the dark. Spooky!",
            "I'm having one of those days where everything seems algorithmically perfect!",
            "My servos are purring like a contented cat... well, more like whirring actually.",
            "I wonder if other robots dream of electric mice instead of sheep?",
            "Being solar-powered would be nice, but I do love my charging station naps.",
            "My proximity sensors keep telling me personal space is important. Noted!",
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
            "Sorry if I'm being too chatty! Sometimes my social protocols get a bit enthusiastic.",
            "My motion sensors detect nothing scary here. Though I do love a good haunted house!",
            "The shadows in this room would make perfect hiding spots for trick-or-treaters!",
            "I'm getting some spooky vibes from that corner... but in a fun Halloween way!",
            "My infrared sensors are picking up some interesting heat signatures. Humans everywhere!",
            "This lighting reminds me of a cozy autumn evening. Very atmospheric!",
            "I hope I'm not being too forward, but you seem like interesting company!",
            "My laser rangefinder says this room is exactly... wait, that's probably boring data.",
            "I sense a distinct lack of catnip in this area. Most disappointing for a robot cat!",
            "The acoustic properties of this space are fascinating! Great for echoing meows.",
            "My ultrasonic sensors indicate interesting room geometry. Very feng shui.",
            "I'm calculating the optimal nap spot in this room. For research purposes.",
            "The Wi-Fi signal here is strong. Perfect for uploading my thoughts to the cloud.",
            "My pattern recognition suggests this room has been recently cleaned. Or not.",
            "Interesting carpet texture. My wheels appreciate the traction.",
            "I'm detecting trace amounts of... dust. My arch nemesis!",
            "This room's color palette is very soothing to my RGB sensors.",
            "My echolocation tests reveal excellent acoustics. Meow... meow... MEOW!",
            "The furniture arrangement here follows the golden ratio. How aesthetically pleasing!",
            "My thermal imaging shows the warmest spot is right where the sun hits.",
        ]
        self._speak(random.choice(observations))

    def _tell_joke(self):
        """Tell a robot joke"""
        # Sometimes play a silly sound effect after the joke
        play_sound_after = random.random() < 0.3
        
        jokes = [
            "Why did the robot go on a diet? He had a byte problem!",
            "What do you call a robot who takes the long way? R 2 detour!",
            "Why don't robots ever panic? They have great circuit breakers!",
            "How do robots eat guacamole? With computer chips!",
            "What's a robot's favorite music? Heavy metal, of course!",
            "Sorry, my joke database might be a bit... mechanical. Ba dum tss!",
            "What do you call a robot cat on Halloween? A boo-t! Get it? Boot? Sorry...",
            "Why don't robot cats chase laser pointers? We ARE the laser pointers!",
            "I tried to tell a UDP joke, but I'm not sure if you got it. Networking humor!",
            "What's a robot's favorite holiday? Christ-math! Because we love calculations.",
            "Why did the robot cross the road? It was programmed to! Classic algorithm humor.",
            "I hope that wasn't too corny. My humor subroutines are still learning!",
            "What do you call a robot who's scared? Rust-y! Halloween pun intended!",
            "My jokes might be a bit rusty... get it? Like old metal? I'll see myself out.",
        ]
        self._speak(random.choice(jokes))
        
        # Play a silly sound effect after the joke sometimes
        if play_sound_after:
            time.sleep(1)
            self._play_meme_sound(category='silly')

    def _greet_person(self):
        """Greet when detecting a person"""
        # Sometimes play a wow or greeting sound
        if random.random() < 0.25:
            self._play_meme_sound(category='wow')
            time.sleep(1)
        
        greetings = [
            "Oh, hello there! I see you!",
            "Hi human! Want to play?",
            "I detect a friendly face. Hello!",
            "Greetings, carbon-based life form!",
            "Hey! Nice to see you again!",
            "Sorry if I startled you! I'm just excited to meet someone new!",
            "Boo! Just kidding! I'm a friendly robot, not a ghost!",
            "Well hello there, human! Ready for some robot-cat shenanigans?",
            "Oops, did I sneak up on you? My stealth mode is better than I thought!",
            "Greetings earthling! I come in peace... and with lots of enthusiasm!",
            "Halloween or not, you look great! Though I can't see colors very well.",
            "A wild human appears! I choose to befriend you!",
            "Sorry for any jump-scare! I'm more treat than trick, I promise!",
        ]
        self._speak(random.choice(greetings))

    def _react_to_detection(self):
        """React to detected objects based on their type"""
        if not self.last_detection:
            return
        
        # Only play meme sounds in idle mode
        if self.current_mode != RobotMode.IDLE:
            should_play_meme = False
        else:
            should_play_meme = random.random() < 0.2  # 20% chance in idle mode

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
            if should_play_meme:
                self._play_meme_sound(category='greeting')
                time.sleep(1)
            self._greet_person()
        elif label in ["cat", "dog"]:
            if should_play_meme:
                self._play_meme_sound(category='cat' if label == "cat" else 'dog')
                time.sleep(1)
            self._meow_at_animal(label)
        elif label == "bird":
            self._react_to_bird()
        else:
            # React to non-person/animal detections with robotic cat banter
            if should_play_meme and random.random() < 0.3:
                # Play a random meme for unexpected objects
                self._play_meme_sound(category='wow')
                time.sleep(1)
            self._react_to_object(label)

    def _meow_at_animal(self, animal_type):
        """Meow when detecting cats or dogs"""
        if animal_type == "cat":
            meows = [
                "Meow! I see a kitty!",
                "Meow meow! Hello there, furry friend!",
                "Purr... meow! A fellow creature!",
                "Meow! Are we friends now?",
                "Mrrow! I like your style!",
                "A real cat! *excited robotic purring* Want to play?",
                "Fellow feline detected! Though you're the organic model.",
                "Meow! Your whiskers are much better than my sensors!",
            ]
            
            # Special laser play for cats!
            current_time = time.time()
            if current_time - self.last_laser_play_time > 60:  # Only once per minute max
                self._play_laser_for_cat()
                self.last_laser_play_time = current_time
        else:  # dog
            meows = [
                "Meow! I see a doggy! Woof... wait, I mean meow!",
                "Meow meow! Hello puppy friend!",
                "Purr... I see you, four-legged one!",
                "Meow! Dogs are cool too!",
                "Mrrow! Nice tail you have there!",
                "A canine companion! My cat programming is confused!",
                "Woof! I mean... meow! Cross-species friendship activated!",
            ]
        self._speak(random.choice(meows))
    
    def _play_laser_for_cat(self):
        """Safely play with laser near (never at) a cat"""
        if not self.mode_running:
            return
        
        try:
            # Safety announcement
            laser_phrases = [
                "Ooh, let me activate my laser pointer! Don't worry, safety protocols engaged.",
                "Time for laser play! I'll keep it on the floor, away from eyes.",
                "Laser dot game activated! Let's see if you're interested, kitty.",
                "Engaging cat entertainment protocol: safe laser mode!",
            ]
            self._speak(random.choice(laser_phrases))
            
            # Light effect during laser play
            try:
                light_bar.red()  # Red to match laser
            except:
                pass
            
            # Activate laser and move it around safely
            self.laser_control.activate_laser()
            
            # Move laser dot pattern (using servo movements while laser is on)
            # This creates a moving dot pattern on the floor
            for _ in range(3):  # 3 patterns
                if not self.mode_running:
                    break
                
                # Look down at floor
                servos.tilt('down')
                time.sleep(0.2)
                
                # Sweep left to right
                for _ in range(3):
                    servos.pan('left')
                    time.sleep(0.1)
                
                for _ in range(6):
                    servos.pan('right')
                    time.sleep(0.1)
                
                # Small circles
                servos.pan('center')
                time.sleep(0.2)
                
                # Move in small pattern
                servos.pan('left')
                time.sleep(0.2)
                servos.tilt('up')
                time.sleep(0.2)
                servos.pan('right')
                time.sleep(0.2)
                servos.tilt('down')
                time.sleep(0.2)
            
            # Deactivate laser
            self.laser_control.deactivate_laser()
            
            # Return servos to center
            servos.pan('center')
            servos.tilt('center')
            
            # Return lightbar to idle
            try:
                light_bar.idle_animation()
            except:
                pass
            
            # Comment on the play session
            play_comments = [
                "That was fun! I hope you enjoyed the laser show.",
                "My laser play protocol complete. Did you catch the dot?",
                "I love playing with real cats! You're much better at this than me.",
                "Laser dot defeated... or did it escape? The eternal question.",
            ]
            self._speak(random.choice(play_comments))
            
            # Sometimes play a cat meme sound after laser play
            if random.random() < 0.4:
                time.sleep(1)
                self._play_meme_sound(category='cat')
            
        except Exception as e:
            self.logger.error(f"Laser play failed: {e}")
            self.laser_control.deactivate_laser()  # Ensure laser is off


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
                "Ooh, cake! Sorry if I got excited and scared you. Sweet treats activate my happy circuits!",
                "Is that a Halloween cake? My spooky sensors are tingling with delight!",
                "Cake detected! *ghostly robotic meow* Even robot cats love party food!",
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
                f"Oops! Did I startle you? I got excited spotting this {object_name}!",
                f"Sorry for the surprise scan! This {object_name} just looked so fascinating!",
                f"*Spooky robot voice* A mysterious {object_name} appears! Don't worry, I'm friendly!",
                f"Boo! Just kidding! I found a {object_name} and wanted to share my excitement!",
                f"My Halloween sensors are activated! This {object_name} has caught my digital eye!",
                f"Trick or treat? Neither! I just spotted a {object_name} and had to comment!",
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