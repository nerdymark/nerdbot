"""
Shared Joystick Input Manager for NerdBot

This module provides direct, zero-latency joystick integration for both motor and servo control.
Eliminates HTTP API overhead by allowing modules to directly subscribe to joystick events.
"""

import pygame
import threading
import time
import logging
from typing import Callable, Dict, List
from collections import defaultdict


class JoystickInputManager:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.joystick = None
        self.running = False
        self.thread = None
        
        # Event subscribers - functions that get called on input events
        self.motor_subscribers: List[Callable] = []
        self.servo_subscribers: List[Callable] = []
        
        # Dead zones to prevent drift
        self.analog_deadzone = 0.15
        self.trigger_deadzone = 0.1
        
        # Button state tracking
        self.button_states = {}
        self.last_button_states = {}
        
        # Input state tracking
        self.last_input_time = defaultdict(float)
        self.input_threshold = 0.01  # 10ms between identical inputs
        
        # Combined control state - tracks current input from all surfaces
        self.control_state = {
            'forward_back': 0.0,    # -1.0 to 1.0 (negative = backward, positive = forward)
            'left_right': 0.0,      # -1.0 to 1.0 (negative = left strafe, positive = right strafe)  
            'rotation': 0.0         # -1.0 to 1.0 (negative = rotate left, positive = rotate right)
        }
        
        # Button debouncing - prevent rapid button presses
        self.button_last_press = defaultdict(float)
        self.button_debounce_time = 0.5  # 500ms debounce for most buttons
        self.meme_button_debounce_time = 1.0  # 1 second debounce for meme sound button
        self.laser_headlight_debounce_time = 0.5  # 500ms debounce for laser/headlight buttons
        
        # Motor state tracking to prevent repeated stop commands
        self.motors_active = False
        
    def find_controller(self) -> bool:
        """Find and initialize the first available joystick"""
        pygame.joystick.quit()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        self.logger.info(f"Found {joystick_count} joystick(s)")
        
        if joystick_count == 0:
            self.logger.warning("No joysticks found")
            return False
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        self.logger.info(f"Initialized joystick: {self.joystick.get_name()}")
        return True
    
    def subscribe_motor_control(self, callback: Callable):
        """Subscribe a function to receive motor control events"""
        self.motor_subscribers.append(callback)
        self.logger.info(f"Motor control subscriber added: {callback.__name__}")
    
    def subscribe_servo_control(self, callback: Callable):
        """Subscribe a function to receive servo control events"""
        self.servo_subscribers.append(callback)
        self.logger.info(f"Servo control subscriber added: {callback.__name__}")
    
    def _notify_motor_subscribers(self, action: str, **kwargs):
        """Notify all motor subscribers of an input event"""
        for callback in self.motor_subscribers:
            try:
                callback(action, **kwargs)
            except Exception as e:
                self.logger.error(f"Motor subscriber {callback.__name__} error: {e}")
    
    def _notify_servo_subscribers(self, action: str, **kwargs):
        """Notify all servo subscribers of an input event"""
        for callback in self.servo_subscribers:
            try:
                callback(action, **kwargs)
            except Exception as e:
                self.logger.error(f"Servo subscriber {callback.__name__} error: {e}")
    
    def _should_process_input(self, input_key: str) -> bool:
        """Check if enough time has passed to process this input"""
        current_time = time.time()
        if current_time - self.last_input_time[input_key] < self.input_threshold:
            return False
        self.last_input_time[input_key] = current_time
        return True
    
    def _apply_combined_control(self):
        """Apply the current combined control state to motors using mecanum wheel math"""
        forward_back = self.control_state['forward_back']
        left_right = self.control_state['left_right'] 
        rotation = self.control_state['rotation']
        
        # If all inputs are within deadzone, stop the motors (but only once)
        if abs(forward_back) < self.analog_deadzone and abs(left_right) < self.analog_deadzone and abs(rotation) < self.trigger_deadzone:
            if self.motors_active:  # Only send stop command if motors were previously active
                self._notify_motor_subscribers('stop')
                self.motors_active = False
            return
            
        # Use mecanum wheel mathematics to combine all motion vectors
        # Each wheel gets a combination of forward/back, strafe, and rotation
        # This allows for complex movements like forward+strafe+rotation simultaneously
        
        # Calculate individual wheel speeds based on mecanum geometry
        # Front-left wheel: forward + strafe + rotation
        # Front-right wheel: forward - strafe - rotation  
        # Rear-left wheel: forward - strafe + rotation
        # Rear-right wheel: forward + strafe - rotation
        
        # Since our motor control functions are discrete, we'll determine
        # the dominant motion and apply it, but with combined intensity
        combined_intensity = min(1.0, abs(forward_back) + abs(left_right) + abs(rotation))
        
        # Create a combined motion vector and determine the best motor action
        if abs(forward_back) > self.analog_deadzone or abs(left_right) > self.analog_deadzone or abs(rotation) > self.trigger_deadzone:
            # Mark motors as active when sending commands
            self.motors_active = True
            # Use the new combined motor control
            self._notify_motor_subscribers('combined', 
                                         forward_back=forward_back,
                                         left_right=left_right, 
                                         rotation=rotation,
                                         intensity=combined_intensity)
    
    def _toggle_headlights(self):
        """Toggle headlights functionality via HTTP API for state synchronization"""
        try:
            import requests
            
            # Use the explicit on/off endpoints based on current state
            status_response = requests.get('http://localhost:5000/api/headlights/status', timeout=0.5)
            if status_response.status_code == 200:
                current_status = status_response.json().get('headlights_on', False)
                self.logger.info(f"Current headlights status: {'on' if current_status else 'off'}")
                
                # Use explicit on/off instead of toggle to avoid race conditions
                if current_status:
                    response = requests.post('http://localhost:5000/api/headlights/off', timeout=0.5)
                    target_state = False
                else:
                    response = requests.post('http://localhost:5000/api/headlights/on', timeout=0.5)
                    target_state = True
                    
                if response.status_code == 200:
                    self.logger.info(f"Headlights turned {'on' if target_state else 'off'} successfully")
                else:
                    self.logger.error(f"Headlights API request failed: {response.status_code}")
            else:
                self.logger.warning("Could not get headlights status")
        except Exception as e:
            self.logger.error(f"Failed to toggle headlights via API: {e}")
    
    def _toggle_laser(self):
        """Toggle laser functionality via HTTP API to avoid GPIO conflicts"""
        try:
            import requests
            
            # Use the explicit on/off endpoints based on current state
            status_response = requests.get('http://localhost:5000/api/laser/status', timeout=0.5)
            if status_response.status_code == 200:
                current_status = status_response.json().get('laser_on', False)
                self.logger.info(f"Current laser status: {'on' if current_status else 'off'}")
                
                # Use explicit on/off instead of toggle to avoid race conditions
                if current_status:
                    response = requests.post('http://localhost:5000/api/laser/off', timeout=0.5)
                    target_state = False
                else:
                    response = requests.post('http://localhost:5000/api/laser/on', timeout=0.5)
                    target_state = True
                    
                if response.status_code == 200:
                    self.logger.info(f"Laser turned {'on' if target_state else 'off'} successfully")
                else:
                    self.logger.error(f"Laser API request failed: {response.status_code}")
            else:
                self.logger.warning("Could not get laser status")
        except Exception as e:
            self.logger.error(f"Failed to toggle laser via API: {e}")
    
    def _play_random_meme_sound(self):
        """Play random meme sound via HTTP API (non-blocking)"""
        def make_request():
            try:
                import requests
                # Use the Flask server's random meme sound API with short timeout
                response = requests.post('http://localhost:5000/api/meme_sound/random', timeout=0.5)
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"Random meme sound started: {data.get('message', 'Success')}")
                else:
                    self.logger.error(f"Random meme sound API request failed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to play random meme sound via API: {e}")
        
        # Run HTTP request in separate thread to avoid blocking button detection
        import threading
        threading.Thread(target=make_request, daemon=True).start()
        self.logger.info("Random meme sound request sent (non-blocking)")
    
    def _play_welcome_message(self):
        """Play welcome message via TTS API (non-blocking)"""
        def make_request():
            try:
                import requests
                # Use the Flask server's TTS API with the welcome message
                response = requests.post('http://localhost:5000/api/tts/welcome', timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"Welcome message started: {data.get('message', 'Success')}")
                else:
                    self.logger.error(f"Welcome message API request failed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to play welcome message via API: {e}")
        
        # Run HTTP request in separate thread to avoid blocking button detection
        import threading
        threading.Thread(target=make_request, daemon=True).start()
        self.logger.info("Welcome message request sent (non-blocking)")
    
    def _cycle_mode(self):
        """Cycle through robot modes via HTTP API (non-blocking)"""
        def make_request():
            try:
                import requests
                # Use the Flask server's mode cycling API with short timeout
                response = requests.post('http://localhost:5000/api/mode/cycle', timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"Mode cycled successfully: {data.get('message', 'Success')}")
                else:
                    self.logger.error(f"Mode cycle API request failed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to cycle mode via API: {e}")
        
        # Run HTTP request in separate thread to avoid blocking button detection
        import threading
        threading.Thread(target=make_request, daemon=True).start()
        self.logger.info("Mode cycle request sent (non-blocking)")
    
    def _handle_dpad_input(self, dpad_x: float, dpad_y: float):
        """Handle D-pad input for movement controls"""
        # Note: D-pad is currently disabled in original code due to Steam Controller mapping issues
        # Keeping the structure for future implementation
        pass
    
    def _handle_analog_stick(self, stick_x: float, stick_y: float):
        """Handle left analog stick input for pan/tilt servo controls"""
        # Apply deadzone
        if abs(stick_x) < self.analog_deadzone:
            stick_x = 0
        if abs(stick_y) < self.analog_deadzone:
            stick_y = 0
            
        # Pan control (left/right)
        if stick_x < -self.analog_deadzone:  # Left
            if self._should_process_input('pan_left'):
                self._notify_servo_subscribers('pan', direction='left')
        elif stick_x > self.analog_deadzone:  # Right
            if self._should_process_input('pan_right'):
                self._notify_servo_subscribers('pan', direction='right')
            
        # Tilt control (up/down) - Y axis typically inverted
        if stick_y < -self.analog_deadzone:  # Up
            if self._should_process_input('tilt_up'):
                self._notify_servo_subscribers('tilt', direction='up')
        elif stick_y > self.analog_deadzone:  # Down
            if self._should_process_input('tilt_down'):
                self._notify_servo_subscribers('tilt', direction='down')

    def _handle_left_touchpad(self, pad_x: float, pad_y: float):
        """Handle left touchpad for robot movement (WASD-style) - updates control state"""
        # Apply deadzone
        if abs(pad_x) < self.analog_deadzone:
            pad_x = 0
        if abs(pad_y) < self.analog_deadzone:
            pad_y = 0
            
        # Update control state instead of sending direct motor commands
        # Y-axis: forward/backward movement (inverted: negative pad_y = forward)
        self.control_state['forward_back'] = -pad_y  # Invert Y axis
        
        # X-axis: left/right strafing (negative pad_x = strafe left)
        self.control_state['left_right'] = pad_x
        
        # Apply the combined control
        if self._should_process_input('left_touchpad'):
            self._apply_combined_control()

    def _handle_right_touchpad(self, pad_x: float, pad_y: float):
        """Handle right circular touchpad for rotation - updates control state"""
        # Apply deadzone to individual axes
        if abs(pad_x) < self.analog_deadzone:
            pad_x = 0
        if abs(pad_y) < self.analog_deadzone:
            pad_y = 0
        
        # Update rotation state based on X position (left/right)
        # Negative pad_x = rotate left, positive pad_x = rotate right
        self.control_state['rotation'] = pad_x
        
        # Apply the combined control
        if self._should_process_input('right_touchpad'):
            self._apply_combined_control()
    
    def _handle_button_input(self, button_id: int, pressed: bool):
        """Handle button press events with debouncing"""
        if not pressed:  # Only handle button press, not release
            return
            
        # Check debouncing for all buttons
        current_time = time.time()
        if button_id == 5:  # Meme sound button
            debounce_time = self.meme_button_debounce_time
        elif button_id in [10, 11]:  # Headlight and laser buttons
            debounce_time = self.laser_headlight_debounce_time
        else:
            debounce_time = self.button_debounce_time
        
        if current_time - self.button_last_press[button_id] < debounce_time:
            return  # Skip if too soon after last press
        self.button_last_press[button_id] = current_time
            
        # Debug logging to identify button IDs
        self.logger.info(f"Button pressed: ID {button_id}")
            
        # Steam Controller specific button mappings
        if button_id == 2:  # A button - center pan/tilt
            self._notify_servo_subscribers('pan_center')
        elif button_id == 1:  # Button ID 1 - reserved for future use
            pass
        elif button_id == 4:  # X button - cycle modes
            self._cycle_mode()
        elif button_id == 3:  # B button - emergency stop motors
            self._notify_motor_subscribers('stop')
        elif button_id == 5:  # Y button - play random meme sound
            self._play_random_meme_sound()
        elif button_id == 6:  # Back/Menu button (left of Steam button) - reserved for future use
            pass
        elif button_id == 7:  # Right shoulder button - play welcome message
            self._play_welcome_message()
        elif button_id == 10:  # Button left of Menu button - toggle headlights
            self._toggle_headlights()
        elif button_id == 11:  # Button ID 11 - toggle laser
            self._toggle_laser()
        # Additional buttons could be mapped here (4=LB, 5=RB, etc.)
    
    def _handle_trigger_input(self, left_trigger: float, right_trigger: float):
        """Handle trigger input for rotation"""
        # Apply deadzone
        if abs(left_trigger) < self.trigger_deadzone:
            left_trigger = 0
        if abs(right_trigger) < self.trigger_deadzone:
            right_trigger = 0
            
        # Rotation controls
        if left_trigger > self.trigger_deadzone:  # Left trigger - rotate left
            if self._should_process_input('rotate_left'):
                self._notify_motor_subscribers('rotate', direction='left', intensity=left_trigger)
        elif right_trigger > self.trigger_deadzone:  # Right trigger - rotate right
            if self._should_process_input('rotate_right'):
                self._notify_motor_subscribers('rotate', direction='right', intensity=right_trigger)
    
    def _update_button_states(self):
        """Update button state tracking"""
        self.last_button_states = self.button_states.copy()
        self.button_states = {}
        
        if self.joystick:
            for i in range(self.joystick.get_numbuttons()):
                self.button_states[i] = self.joystick.get_button(i)
    
    def _process_input(self):
        """Process all joystick inputs for Steam Controller"""
        if not self.joystick:
            return
            
        pygame.event.pump()  # Update joystick state
        
        # Update button states
        self._update_button_states()
        
        # Steam Controller axis mapping:
        # Axis 0: Left analog stick X
        # Axis 1: Left analog stick Y  
        # Axis 2: Right touchpad X
        # Axis 3: Right touchpad Y
        # Axis 4: Left touchpad X
        # Axis 5: Left touchpad Y
        
        # Handle left analog stick for pan/tilt servos
        if self.joystick.get_numaxes() >= 2:
            left_stick_x = self.joystick.get_axis(0)  # Left analog stick X
            left_stick_y = self.joystick.get_axis(1)  # Left analog stick Y
            self._handle_analog_stick(left_stick_x, left_stick_y)
        
        # Handle left touchpad for robot movement (WASD)
        if self.joystick.get_numaxes() >= 6:
            left_touchpad_x = self.joystick.get_axis(4)  # Left touchpad X
            left_touchpad_y = self.joystick.get_axis(5)  # Left touchpad Y
            self._handle_left_touchpad(left_touchpad_x, left_touchpad_y)
            
        # Handle right touchpad for rotation
        if self.joystick.get_numaxes() >= 4:
            right_touchpad_x = self.joystick.get_axis(2)  # Right touchpad X
            right_touchpad_y = self.joystick.get_axis(3)  # Right touchpad Y
            self._handle_right_touchpad(right_touchpad_x, right_touchpad_y)
                    
        # Handle button presses
        for button_id, pressed in self.button_states.items():
            last_pressed = self.last_button_states.get(button_id, False)
            if pressed and not last_pressed:  # Button just pressed
                self._handle_button_input(button_id, True)
    
    def _run_loop(self):
        """Main input processing loop - runs in separate thread"""
        self.logger.info("Joystick input manager thread started")
        
        # Add startup delay to prevent initial input processing
        self.logger.info("Waiting 2 seconds before processing joystick input to prevent startup interference...")
        time.sleep(2.0)
        
        try:
            # Clear any initial button/axis states to prevent phantom inputs
            if self.joystick:
                pygame.event.pump()
                self._update_button_states()
                self.logger.info("Initial joystick states cleared, starting input processing")
            
            while self.running:
                self._process_input()
                time.sleep(0.01)  # 100Hz update rate for responsiveness
                
        except Exception as e:
            self.logger.error(f"Input processing error: {e}")
        finally:
            self.logger.info("Joystick input manager thread stopped")
    
    def start(self) -> bool:
        """Start the joystick input manager"""
        if self.running:
            self.logger.warning("Input manager already running")
            return True
            
        if not self.find_controller():
            self.logger.error("No controller found")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        self.logger.info("Joystick input manager started successfully")
        return True
    
    def stop(self):
        """Stop the joystick input manager"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        
        self.logger.info("Joystick input manager stopped")


# Global instance for easy access
_input_manager = None

def get_input_manager() -> JoystickInputManager:
    """Get the global input manager instance"""
    global _input_manager
    if _input_manager is None:
        _input_manager = JoystickInputManager()
    return _input_manager