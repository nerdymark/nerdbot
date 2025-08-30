"""
Joystick Control Service for NerdBot

This service monitors USB game controllers and translates input to robot movement.
Designed for Valve Steam Controller but works with most USB gamepads.

Controls mapping:
- Left direction pad: Forward/backward movement, left/right strafing
- Right universal pad: Rotate left/right  
- Analog stick: Pan/tilt camera controls
"""

import pygame
import requests
import time
import logging
import threading
from collections import defaultdict

# Flask server configuration
FLASK_BASE_URL = "http://localhost:5000/api"

class JoystickService:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.joystick = None
        self.running = False
        self.last_commands = defaultdict(float)  # Track last command times to prevent spam
        self.command_cooldown = 0.05  # 50ms cooldown between commands
        
        # Dead zones to prevent drift
        self.analog_deadzone = 0.15
        self.trigger_deadzone = 0.1
        
        # Speed settings
        self.motor_speed = 0.8
        self.servo_step = 2  # degrees per input
        
        # Button state tracking
        self.button_states = {}
        self.last_button_states = {}
        
    def find_controller(self):
        """Find and initialize the first available joystick"""
        pygame.joystick.quit()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        self.logger.info(f"Found {joystick_count} joystick(s)")
        
        if joystick_count == 0:
            self.logger.warning("No joysticks found")
            return False
            
        # Use the first available joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        self.logger.info(f"Initialized joystick: {self.joystick.get_name()}")
        self.logger.info(f"Axes: {self.joystick.get_numaxes()}")
        self.logger.info(f"Buttons: {self.joystick.get_numbuttons()}")
        self.logger.info(f"Hats: {self.joystick.get_numhats()}")
        
        return True
        
    def send_motor_command(self, direction):
        """Send motor command to Flask API"""
        current_time = time.time()
        if current_time - self.last_commands['motor'] < self.command_cooldown:
            return
            
        try:
            url = f"{FLASK_BASE_URL}/motor/{direction}"
            response = requests.post(url, timeout=1)
            if response.status_code == 200:
                self.logger.debug(f"Motor command sent: {direction}")
                self.last_commands['motor'] = current_time
            else:
                self.logger.error(f"Motor command failed: {response.status_code}")
        except requests.RequestException as e:
            self.logger.error(f"Motor API request failed: {e}")
            
    def send_servo_command(self, servo_type, direction):
        """Send servo command to Flask API"""
        current_time = time.time()
        command_key = f"{servo_type}_{direction}"
        if current_time - self.last_commands[command_key] < self.command_cooldown:
            return
            
        try:
            url = f"{FLASK_BASE_URL}/{servo_type}/{direction}"
            response = requests.post(url, timeout=1)
            if response.status_code == 200:
                self.logger.debug(f"Servo command sent: {servo_type} {direction}")
                self.last_commands[command_key] = current_time
            else:
                self.logger.error(f"Servo command failed: {response.status_code}")
        except requests.RequestException as e:
            self.logger.error(f"Servo API request failed: {e}")
            
    def handle_dpad_input(self, dpad_x, dpad_y):
        """Handle D-pad input for movement controls"""
        # DISABLED FOR SAFETY - Steam Controller D-pad mapping needs investigation
        # The axes 6,7 have default values of -1.0 which were causing auto-movement
        self.logger.debug(f"D-pad input: X={dpad_x:.3f}, Y={dpad_y:.3f} (DISABLED)")
        return
            
    def handle_analog_stick(self, stick_x, stick_y):
        """Handle analog stick input for pan/tilt controls"""
        # Apply deadzone
        if abs(stick_x) < self.analog_deadzone:
            stick_x = 0
        if abs(stick_y) < self.analog_deadzone:
            stick_y = 0
            
        # Pan control (left/right)
        if stick_x < -self.analog_deadzone:  # Left
            self.send_servo_command('pan', 'left')
        elif stick_x > self.analog_deadzone:  # Right
            self.send_servo_command('pan', 'right')
            
        # Tilt control (up/down) - note: Y axis is typically inverted
        if stick_y < -self.analog_deadzone:  # Up
            self.send_servo_command('tilt', 'up')
        elif stick_y > self.analog_deadzone:  # Down
            self.send_servo_command('tilt', 'down')
            
    def handle_button_input(self, button_id, pressed):
        """Handle button press events"""
        if not pressed:  # Only handle button press, not release
            return
            
        # Steam Controller specific button mappings
        # These button IDs may vary by controller
        if button_id == 0:  # A button - center servos
            self.send_servo_command('pan', 'center')
        elif button_id == 1:  # B button - stop motors
            self.send_motor_command('stop')
        elif button_id == 2:  # X button - could be used for other functions
            pass
        elif button_id == 3:  # Y button - could be used for other functions
            pass
            
    def handle_trigger_input(self, left_trigger, right_trigger):
        """Handle trigger input for rotation"""
        # Apply deadzone
        if abs(left_trigger) < self.trigger_deadzone:
            left_trigger = 0
        if abs(right_trigger) < self.trigger_deadzone:
            right_trigger = 0
            
        # Rotation controls
        if left_trigger > self.trigger_deadzone:  # Left trigger - rotate left
            self.send_motor_command('left')
        elif right_trigger > self.trigger_deadzone:  # Right trigger - rotate right
            self.send_motor_command('right')
            
    def update_button_states(self):
        """Update button state tracking"""
        self.last_button_states = self.button_states.copy()
        self.button_states = {}
        
        if self.joystick:
            for i in range(self.joystick.get_numbuttons()):
                self.button_states[i] = self.joystick.get_button(i)
                
    def process_input(self):
        """Process all joystick inputs"""
        if not self.joystick:
            return
            
        pygame.event.pump()  # Update joystick state
        
        # Update button states
        self.update_button_states()
        
        # Handle D-pad input - for movement
        # Steam Controller uses axes 6 and 7 for D-pad, not hats
        if self.joystick.get_numaxes() >= 8:
            # Steam Controller D-pad mapping
            dpad_x = self.joystick.get_axis(6)  # D-pad X (left/right)
            dpad_y = self.joystick.get_axis(7)  # D-pad Y (up/down)
            self.handle_dpad_input(dpad_x, dpad_y)
        elif self.joystick.get_numhats() > 0:
            # Fallback for traditional controllers with hat-based D-pad
            hat_x, hat_y = self.joystick.get_hat(0)
            self.handle_dpad_input(hat_x, hat_y)
            
        # Handle analog stick - for pan/tilt 
        # Steam Controller: right stick is axes 2,3
        if self.joystick.get_numaxes() >= 4:
            stick_x = self.joystick.get_axis(2)  # Right stick X
            stick_y = self.joystick.get_axis(3)  # Right stick Y
            self.handle_analog_stick(stick_x, stick_y)
            
        # Handle triggers - for rotation
        if self.joystick.get_numaxes() >= 6:
            left_trigger = self.joystick.get_axis(4)   # Left trigger
            right_trigger = self.joystick.get_axis(5)  # Right trigger
            self.handle_trigger_input(left_trigger, right_trigger)
        elif self.joystick.get_numaxes() >= 2:
            # Fallback: use left stick X-axis for rotation if no triggers
            left_stick_x = self.joystick.get_axis(0)  # Left stick X
            if abs(left_stick_x) > self.analog_deadzone:
                if left_stick_x < -self.analog_deadzone:
                    self.send_motor_command('left')
                elif left_stick_x > self.analog_deadzone:
                    self.send_motor_command('right')
                    
        # Handle button presses
        for button_id, pressed in self.button_states.items():
            last_pressed = self.last_button_states.get(button_id, False)
            if pressed and not last_pressed:  # Button just pressed
                self.handle_button_input(button_id, True)
                
    def run(self):
        """Main service loop"""
        self.logger.info("Starting joystick service...")
        
        if not self.find_controller():
            self.logger.error("No controller found. Exiting.")
            return
            
        self.running = True
        self.logger.info("Joystick service running. Press Ctrl+C to stop.")
        
        try:
            while self.running:
                self.process_input()
                time.sleep(0.02)  # 50Hz update rate
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested...")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the service"""
        self.running = False
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        self.logger.info("Joystick service stopped.")
        
    def print_controller_info(self):
        """Print detailed controller information for debugging"""
        if not self.joystick:
            self.logger.error("No joystick initialized")
            return
            
        print(f"\nController: {self.joystick.get_name()}")
        print(f"Axes: {self.joystick.get_numaxes()}")
        print(f"Buttons: {self.joystick.get_numbuttons()}")
        print(f"Hats: {self.joystick.get_numhats()}")
        
        # Print current state
        print("\nCurrent State:")
        for i in range(self.joystick.get_numaxes()):
            value = self.joystick.get_axis(i)
            print(f"  Axis {i}: {value:.3f}")
            
        for i in range(self.joystick.get_numbuttons()):
            pressed = self.joystick.get_button(i)
            print(f"  Button {i}: {'PRESSED' if pressed else 'released'}")
            
        for i in range(self.joystick.get_numhats()):
            hat_x, hat_y = self.joystick.get_hat(i)
            print(f"  Hat {i}: ({hat_x}, {hat_y})")


def test_mode():
    """Test mode to debug controller inputs"""
    service = JoystickService()
    
    if not service.find_controller():
        print("No controller found!")
        return
        
    print("Test mode - press buttons and move sticks to see values")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            pygame.event.pump()
            service.print_controller_info()
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mode()
    else:
        service = JoystickService()
        service.run()