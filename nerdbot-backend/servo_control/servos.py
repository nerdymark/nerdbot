import time
import random
from adafruit_servokit import ServoKit


# Hat is addressed at 0x42
kit = ServoKit(channels=16, address=0x42)

tilt_servo = kit.servo[0]
pan_servo = kit.servo[1]
laser_servo = kit.servo[4]

# tilt_servo.set_pulse_width_range(900, 2600)
tilt_servo.set_pulse_width_range(1700, 1100)
pan_servo.set_pulse_width_range(2750, 990)

# Laser servo is a laser diode that we want to dim and brighten with PWM
laser_servo.set_pulse_width_range(900, 2600)

# Dicts that adjust the servo angles to the correct values
tilt_dict = {
    'up': 90,
    'forward': 150,
    'backward': 0
}


pan_dict = {
    'left': 0,
    'center': 90,
    'right': 180
}


def tilt_full_direction(direction):
    """
    Tilt the camera to a full direction
    """
    tilt_servo.angle = tilt_dict[direction]
    return f"Tilted {direction}"


def pan_full_direction(direction):
    """
    Pan the camera to a full direction
    """
    pan_servo.angle = pan_dict[direction]
    return f"Panned {direction}"


def tilt(direction):
    """
    Tilt the camera up or down
    """
    current_angle = tilt_servo.angle

    if direction == 'up':
        new_angle = max(0, current_angle - 2)  # Clamp to minimum 0
        tilt_servo.angle = new_angle
    elif direction == 'down':
        new_angle = min(180, current_angle + 2)  # Clamp to maximum 180
        tilt_servo.angle = new_angle
    return tilt_servo.angle


def pan(direction):
    """
    Pan the camera left or right
    """
    current_angle = pan_servo.angle

    if direction == 'left':
        new_angle = max(0, current_angle - 2)  # Clamp to minimum 0
        pan_servo.angle = new_angle
    elif direction == 'right':
        new_angle = min(180, current_angle + 2)  # Clamp to maximum 180
        pan_servo.angle = new_angle
    elif direction == 'center':
        reset_servos()
    return pan_servo.angle


def tilt_to(angle):
    """
    Tilt the camera to a specific angle
    """
    try:
        tilt_servo.angle = angle
    except ValueError:
        return "Invalid angle"
    return tilt_servo.angle


def pan_to(angle):
    """
    Pan the camera to a specific angle
    """
    try:
        pan_servo.angle = angle
    except ValueError:
        return "Invalid angle"
    return pan_servo.angle


def reset_servos():
    """
    Reset the servos to their default positions
    """
    tilt_servo.angle = tilt_dict['forward']
    pan_servo.angle = pan_dict['center']
    return "Servos Reset"


def move_servos():
    """
    Move the servos to their extreme positions
    """
    tilt_servo.angle = 0
    pan_servo.angle = 0
    time.sleep(1)
    tilt_servo.angle = 180
    pan_servo.angle = 180
    time.sleep(1)
    tilt_servo.angle = 90
    pan_servo.angle = 90
    return "Servos Moved"


def move_servos_random():
    """
    Move the servos to random positions
    """
    tilt_servo.angle = random.randint(0, 180)
    pan_servo.angle = random.randint(0, 180)
    return "Servos Moved Randomly"


def laser_on():
    """
    Turn the laser on
    """
    laser_servo.angle = 180
    return "Laser On"


def laser_off():
    """
    Turn the laser off
    """
    laser_servo.angle = 0
    return "Laser Off"


def joystick_servo_handler(action, **kwargs):
    """
    Handle joystick input for direct servo control - zero latency
    """
    if action == 'pan':
        direction = kwargs.get('direction')
        if direction in ['left', 'right']:
            pan(direction)
    elif action == 'tilt':
        direction = kwargs.get('direction')
        if direction in ['up', 'down']:
            tilt(direction)
    elif action == 'center':
        reset_servos()


def enable_joystick_control():
    """
    Enable direct joystick control for servos
    """
    try:
        from joystick_input_manager import get_input_manager
        input_manager = get_input_manager()
        input_manager.subscribe_servo_control(joystick_servo_handler)
        return True
    except ImportError:
        print("Joystick input manager not available")
        return False