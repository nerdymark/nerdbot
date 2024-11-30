import time
import random
from adafruit_servokit import ServoKit


# Hat is addressed at 0x42
kit = ServoKit(channels=16, address=0x42)

tilt_servo = kit.servo[0]
pan_servo = kit.servo[1]
laser_servo = kit.servo[4]

# tilt_servo.set_pulse_width_range(900, 2600)
tilt_servo.set_pulse_width_range(2600, 900)
pan_servo.set_pulse_width_range(2850, 890)

# Laser servo is a laser diode that we want to dim and brighten with PWM
laser_servo.set_pulse_width_range(900, 2600)

# Dicts that adjust the servo angles to the correct values
tilt_dict = {
    'up': 90,
    'forward': 180,
    'backward': 0
}


pan_dict = {
    'left': 0,
    'center': 90,
    'right': 180
}


def tilt_full_direction(direction):
    tilt_servo.angle = tilt_dict[direction]
    return f"Tilted {direction}"


def pan_full_direction(direction):
    pan_servo.angle = pan_dict[direction]
    return f"Panned {direction}"


def tilt(direction):
    # Get the current angle of the tilt servo
    current_angle = tilt_servo.angle
    # go towards the desired angle by 10 degrees
    if direction == 'up':
        tilt_servo.angle = current_angle - 5
    elif direction == 'down':
        tilt_servo.angle = current_angle + 5
    return tilt_servo.angle


def pan(direction):
    # Get the current angle of the pan servo
    current_angle = pan_servo.angle
    # go towards the desired angle by 10 degrees
    if direction == 'left':
        pan_servo.angle = current_angle - 5
    elif direction == 'right':
        pan_servo.angle = current_angle + 5
    elif direction == 'center':
        reset_servos()
    return pan_servo.angle


def tilt_to(angle):
    tilt_servo.angle = angle
    return tilt_servo.angle


def pan_to(angle):
    pan_servo.angle = angle
    return pan_servo.angle


def reset_servos():
    tilt_servo.angle = tilt_dict['forward']
    pan_servo.angle = pan_dict['center']
    return "Servos Reset"


def move_servos():
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
    tilt_servo.angle = random.randint(0, 180)
    pan_servo.angle = random.randint(0, 180)
    return "Servos Moved Randomly"


def laser_on():
    laser_servo.angle = 180
    return "Laser On"


def laser_off():
    laser_servo.angle = 0
    return "Laser Off"
