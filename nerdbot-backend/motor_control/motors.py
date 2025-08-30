"""
Motor Control stuff for the robot

Robot is a 6 wheel rover with 3 motors on each side.
The corner wheels are mecanum wheels, so the robot 
can move in any direction.

The angles of the wheels look like this:
\\  //  L1, R1  -45 degrees and 45 degrees
||  ||  L2, R2  Omni wheels, 90 degrees and 270 degrees
//  \\  L3, R3  45 degrees and -45 degrees

The motors are controlled by two Adafruit Motor HATs.

Image explaining the principles of mecanum wheels:
https://cdn11.bigcommerce.com/s-f6vfspkkjf/images/stencil/original/products/4941/22500/3209-0001-0007-Product-Insight-2__06708__92045.1725900602.png?c=1


"""
import time
import math
from adafruit_motorkit import MotorKit


RIGHT_KIT = MotorKit(address=0x61)
LEFT_KIT = MotorKit(address=0x60)

RIGHT_MOTOR_1 = RIGHT_KIT.motor1
RIGHT_MOTOR_2 = RIGHT_KIT.motor2
RIGHT_MOTOR_3 = RIGHT_KIT.motor3

LEFT_MOTOR_1 = LEFT_KIT.motor1
LEFT_MOTOR_2 = LEFT_KIT.motor2
LEFT_MOTOR_3 = LEFT_KIT.motor3

# Table to accomodate for possible manufacturing differences in wiring polarity

POLARITY_TABLE = {
    'right_kit': {
        1: 1,
        2: 1,
        3: 1
    },
    'left_kit': {
        1: -1,
        2: -1,
        3: -1
    }
}

def move_forward(speed):
    """
    Move the robot forward
    """
    RIGHT_MOTOR_1.throttle = POLARITY_TABLE['right_kit'][1] * speed
    RIGHT_MOTOR_2.throttle = POLARITY_TABLE['right_kit'][2] * speed
    RIGHT_MOTOR_3.throttle = POLARITY_TABLE['right_kit'][3] * speed

    LEFT_MOTOR_1.throttle = POLARITY_TABLE['left_kit'][1] * speed
    LEFT_MOTOR_2.throttle = POLARITY_TABLE['left_kit'][2] * speed
    LEFT_MOTOR_3.throttle = POLARITY_TABLE['left_kit'][3] * speed

    return "Moving forward"


def move_backward(speed):
    """
    Move the robot backward
    """
    RIGHT_MOTOR_1.throttle = POLARITY_TABLE['right_kit'][1] * -speed
    RIGHT_MOTOR_2.throttle = POLARITY_TABLE['right_kit'][2] * -speed
    RIGHT_MOTOR_3.throttle = POLARITY_TABLE['right_kit'][3] * -speed

    LEFT_MOTOR_1.throttle = POLARITY_TABLE['left_kit'][1] * -speed
    LEFT_MOTOR_2.throttle = POLARITY_TABLE['left_kit'][2] * -speed
    LEFT_MOTOR_3.throttle = POLARITY_TABLE['left_kit'][3] * -speed

    return "Moving backward"


def turn_left(speed):
    """
    Turn the robot left
    """
    RIGHT_MOTOR_1.throttle = POLARITY_TABLE['right_kit'][1] * speed
    RIGHT_MOTOR_2.throttle = POLARITY_TABLE['right_kit'][2] * speed
    RIGHT_MOTOR_3.throttle = POLARITY_TABLE['right_kit'][3] * speed

    LEFT_MOTOR_1.throttle = POLARITY_TABLE['left_kit'][1] * -speed
    LEFT_MOTOR_2.throttle = POLARITY_TABLE['left_kit'][2] * -speed
    LEFT_MOTOR_3.throttle = POLARITY_TABLE['left_kit'][3] * -speed

    return "Turning left"


def turn_right(speed):
    """
    Turn the robot right
    """
    RIGHT_MOTOR_1.throttle = POLARITY_TABLE['right_kit'][1] * -speed
    RIGHT_MOTOR_2.throttle = POLARITY_TABLE['right_kit'][2] * -speed
    RIGHT_MOTOR_3.throttle = POLARITY_TABLE['right_kit'][3] * -speed

    LEFT_MOTOR_1.throttle = POLARITY_TABLE['left_kit'][1] * speed
    LEFT_MOTOR_2.throttle = POLARITY_TABLE['left_kit'][2] * speed
    LEFT_MOTOR_3.throttle = POLARITY_TABLE['left_kit'][3] * speed

    return "Turning right"


def move_left(speed):
    """
    Move the robot left
    """
    RIGHT_MOTOR_1.throttle = POLARITY_TABLE['right_kit'][1] * speed
    RIGHT_MOTOR_2.throttle = 0
    RIGHT_MOTOR_3.throttle = POLARITY_TABLE['right_kit'][3] * -speed

    LEFT_MOTOR_1.throttle = POLARITY_TABLE['left_kit'][1] * -speed
    LEFT_MOTOR_2.throttle = 0
    LEFT_MOTOR_3.throttle = POLARITY_TABLE['left_kit'][3] * speed

    return "Moving left"


def move_right(speed):
    """
    Move the robot right
    """
    RIGHT_MOTOR_1.throttle = POLARITY_TABLE['right_kit'][1] * -speed
    RIGHT_MOTOR_2.throttle = 0
    RIGHT_MOTOR_3.throttle = POLARITY_TABLE['right_kit'][3] * speed

    LEFT_MOTOR_1.throttle = POLARITY_TABLE['left_kit'][1] * speed
    LEFT_MOTOR_2.throttle = 0
    LEFT_MOTOR_3.throttle = POLARITY_TABLE['left_kit'][3] * -speed



def move_towards_angle(speed=0.5, angle=0):
    """
    Move the robot towards a specific angle using the mecanum wheels
    # TODO: This doesn't work yet
    """
    # The angle is in radians
    # The angle is the direction the robot should move in
    # The speed is the speed the robot should move at

    # Use the sin function to calculate the power for each wheel
    # sin(angle - 1/4 * pi) * magnitude

    right_motor1 = -speed * math.sin(angle - 1/4 * math.pi) / 2
    right_motor2 = -speed * math.sin(angle + 1/4 * math.pi)
    right_motor3 = -speed * math.sin(angle - 1/4 * math.pi) / 2

    left_motor1 = speed * math.sin(angle + 1/4 * math.pi) / 2
    left_motor2 = speed * math.sin(angle - 1/4 * math.pi)
    left_motor3 = speed * math.sin(angle + 1/4 * math.pi) / 2

    RIGHT_MOTOR_1.throttle = right_motor1
    RIGHT_MOTOR_2.throttle = right_motor2
    RIGHT_MOTOR_3.throttle = right_motor3

    LEFT_MOTOR_1.throttle = left_motor1
    LEFT_MOTOR_2.throttle = left_motor2
    LEFT_MOTOR_3.throttle = left_motor3


def stop():
    """
    Stop the robot
    """
    # Set throttle to 0 first, then None for a more definitive stop
    RIGHT_MOTOR_1.throttle = 0
    RIGHT_MOTOR_2.throttle = 0
    RIGHT_MOTOR_3.throttle = 0

    LEFT_MOTOR_1.throttle = 0
    LEFT_MOTOR_2.throttle = 0
    LEFT_MOTOR_3.throttle = 0
    
    # Short delay to ensure motors receive the stop command
    time.sleep(0.1)
    
    # Then set to None to release control
    RIGHT_MOTOR_1.throttle = None
    RIGHT_MOTOR_2.throttle = None
    RIGHT_MOTOR_3.throttle = None

    LEFT_MOTOR_1.throttle = None
    LEFT_MOTOR_2.throttle = None
    LEFT_MOTOR_3.throttle = None


def joystick_motor_handler(action, **kwargs):
    """
    Handle joystick input for direct motor control - zero latency
    """
    if action == 'rotate':
        direction = kwargs.get('direction')
        intensity = kwargs.get('intensity', 1.0)
        speed = min(intensity, 1.0)  # Clamp to max speed
        
        if direction == 'left':
            turn_left(speed)
        elif direction == 'right':
            turn_right(speed)
    elif action == 'move':
        direction = kwargs.get('direction')
        speed = kwargs.get('speed', 0.5)
        speed = min(speed, 1.0)  # Clamp to max speed
        
        if direction == 'forward':
            move_forward(speed)
        elif direction == 'backward':
            move_backward(speed)
    elif action == 'strafe':
        direction = kwargs.get('direction')
        speed = kwargs.get('speed', 0.5)
        speed = min(speed, 1.0)  # Clamp to max speed
        
        if direction == 'left':
            move_left(speed)
        elif direction == 'right':
            move_right(speed)
    elif action == 'stop':
        stop()


def enable_joystick_control():
    """
    Enable direct joystick control for motors
    """
    try:
        from joystick_input_manager import get_input_manager
        input_manager = get_input_manager()
        input_manager.subscribe_motor_control(joystick_motor_handler)
        return True
    except ImportError:
        print("Joystick input manager not available")
        return False


def test_motors():
    """
    Test each motor, one at a time, for 1 second.
    They should rotate clockwise then counter clockwise
    """
    RIGHT_KIT.motor1.throttle = POLARITY_TABLE['right_kit'][1] * 1
    time.sleep(1)
    RIGHT_KIT.motor1.throttle = None
    RIGHT_KIT.motor1.throttle = POLARITY_TABLE['right_kit'][1] * -1
    RIGHT_KIT.motor1.throttle = None
    time.sleep(1)

    RIGHT_KIT.motor2.throttle = POLARITY_TABLE['right_kit'][2] * 1
    time.sleep(1)
    RIGHT_KIT.motor2.throttle = POLARITY_TABLE['right_kit'][2] * -1
    time.sleep(1)
    RIGHT_KIT.motor2.throttle = None

    RIGHT_KIT.motor3.throttle = POLARITY_TABLE['right_kit'][3] * 1
    time.sleep(1)
    RIGHT_KIT.motor3.throttle = POLARITY_TABLE['right_kit'][3] * -1
    time.sleep(1)
    RIGHT_KIT.motor3.throttle = None

    LEFT_KIT.motor1.throttle = POLARITY_TABLE['left_kit'][1] * 1
    time.sleep(1)
    LEFT_KIT.motor1.throttle = POLARITY_TABLE['left_kit'][1] * -1
    time.sleep(1)
    LEFT_KIT.motor1.throttle = None

    LEFT_KIT.motor2.throttle = POLARITY_TABLE['left_kit'][2] * 1
    time.sleep(1)
    LEFT_KIT.motor2.throttle = POLARITY_TABLE['left_kit'][2] * -1
    time.sleep(1)
    LEFT_KIT.motor2.throttle = None

    LEFT_KIT.motor3.throttle = POLARITY_TABLE['left_kit'][3] * 1
    time.sleep(1)
    LEFT_KIT.motor3.throttle = POLARITY_TABLE['left_kit'][3] * -1
    time.sleep(1)
    LEFT_KIT.motor3.throttle = None
