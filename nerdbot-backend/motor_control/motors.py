"""
Motor Control stuff for the robot

Robot is a 6 wheel rover with 3 motors on each side.
The corner wheels are mecanum wheels, so the robot 
can move in any direction.

The angles of the wheels look like this:
\  /  L1, R1  -45 degrees and 45 degrees
|  |  L2, R2  Omni wheels, 90 degrees and 270 degrees
/  \  L3, R3  45 degrees and -45 degrees

The motors are controlled by two Adafruit Motor HATs.
"""
import math
from adafruit_motorkit import MotorKit


RIGHT_KIT = MotorKit(address=0x60)
LEFT_KIT = MotorKit(address=0x61)

def move_forward(speed):
    """
    Move the robot forward
    """
    RIGHT_KIT.motor1.throttle = -speed
    RIGHT_KIT.motor2.throttle = -speed
    RIGHT_KIT.motor3.throttle = -speed

    LEFT_KIT.motor1.throttle = speed
    LEFT_KIT.motor2.throttle = speed
    LEFT_KIT.motor3.throttle = speed

    return "Moving forward"


def move_backward(speed):
    """
    Move the robot backward
    """
    RIGHT_KIT.motor1.throttle = speed
    RIGHT_KIT.motor2.throttle = speed
    RIGHT_KIT.motor3.throttle = speed

    LEFT_KIT.motor1.throttle = -speed
    LEFT_KIT.motor2.throttle = -speed
    LEFT_KIT.motor3.throttle = -speed

    return "Moving backward"


def turn_left(speed):
    """
    Turn the robot left
    """
    RIGHT_KIT.motor1.throttle = speed
    RIGHT_KIT.motor2.throttle = speed
    RIGHT_KIT.motor3.throttle = speed

    LEFT_KIT.motor1.throttle = speed
    LEFT_KIT.motor2.throttle = speed
    LEFT_KIT.motor3.throttle = speed

    return "Turning left"


def turn_right(speed):
    """
    Turn the robot right
    """
    RIGHT_KIT.motor1.throttle = -speed
    RIGHT_KIT.motor2.throttle = -speed
    RIGHT_KIT.motor3.throttle = -speed

    LEFT_KIT.motor1.throttle = -speed
    LEFT_KIT.motor2.throttle = -speed
    LEFT_KIT.motor3.throttle = -speed

    return "Turning right"


def move_left(speed):
    """
    Move the robot left
    Motor2 on RIGHT_KIT and LEFT_LIT are equipped with omni wheels, so they don't need to move
    """
    RIGHT_KIT.motor1.throttle = -speed
    RIGHT_KIT.motor2.throttle = 0
    RIGHT_KIT.motor3.throttle = speed

    LEFT_KIT.motor1.throttle = speed
    LEFT_KIT.motor2.throttle = 0
    LEFT_KIT.motor3.throttle = -speed

    return "Moving left"


def move_right(speed):
    """
    Move the robot right
    Motor2 needs the extra speed to move the robot in the desired direction
    """
    RIGHT_KIT.motor1.throttle = speed
    RIGHT_KIT.motor2.throttle = 0
    RIGHT_KIT.motor3.throttle = -speed

    LEFT_KIT.motor1.throttle = -speed
    LEFT_KIT.motor2.throttle = 0
    LEFT_KIT.motor3.throttle = speed



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

    RIGHT_KIT.motor1.throttle = right_motor1
    RIGHT_KIT.motor2.throttle = right_motor2
    RIGHT_KIT.motor3.throttle = right_motor3

    LEFT_KIT.motor1.throttle = left_motor1
    LEFT_KIT.motor2.throttle = left_motor2
    LEFT_KIT.motor3.throttle = left_motor3


def stop():
    """
    Stop the robot
    """
    RIGHT_KIT.motor1.throttle = None
    RIGHT_KIT.motor2.throttle = None
    RIGHT_KIT.motor3.throttle = None

    LEFT_KIT.motor1.throttle = None
    LEFT_KIT.motor2.throttle = None
    LEFT_KIT.motor3.throttle = None


