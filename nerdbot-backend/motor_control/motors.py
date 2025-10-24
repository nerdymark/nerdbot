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


# Initialize motor kits
RIGHT_KIT = MotorKit(address=0x61)
LEFT_KIT = MotorKit(address=0x60)

# Immediately stop all motors on the kits before assigning
print("Stopping all motors during initialization...")
for i in range(1, 5):  # MotorKit has 4 motor channels
    try:
        motor = getattr(RIGHT_KIT, f'motor{i}')
        motor.throttle = 0
        motor.throttle = None
        print(f"RIGHT_KIT.motor{i} stopped")
    except Exception as e:
        print(f"RIGHT_KIT.motor{i} stop failed: {e}")
    try:
        motor = getattr(LEFT_KIT, f'motor{i}')
        motor.throttle = 0
        motor.throttle = None
        print(f"LEFT_KIT.motor{i} stopped")
    except Exception as e:
        print(f"LEFT_KIT.motor{i} stop failed: {e}")

# Small delay to ensure hardware processes the stop commands
time.sleep(0.1)

# Now assign the motor objects
RIGHT_MOTOR_1 = RIGHT_KIT.motor1
RIGHT_MOTOR_2 = RIGHT_KIT.motor2
RIGHT_MOTOR_3 = RIGHT_KIT.motor3
RIGHT_MOTOR_4 = RIGHT_KIT.motor4  # Unused but stop it anyway

LEFT_MOTOR_1 = LEFT_KIT.motor1
LEFT_MOTOR_2 = LEFT_KIT.motor2
LEFT_MOTOR_3 = LEFT_KIT.motor3
LEFT_MOTOR_4 = LEFT_KIT.motor4  # Unused but stop it anyway

# Set each motor to stopped state explicitly (including motor4)
RIGHT_MOTOR_1.throttle = 0
RIGHT_MOTOR_2.throttle = 0
RIGHT_MOTOR_3.throttle = 0
RIGHT_MOTOR_4.throttle = 0
LEFT_MOTOR_1.throttle = 0
LEFT_MOTOR_2.throttle = 0
LEFT_MOTOR_3.throttle = 0
LEFT_MOTOR_4.throttle = 0

time.sleep(0.05)

RIGHT_MOTOR_1.throttle = None
RIGHT_MOTOR_2.throttle = None
RIGHT_MOTOR_3.throttle = None
RIGHT_MOTOR_4.throttle = None
LEFT_MOTOR_1.throttle = None
LEFT_MOTOR_2.throttle = None
LEFT_MOTOR_3.throttle = None
LEFT_MOTOR_4.throttle = None

# Immediately initialize all motors to stopped state
def initialize_motors():
    """
    Initialize all motors to a guaranteed stopped state
    """
    try:
        # Add delay to ensure I2C bus and hardware are ready
        time.sleep(0.2)
        
        # Force all motors to 0 throttle first
        RIGHT_MOTOR_1.throttle = 0
        RIGHT_MOTOR_2.throttle = 0
        RIGHT_MOTOR_3.throttle = 0
        LEFT_MOTOR_1.throttle = 0
        LEFT_MOTOR_2.throttle = 0
        LEFT_MOTOR_3.throttle = 0
        
        # Longer delay to ensure commands are processed
        time.sleep(0.1)
        
        # Then release control by setting to None
        RIGHT_MOTOR_1.throttle = None
        RIGHT_MOTOR_2.throttle = None
        RIGHT_MOTOR_3.throttle = None
        LEFT_MOTOR_1.throttle = None
        LEFT_MOTOR_2.throttle = None
        LEFT_MOTOR_3.throttle = None
        
        print("Motors initialized to stopped state")
        
    except Exception as e:
        print(f"Warning: Motor initialization error: {e}")
        # Try individual motor stops as fallback
        try:
            for motor in [RIGHT_MOTOR_1, RIGHT_MOTOR_2, RIGHT_MOTOR_3, LEFT_MOTOR_1, LEFT_MOTOR_2, LEFT_MOTOR_3]:
                try:
                    motor.throttle = 0
                    motor.throttle = None
                except:
                    pass
        except:
            pass

# Initialize motors immediately when module is imported
initialize_motors()

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
    Stop the robot with enhanced stopping procedure
    """
    try:
        # Set throttle to 0 first for all motors
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
        
        # Additional safety delay
        time.sleep(0.05)
        
    except Exception as e:
        # If normal stop fails, try to force stop each motor individually
        print(f"Error during stop: {e}")
        try:
            for motor in [RIGHT_MOTOR_1, RIGHT_MOTOR_2, RIGHT_MOTOR_3, LEFT_MOTOR_1, LEFT_MOTOR_2, LEFT_MOTOR_3]:
                motor.throttle = 0
                motor.throttle = None
        except Exception as e2:
            print(f"Critical error in motor stop: {e2}")


# Global variables to track last motor command and timing to prevent rapid switching
_last_motor_command = None
_last_command_time = 0

def combined_mecanum_control(forward_back, left_right, rotation, intensity=1.0):
    """
    Advanced mecanum wheel control that combines forward/back, strafe, and rotation
    simultaneously for complex navigation maneuvers.
    
    Args:
        forward_back: -1.0 to 1.0 (negative=backward, positive=forward)
        left_right: -1.0 to 1.0 (negative=strafe left, positive=strafe right)  
        rotation: -1.0 to 1.0 (negative=rotate left, positive=rotate right)
        intensity: Overall speed multiplier (0.0 to 1.0)
    """
    global _last_motor_command, _last_command_time
    import time
    
    # Use same deadzone values as joystick input manager
    ANALOG_DEADZONE = 0.15
    TRIGGER_DEADZONE = 0.1
    # Calculate individual wheel speeds using mecanum mathematics
    # Each wheel contributes to forward/back, strafe, and rotation differently
    
    # Mecanum wheel equations for 6-wheel robot:
    # - Front/rear wheels handle forward/back + rotation
    # - Middle wheels handle pure strafe (omni wheels at 90 degrees)
    
    # Scale intensity
    fb = forward_back * intensity
    lr = left_right * intensity  
    rot = rotation * intensity
    
    # Calculate motor speeds for each side
    # Left side motors: forward + rotation (opposite of right for rotation)
    left_speed = fb + rot
    
    # Right side motors: forward - rotation  
    right_speed = fb - rot
    
    # Check if all inputs are within deadzone
    if abs(fb) < ANALOG_DEADZONE and abs(lr) < ANALOG_DEADZONE and abs(rot) < TRIGGER_DEADZONE:
        stop()
        return
    
    # True mecanum wheel control - calculate individual wheel speeds
    # For a 6-wheel mecanum robot with middle omni wheels:
    # Front/rear wheels: contribute to forward/back and rotation
    # Middle wheels: contribute primarily to strafing
    
    # Calculate combined motion vectors
    combined_fb = fb
    combined_lr = lr 
    combined_rot = rot
    
    # Apply mecanum mathematics for each wheel group
    # Front wheels (normal mecanum): fb + rot
    front_left = combined_fb + combined_rot
    front_right = combined_fb - combined_rot
    
    # Rear wheels (normal mecanum): fb + rot  
    rear_left = combined_fb + combined_rot
    rear_right = combined_fb - combined_rot
    
    # Middle wheels (omni wheels for pure strafe)
    middle_left = combined_lr
    middle_right = -combined_lr  # Opposite for strafing
    
    # Determine the most significant combined motion
    max_magnitude = max(abs(combined_fb), abs(combined_lr), abs(combined_rot))
    
    if max_magnitude < 0.1:  # Very small inputs, just stop
        stop()
        return
    
    # Apply stability timing to prevent clicking during transitions
    current_time = time.time()
    motion_type = None
    
    # Determine primary motion for stability tracking
    if abs(combined_lr) > abs(combined_fb) * 1.2 and abs(combined_lr) > abs(combined_rot) * 1.2:
        motion_type = 'strafe'
    elif abs(combined_rot) > abs(combined_fb) * 1.2 and abs(combined_rot) > abs(combined_lr) * 1.2:
        motion_type = 'rotate'  
    elif abs(combined_fb) > 0.1:
        motion_type = 'forward_back'
    else:
        motion_type = 'combined'
    
    # Apply stability delay for major motion type changes
    should_execute = True
    if (_last_motor_command and motion_type != _last_motor_command and 
        motion_type != 'combined' and _last_motor_command != 'combined'):
        time_since_last = current_time - _last_command_time
        if time_since_last < 0.025:  # 25ms minimum between major motion changes
            return  # Skip to prevent clicking
    
    # Execute the motion with better combined control
    # Use a softer threshold for dominance to allow more blending
    dominant_threshold = 1.1  # Only 10% difference needed for dominance
    
    if abs(combined_lr) > abs(combined_fb) * dominant_threshold and abs(combined_lr) > abs(combined_rot) * dominant_threshold:
        # Strafing is clearly dominant
        if combined_lr < 0:
            move_left(min(abs(combined_lr), 1.0))
        else:
            move_right(min(abs(combined_lr), 1.0))
    elif abs(combined_rot) > abs(combined_fb) * dominant_threshold and abs(combined_rot) > abs(combined_lr) * dominant_threshold:
        # Rotation is clearly dominant
        if combined_rot < 0:
            turn_left(min(abs(combined_rot), 1.0))
        else:
            turn_right(min(abs(combined_rot), 1.0))
    elif abs(combined_fb) > abs(combined_lr) * dominant_threshold and abs(combined_fb) > abs(combined_rot) * dominant_threshold:
        # Forward/back is clearly dominant
        if combined_fb > 0:
            move_forward(min(abs(combined_fb), 1.0))
        else:
            move_backward(min(abs(combined_fb), 1.0))
    else:
        # Mixed motion - choose the strongest but allow quicker transitions
        strongest = max(abs(combined_fb), abs(combined_lr), abs(combined_rot))
        if strongest == abs(combined_lr):
            if combined_lr < 0:
                move_left(min(abs(combined_lr) * 0.8, 1.0))  # Slightly reduced for smoother blending
            else:
                move_right(min(abs(combined_lr) * 0.8, 1.0))
        elif strongest == abs(combined_rot):
            if combined_rot < 0:
                turn_left(min(abs(combined_rot) * 0.8, 1.0))
            else:
                turn_right(min(abs(combined_rot) * 0.8, 1.0))
        else:
            if combined_fb > 0:
                move_forward(min(abs(combined_fb) * 0.8, 1.0))
            else:
                move_backward(min(abs(combined_fb) * 0.8, 1.0))
    
    _last_motor_command = motion_type
    _last_command_time = current_time


def joystick_motor_handler(action, **kwargs):
    """
    Handle joystick input for direct motor control - zero latency
    """
    if action == 'combined':
        # New combined control mode
        forward_back = kwargs.get('forward_back', 0.0)
        left_right = kwargs.get('left_right', 0.0)
        rotation = kwargs.get('rotation', 0.0)
        intensity = kwargs.get('intensity', 1.0)
        
        combined_mecanum_control(forward_back, left_right, rotation, intensity)
        
    elif action == 'rotate':
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
