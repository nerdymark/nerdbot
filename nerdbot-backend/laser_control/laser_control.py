# pylint: disable=no-member
"""
Activate / Deactivate a laser connected to a relay module
Relay module is connected to the Raspberry Pi GPIO pins
"""
from RPi import GPIO


LASER_PIN = 21

class LaserControl:
    """
    Class to control the laser
    """
    def __init__(self):
        GPIO.setmode(GPIO.BCM)  # pylint: disable=no-member
        GPIO.setup(LASER_PIN, GPIO.OUT)

    def activate_laser(self):
        """
        Activate the laser by setting the GPIO pin to HIGH
        """
        GPIO.output(LASER_PIN, GPIO.HIGH)

    def deactivate_laser(self):
        """
        Deactivate the laser by setting the GPIO pin to LOW
        """
        GPIO.output(LASER_PIN, GPIO.LOW)

    def cleanup(self):
        """
        Cleanup the GPIO pins
        """
        GPIO.cleanup()

if __name__ == "__main__":
    laser = LaserControl()
    laser.activate_laser()
    input("Press Enter to deactivate the laser")
    laser.deactivate_laser()
    laser.cleanup()
    print("Laser deactivated")
