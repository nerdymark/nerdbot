# pylint: disable=no-member
"""
Activate / Deactivate a laser connected to a relay module
Relay module is connected to the Raspberry Pi GPIO pins
"""
import logging
import os

# Check if we're on a Raspberry Pi 5 (uses lgpio)
IS_PI5 = os.path.exists('/dev/gpiochip0') and os.path.exists('/proc/device-tree/model')
if IS_PI5:
    try:
        with open('/proc/device-tree/model', 'r') as f:
            IS_PI5 = 'Raspberry Pi 5' in f.read()
    except:
        IS_PI5 = False

try:
    if IS_PI5:
        # Use lgpio for Pi 5
        from gpiozero import Device
        from gpiozero.pins.lgpio import LGPIOFactory
        Device.pin_factory = LGPIOFactory()
    from gpiozero import LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("gpiozero not available - laser control will use simulation mode")


LASER_PIN = 21

class LaserControl:
    """
    Class to control the laser
    """
    def __init__(self):
        self.is_active = False
        self.gpio_available = GPIO_AVAILABLE
        self.laser_relay = None
        
        if self.gpio_available:
            try:
                # Use gpiozero LED class for relay control (LED class works for any digital output)
                self.laser_relay = LED(LASER_PIN)
                # Ensure it starts in OFF state
                self.laser_relay.off()
                logging.info(f"Laser control initialized with gpiozero on GPIO pin {LASER_PIN}")
                if IS_PI5:
                    logging.info("Using lgpio backend for Raspberry Pi 5")
            except Exception as e:
                logging.error(f"Failed to initialize gpiozero for laser control: {e}")
                logging.error(f"If GPIO {LASER_PIN} is busy, try: sudo python3 force_reset_gpio.py")
                self.gpio_available = False
                self.laser_relay = None
        else:
            logging.info("Laser control initialized in simulation mode")

    def activate_laser(self):
        """
        Activate the laser by turning on the relay
        """
        if self.gpio_available and self.laser_relay:
            self.laser_relay.on()
            logging.info("Laser activated via GPIO")
        else:
            logging.info("Simulating laser activation")
        self.is_active = True
        return True

    def deactivate_laser(self):
        """
        Deactivate the laser by turning off the relay
        """
        if self.gpio_available and self.laser_relay:
            self.laser_relay.off()
            logging.info("Laser deactivated via GPIO")
        else:
            logging.info("Simulating laser deactivation")
        self.is_active = False
        return True

    def toggle_laser(self):
        """
        Toggle laser on/off
        """
        if self.is_active:
            return self.deactivate_laser()
        else:
            return self.activate_laser()

    def is_laser_active(self):
        """
        Check if laser is currently active
        """
        return self.is_active

    def cleanup(self):
        """
        Cleanup the GPIO resources
        """
        if self.gpio_available and self.laser_relay:
            try:
                self.laser_relay.off()  # Ensure laser is off before cleanup
                self.laser_relay.close()  # gpiozero cleanup
                logging.info("Laser GPIO resources cleaned up")
            except Exception as e:
                logging.warning(f"Error during laser GPIO cleanup: {e}")
        else:
            logging.info("Simulating GPIO cleanup")

if __name__ == "__main__":
    laser = LaserControl()
    laser.activate_laser()
    input("Press Enter to deactivate the laser")
    laser.deactivate_laser()
    laser.cleanup()
    print("Laser deactivated")
