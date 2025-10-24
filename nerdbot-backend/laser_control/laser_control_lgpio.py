#!/usr/bin/env python3
"""
Laser control using lgpio directly for Raspberry Pi 5
This version directly controls GPIO 21 without gpiozero abstraction
"""
import logging
import os

# Check if we're on a Raspberry Pi 5
IS_PI5 = os.path.exists('/dev/gpiochip0') and os.path.exists('/proc/device-tree/model')
if IS_PI5:
    try:
        with open('/proc/device-tree/model', 'r') as f:
            IS_PI5 = 'Raspberry Pi 5' in f.read()
    except:
        IS_PI5 = False

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False
    logging.warning("lgpio not available - laser control will use simulation mode")

LASER_PIN = 21

class LaserControl:
    """
    Direct laser control using lgpio for Raspberry Pi 5
    """
    def __init__(self):
        self.is_active = False
        self.gpio_available = LGPIO_AVAILABLE and IS_PI5
        self.chip = None
        self.pin_claimed = False
        
        if self.gpio_available:
            try:
                # Open GPIO chip
                self.chip = lgpio.gpiochip_open(0)
                
                # Try to free the pin first if it's already in use
                try:
                    lgpio.gpio_free(self.chip, LASER_PIN)
                    logging.info(f"Freed existing claim on GPIO {LASER_PIN}")
                except:
                    pass  # Pin wasn't claimed, that's fine
                
                # Claim the pin as output (start with LOW/OFF)
                lgpio.gpio_claim_output(self.chip, LASER_PIN, 0)
                self.pin_claimed = True
                logging.info(f"Laser control initialized with lgpio on GPIO pin {LASER_PIN}")
                
            except Exception as e:
                logging.error(f"Failed to initialize lgpio for laser control: {e}")
                self.gpio_available = False
                self.pin_claimed = False
                if self.chip is not None:
                    try:
                        lgpio.gpiochip_close(self.chip)
                    except:
                        pass
                    self.chip = None
        else:
            logging.info("Laser control initialized in simulation mode (not Pi5 or lgpio unavailable)")

    def activate_laser(self):
        """
        Activate the laser by setting GPIO HIGH
        """
        if self.gpio_available and self.chip is not None and self.pin_claimed:
            try:
                lgpio.gpio_write(self.chip, LASER_PIN, 1)
                logging.info("Laser activated via GPIO (HIGH)")
                self.is_active = True
            except Exception as e:
                logging.error(f"Failed to activate laser: {e}")
                return False
        else:
            logging.info("Simulating laser activation")
            self.is_active = True
        return True

    def deactivate_laser(self):
        """
        Deactivate the laser by setting GPIO LOW
        """
        if self.gpio_available and self.chip is not None and self.pin_claimed:
            try:
                lgpio.gpio_write(self.chip, LASER_PIN, 0)
                logging.info("Laser deactivated via GPIO (LOW)")
                self.is_active = False
            except Exception as e:
                logging.error(f"Failed to deactivate laser: {e}")
                return False
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
        if self.gpio_available and self.chip is not None:
            try:
                if self.pin_claimed:
                    # Ensure laser is off before cleanup
                    lgpio.gpio_write(self.chip, LASER_PIN, 0)
                    lgpio.gpio_free(self.chip, LASER_PIN)
                    logging.info(f"Freed GPIO {LASER_PIN}")
                
                lgpio.gpiochip_close(self.chip)
                logging.info("Laser GPIO chip closed")
            except Exception as e:
                logging.warning(f"Error during laser GPIO cleanup: {e}")
        else:
            logging.info("Simulating GPIO cleanup")
        
        self.chip = None
        self.pin_claimed = False
        self.is_active = False

if __name__ == "__main__":
    # Test the laser control
    logging.basicConfig(level=logging.INFO)
    
    print("Testing laser control with lgpio...")
    laser = LaserControl()
    
    if laser.gpio_available:
        print("GPIO available, testing real hardware control")
    else:
        print("GPIO not available, running in simulation mode")
    
    print("\nActivating laser...")
    laser.activate_laser()
    input("Press Enter to deactivate the laser")
    
    print("Deactivating laser...")
    laser.deactivate_laser()
    
    print("Cleaning up...")
    laser.cleanup()
    print("Test complete!")