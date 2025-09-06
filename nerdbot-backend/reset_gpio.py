#!/usr/bin/env python3
"""
Reset GPIO pins - specifically GPIO 21 used by laser control
"""
import sys
import time

def reset_gpio_gpiozero():
    """Reset GPIO using gpiozero library"""
    try:
        from gpiozero import LED
        print("Resetting GPIO 21 using gpiozero...")
        
        # Create LED object and ensure it's off
        laser = LED(21)
        laser.off()
        time.sleep(0.1)
        
        # Properly close and cleanup
        laser.close()
        print("GPIO 21 reset successfully using gpiozero")
        return True
    except Exception as e:
        print(f"gpiozero reset failed: {e}")
        return False

def reset_gpio_rpigpio():
    """Reset GPIO using RPi.GPIO library as fallback"""
    try:
        import RPi.GPIO as GPIO
        print("Resetting GPIO 21 using RPi.GPIO...")
        
        # Use BCM numbering
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup pin as output and set to LOW
        GPIO.setup(21, GPIO.OUT)
        GPIO.output(21, GPIO.LOW)
        time.sleep(0.1)
        
        # Cleanup
        GPIO.cleanup(21)
        print("GPIO 21 reset successfully using RPi.GPIO")
        return True
    except Exception as e:
        print(f"RPi.GPIO reset failed: {e}")
        return False

def reset_gpio_sysfs():
    """Reset GPIO using sysfs interface as last resort"""
    try:
        import os
        print("Resetting GPIO 21 using sysfs...")
        
        gpio_num = 21
        gpio_path = f"/sys/class/gpio/gpio{gpio_num}"
        
        # Unexport if already exported
        if os.path.exists(gpio_path):
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(gpio_num))
            time.sleep(0.1)
        
        # Export GPIO
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(gpio_num))
        time.sleep(0.1)
        
        # Set direction to out and value to 0
        if os.path.exists(gpio_path):
            with open(f"{gpio_path}/direction", "w") as f:
                f.write("out")
            with open(f"{gpio_path}/value", "w") as f:
                f.write("0")
            time.sleep(0.1)
            
            # Unexport to clean up
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(gpio_num))
        
        print("GPIO 21 reset successfully using sysfs")
        return True
    except Exception as e:
        print(f"sysfs reset failed: {e}")
        return False

def main():
    """Main function to reset GPIO 21"""
    print("=== GPIO 21 Reset Utility ===")
    print("Attempting to reset GPIO 21 (laser control pin)...")
    
    # Try each method in order
    success = False
    
    # Try gpiozero first (preferred)
    if reset_gpio_gpiozero():
        success = True
    
    # Try RPi.GPIO as fallback
    elif reset_gpio_rpigpio():
        success = True
    
    # Try sysfs as last resort
    elif reset_gpio_sysfs():
        success = True
    
    if success:
        print("\n✅ GPIO 21 has been successfully reset and is now available")
        print("You can now restart the Flask service or use the laser control")
        return 0
    else:
        print("\n❌ Failed to reset GPIO 21")
        print("You may need to:")
        print("1. Stop the Flask service: sudo systemctl stop nerdbot-flask")
        print("2. Run this script with sudo: sudo python3 reset_gpio.py")
        print("3. Restart the Flask service: sudo systemctl start nerdbot-flask")
        return 1

if __name__ == "__main__":
    sys.exit(main())