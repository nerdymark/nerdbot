#!/usr/bin/env python3
"""
GPIO utility functions for NerdBot
Handles GPIO reset and status checking for Raspberry Pi 5
"""
import sys
import subprocess
import time

def check_gpio_status(pin=21):
    """Check the status of a GPIO pin"""
    try:
        result = subprocess.run(
            ["gpioinfo", "gpiochip0"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if f"line {pin:>3}:" in line:
                return line.strip()
        return f"GPIO {pin} not found"
    except Exception as e:
        return f"Error checking GPIO {pin}: {e}"

def reset_gpio(pin=21):
    """Reset a GPIO pin using the force reset method"""
    try:
        # Import and use the reset function
        import lgpio
        chip = lgpio.gpiochip_open(0)
        
        try:
            # Try to free the GPIO if it's claimed
            lgpio.gpio_free(chip, pin)
            print(f"Freed GPIO {pin}")
        except:
            pass
        
        lgpio.gpiochip_close(chip)
        return True
    except Exception as e:
        print(f"Reset failed: {e}")
        return False

def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python3 gpio_utils.py [status|reset] [pin_number]")
        print("Examples:")
        print("  python3 gpio_utils.py status 21")
        print("  python3 gpio_utils.py reset 21") 
        return 1
    
    command = sys.argv[1]
    pin = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    
    if command == "status":
        print(f"GPIO {pin} status:")
        print(check_gpio_status(pin))
    elif command == "reset":
        print(f"Resetting GPIO {pin}...")
        if reset_gpio(pin):
            print("✅ Reset completed")
            time.sleep(0.1)
            print("New status:")
            print(check_gpio_status(pin))
        else:
            print("❌ Reset failed")
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())