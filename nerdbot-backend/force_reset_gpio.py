#!/usr/bin/env python3
"""
Force reset GPIO 21 using lgpio library (Raspberry Pi 5 compatible)
"""
import sys
import time

def force_reset_lgpio():
    """Force reset using lgpio which is used on Pi 5"""
    try:
        import lgpio
        print("Attempting to reset GPIO 21 using lgpio...")
        
        # Open the GPIO chip
        chip = lgpio.gpiochip_open(0)
        
        # Try to claim the GPIO as output
        try:
            lgpio.gpio_claim_output(chip, 21, 0)
            print("Successfully claimed GPIO 21")
            time.sleep(0.1)
        except:
            print("GPIO 21 already claimed, attempting to free it...")
        
        # Write low value
        try:
            lgpio.gpio_write(chip, 21, 0)
            print("Set GPIO 21 to LOW")
        except:
            pass
        
        # Free the GPIO
        try:
            lgpio.gpio_free(chip, 21)
            print("Freed GPIO 21")
        except:
            pass
        
        # Close the chip
        lgpio.gpiochip_close(chip)
        print("Closed GPIO chip")
        
        return True
    except Exception as e:
        print(f"lgpio reset failed: {e}")
        return False

def reset_via_gpiod():
    """Reset using libgpiod tools"""
    import subprocess
    try:
        print("Attempting to reset GPIO 21 using gpiod...")
        
        # First try to release it
        result = subprocess.run(
            ["sudo", "gpioget", "gpiochip0", "21"],
            capture_output=True,
            text=True,
            timeout=1
        )
        print(f"Current GPIO 21 state: {result.stdout.strip()}")
        
        # Set to output low briefly
        subprocess.run(
            ["sudo", "timeout", "0.1", "gpioset", "gpiochip0", "21=0"],
            capture_output=True,
            text=True
        )
        print("Reset GPIO 21 to LOW")
        
        return True
    except Exception as e:
        print(f"gpiod reset failed: {e}")
        return False

def main():
    """Main function to force reset GPIO 21"""
    print("=== Force GPIO 21 Reset for Raspberry Pi 5 ===")
    
    success = False
    
    # Try lgpio (Pi 5 compatible)
    if force_reset_lgpio():
        success = True
    
    # Try gpiod as fallback
    elif reset_via_gpiod():
        success = True
    
    if success:
        print("\n✅ GPIO 21 reset attempt completed")
        print("The pin should now be available for use")
        return 0
    else:
        print("\n⚠️  Could not reset GPIO 21")
        print("The pin may still be in use by another process")
        print("Try restarting the Raspberry Pi if the issue persists")
        return 1

if __name__ == "__main__":
    sys.exit(main())