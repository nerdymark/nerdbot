#!/usr/bin/env python3
"""
Test script to verify laser GPIO control
"""
import time
import logging

try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
    print("gpiozero imported successfully")
except ImportError as e:
    GPIO_AVAILABLE = False
    print(f"gpiozero not available: {e}")
    exit(1)

LASER_PIN = 21

def test_laser_gpio():
    """Test GPIO control for laser relay"""
    print(f"Testing GPIO pin {LASER_PIN} for laser control...")
    
    try:
        # Initialize LED (works for any digital output including relays)
        laser_relay = LED(LASER_PIN)
        print("GPIO initialized successfully")
        
        # Test on/off cycle
        for i in range(3):
            print(f"Cycle {i+1}/3:")
            print("  Turning relay ON...")
            laser_relay.on()
            time.sleep(2)
            
            print("  Turning relay OFF...")
            laser_relay.off()
            time.sleep(2)
        
        print("Cleaning up GPIO...")
        laser_relay.close()
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error during GPIO test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Enable logging to see detailed info
    logging.basicConfig(level=logging.INFO)
    
    print("Starting laser GPIO test...")
    success = test_laser_gpio()
    
    if success:
        print("✓ GPIO test passed")
    else:
        print("✗ GPIO test failed")