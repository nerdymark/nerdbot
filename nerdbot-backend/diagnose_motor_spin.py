#!/usr/bin/env python3
"""
Diagnostic script to identify which motor spins on initialization
"""
import time
from adafruit_motorkit import MotorKit

print("Motor Spin Diagnostic Tool")
print("=" * 40)

# Test RIGHT_KIT motors
print("\nTesting RIGHT_KIT (address 0x61)...")
try:
    right_kit = MotorKit(address=0x61)
    print("RIGHT_KIT initialized")
    
    # Test each motor individually
    for i in range(1, 5):
        try:
            motor = getattr(right_kit, f'motor{i}')
            print(f"  Testing motor{i}...")
            
            # Check if motor is already spinning
            print(f"    Setting throttle to 0...")
            motor.throttle = 0
            time.sleep(0.1)
            
            print(f"    Setting throttle to None...")
            motor.throttle = None
            time.sleep(0.1)
            
            print(f"    Motor{i} stopped successfully")
        except Exception as e:
            print(f"    Motor{i} error: {e}")
            
except Exception as e:
    print(f"RIGHT_KIT initialization failed: {e}")

print("\nTesting LEFT_KIT (address 0x60)...")
try:
    left_kit = MotorKit(address=0x60)
    print("LEFT_KIT initialized")
    
    # Test each motor individually
    for i in range(1, 5):
        try:
            motor = getattr(left_kit, f'motor{i}')
            print(f"  Testing motor{i}...")
            
            # Check if motor is already spinning
            print(f"    Setting throttle to 0...")
            motor.throttle = 0
            time.sleep(0.1)
            
            print(f"    Setting throttle to None...")
            motor.throttle = None
            time.sleep(0.1)
            
            print(f"    Motor{i} stopped successfully")
        except Exception as e:
            print(f"    Motor{i} error: {e}")
            
except Exception as e:
    print(f"LEFT_KIT initialization failed: {e}")

print("\n" + "=" * 40)
print("Diagnostic complete!")
print("If any motor was spinning, it should have stopped during this test.")
print("Note which motor was spinning when the script started.")