#!/usr/bin/env python3
"""
Debug script to find which module is claiming GPIO 21
"""
import lgpio
import logging

logging.basicConfig(level=logging.INFO)

def check_gpio_21(description):
    """Check if GPIO 21 is available"""
    chip = lgpio.gpiochip_open(0)
    try:
        lgpio.gpio_claim_output(chip, 21, 0)
        print(f"✓ {description}: GPIO 21 is FREE")
        lgpio.gpio_free(chip, 21)
        available = True
    except Exception as e:
        print(f"✗ {description}: GPIO 21 is BUSY - {e}")
        available = False
    lgpio.gpiochip_close(chip)
    return available

print("=== GPIO 21 Conflict Detection ===\n")

check_gpio_21("Initial state")

print("\nImporting modules one by one...")

print("\n1. Importing basic Flask modules...")
from flask import Flask
from flask_restful import Api
from flask_cors import CORS
check_gpio_21("After Flask imports")

print("\n2. Importing motor_control...")
try:
    from motor_control import motors
    check_gpio_21("After motor_control")
except Exception as e:
    print(f"Error importing motor_control: {e}")

print("\n3. Importing servo_control...")
try:
    from servo_control import servos
    check_gpio_21("After servo_control")
except Exception as e:
    print(f"Error importing servo_control: {e}")

print("\n4. Importing x120 battery...")
try:
    from x120 import bat
    check_gpio_21("After x120 battery")
except Exception as e:
    print(f"Error importing x120: {e}")

print("\n5. Importing light_bar...")
try:
    from light_bar.light_bar import light_bar
    check_gpio_21("After light_bar")
except Exception as e:
    print(f"Error importing light_bar: {e}")

print("\n6. Finally importing laser_control...")
try:
    from laser_control.laser_control import LaserControl
    check_gpio_21("After laser_control import")
    
    print("\n7. Creating LaserControl instance...")
    laser = LaserControl()
    print(f"   GPIO available: {laser.gpio_available}")
    print(f"   Pin claimed: {laser.pin_claimed}")
    laser.cleanup()
    
except Exception as e:
    print(f"Error with laser_control: {e}")

print("\n=== Debug complete ===")