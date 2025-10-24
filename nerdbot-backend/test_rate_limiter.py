#!/usr/bin/env python3
"""
Test script to verify WLED rate limiter is working properly
"""

import time
import sys
import os
sys.path.append('/home/mark/nerdbot-backend')

from light_bar.light_bar import light_bar

def test_rate_limiter():
    """Test the WLED rate limiter functionality"""
    print("Testing WLED Rate Limiter...")

    if not light_bar.use_wled:
        print("WLED not available, skipping test")
        return

    # Check if rate limiting is enabled
    if hasattr(light_bar.controller, 'get_queue_status'):
        status = light_bar.controller.get_queue_status()
        print(f"Initial queue status: {status}")

    print("\n1. Testing rapid fire commands (should be rate limited)...")
    start_time = time.time()

    # Send 20 rapid commands
    for i in range(20):
        print(f"  Sending command {i+1}/20: rainbow", end='')
        result = light_bar.rainbow()
        print(f" -> {'OK' if result else 'FAIL'}")
        # No delay - let the rate limiter handle it

    duration = time.time() - start_time
    print(f"  Completed 20 commands in {duration:.2f} seconds")

    if hasattr(light_bar.controller, 'get_queue_status'):
        status = light_bar.controller.get_queue_status()
        print(f"  Queue status after rapid fire: {status}")

    print("\n2. Testing normal pace commands...")
    start_time = time.time()

    # Send commands with proper delays
    commands = ['red', 'green', 'blue', 'white', 'rainbow']
    for i, cmd in enumerate(commands):
        print(f"  Sending command {i+1}/{len(commands)}: {cmd}", end='')
        result = getattr(light_bar, cmd)()
        print(f" -> {'OK' if result else 'FAIL'}")
        time.sleep(0.2)  # 200ms between commands

    duration = time.time() - start_time
    print(f"  Completed {len(commands)} commands in {duration:.2f} seconds")

    if hasattr(light_bar.controller, 'get_queue_status'):
        status = light_bar.controller.get_queue_status()
        print(f"  Final queue status: {status}")

    print("\n3. Testing duplicate command throttling...")
    start_time = time.time()

    # Send the same command multiple times rapidly
    for i in range(10):
        print(f"  Duplicate command {i+1}/10: idle", end='')
        result = light_bar.idle_animation()
        print(f" -> {'OK' if result else 'FAIL'}")

    duration = time.time() - start_time
    print(f"  Completed 10 duplicate commands in {duration:.2f} seconds")
    print("  (Should be very fast due to throttling)")

    # Wait for queue to clear
    print("\n4. Waiting for queue to process...")
    time.sleep(3)

    if hasattr(light_bar.controller, 'get_queue_status'):
        status = light_bar.controller.get_queue_status()
        print(f"  Final queue status: {status}")

    print("\nRate limiter test completed!")

if __name__ == "__main__":
    test_rate_limiter()