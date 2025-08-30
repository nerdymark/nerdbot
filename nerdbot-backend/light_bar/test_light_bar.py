#!/usr/bin/env python3
"""
Test script for the light_bar module
"""
import sys
import time
sys.path.insert(0, '/home/mark/nerdbot-backend')

from light_bar.light_bar import light_bar, rainbow_cycle, loading_bar, vu_meter

def test_basic_functions():
    print("Testing light_bar module...")
    
    print("\n1. Testing clear strip...")
    light_bar.clear_strip()
    light_bar.update_strip()
    time.sleep(1)
    
    print("\n2. Testing single LED (LED 0 - Red)...")
    light_bar.set_led_color(0, 255, 0, 0)
    light_bar.update_strip()
    time.sleep(2)
    
    print("\n3. Testing fill strip (Blue)...")
    light_bar.fill_strip(0, 0, 255)
    light_bar.update_strip()
    time.sleep(2)
    
    print("\n4. Testing rainbow cycle...")
    rainbow_cycle(light_bar, delay=0.5)
    
    print("\n5. Testing loading bar...")
    loading_bar(light_bar)
    
    print("\n6. Testing VU meter at different levels...")
    volumes = [10, 30, 50, 70, 90, 100]
    for vol in volumes:
        print(f"   Volume: {vol}%")
        vu_meter(light_bar, vol)
        time.sleep(1)
    
    print("\n7. Clearing strip...")
    light_bar.clear_strip()
    light_bar.update_strip()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    try:
        test_basic_functions()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()