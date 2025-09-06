#!/usr/bin/env python3
"""
Test brightness levels for light bar
"""

import sys
import time

# Add path for imports
sys.path.insert(0, '/home/mark/nerdbot-backend')

# Clear any cached modules
for module in list(sys.modules.keys()):
    if module.startswith('light_bar'):
        del sys.modules[module]

from light_bar.wled_controller import WLEDController

def test_brightness():
    controller = WLEDController()
    
    print("=== Light Bar Brightness Test ===\n")
    
    print("Testing with solid colors to see brightness difference...\n")
    
    print("1. RED at 10% brightness (normal operation)")
    controller.set_color(255, 0, 0, brightness=26)  # 10%
    time.sleep(3)
    
    print("2. RED at 100% brightness (attention-getting)")
    controller.set_color(255, 0, 0, brightness=255)  # 100%
    time.sleep(3)
    
    print("3. BLUE at 10% brightness")
    controller.set_color(0, 0, 255, brightness=26)  # 10%
    time.sleep(3)
    
    print("4. BLUE at 100% brightness") 
    controller.set_color(0, 0, 255, brightness=255)  # 100%
    time.sleep(3)
    
    print("5. Testing robot states with proper brightness...")
    
    print("   Idle (10% brightness)")
    controller.idle()
    time.sleep(2)
    
    print("   Detection alert (100% brightness)")
    controller.detection()
    time.sleep(2)
    
    print("   Dimming to normal")
    controller.dim_to_normal()
    time.sleep(2)
    
    print("   Moving (10% brightness)")
    controller.moving()
    time.sleep(2)
    
    print("6. Turning off")
    controller.turn_off()
    
    print("\n=== Test Complete ===")
    print("Normal states now use 10% brightness")
    print("Alert states use 100% brightness initially, then auto-dim")

if __name__ == "__main__":
    test_brightness()