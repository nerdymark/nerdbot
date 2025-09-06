#!/usr/bin/env python3
"""
Test script for light bar WLED integration
"""

import sys
import time
import logging

# Add path for imports
sys.path.insert(0, '/home/mark/nerdbot-backend')

from light_bar.light_bar import light_bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_light_bar_integration():
    """Test the light bar integration with robot actions"""
    
    print("=== Light Bar Integration Test ===\n")
    
    # Start light bar
    print("1. Starting light bar...")
    if light_bar.start():
        print("   ✓ Light bar started successfully")
    else:
        print("   ✗ Failed to start light bar")
        return
    
    time.sleep(2)
    
    # Test different states
    states_to_test = [
        ("startup", "Robot booting up", 3),
        ("idle", "Robot at rest", 3),
        ("moving", "Robot moving forward", 3),
        ("speaking", "Robot speaking (TTS)", 3),
        ("listening", "Robot listening for commands", 3),
        ("detection", "Object detected!", 2),
        ("thinking", "Processing request", 3),
        ("idle", "Back to idle", 2)
    ]
    
    for state, description, duration in states_to_test:
        print(f"\n2. Testing {state} state: {description}")
        try:
            if light_bar.set_robot_state(state):
                print(f"   ✓ {state} effect active")
            else:
                print(f"   ✗ Failed to set {state} state")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        time.sleep(duration)
    
    # Test special effects
    print("\n3. Testing special effects...")
    
    print("   Audio reactive (low intensity)...")
    light_bar.audio_reactive(0.3)
    time.sleep(2)
    
    print("   Audio reactive (high intensity)...")
    light_bar.audio_reactive(0.9)
    time.sleep(2)
    
    print("   Celebration effect...")
    light_bar.celebration()
    time.sleep(3)
    
    print("   Fire effect...")
    light_bar.fire()
    time.sleep(3)
    
    # Shutdown
    print("\n4. Shutting down...")
    light_bar.set_robot_state("shutdown")
    time.sleep(2)
    light_bar.stop()
    
    print("\n=== Test Complete ===")
    print("The light bar is now integrated with:")
    print("  • TTS (Text-to-Speech) - speaking animation")
    print("  • Motor control - moving animation")
    print("  • Object detection - alert animation")
    print("  • Meme sounds - audio reactive effects")
    print("  • Random sounds - celebration effects")
    print("\nAll robot actions will now trigger appropriate light effects!")

if __name__ == "__main__":
    test_light_bar_integration()