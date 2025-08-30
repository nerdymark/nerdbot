"""
Test script for optimized tracker
"""
import sys
sys.path.append('.')
from tracking import OptimizedTracker

# Mock servo class for testing
class MockServos:
    def __init__(self):
        self.pan_angle = 90
        self.tilt_angle = 150
        
    def pan_to(self, angle):
        self.pan_angle = angle
        print(f"Pan to: {angle}")
        
    def tilt_to(self, angle):  
        self.tilt_angle = angle
        print(f"Tilt to: {angle}")
        
    def reset_servos(self):
        self.pan_angle = 90
        self.tilt_angle = 150
        print("Servos reset")

# Test the tracker
mock_servos = MockServos()
tracker = OptimizedTracker(mock_servos)

print("Testing optimized tracker...")

# Test case 1: Object in center (should not move)
print("\n1. Object in center:")
bbox = (300, 230, 40, 20)  # Center of 640x480 frame
tracker.track_object(bbox)

# Test case 2: Object far left (should pan right)
print("\n2. Object far left:")
bbox = (50, 240, 40, 20)
tracker.track_object(bbox)

# Test case 3: Object far right (should pan left)
print("\n3. Object far right:")
bbox = (550, 240, 40, 20)
tracker.track_object(bbox)

# Test case 4: Object at top (should tilt up)
print("\n4. Object at top:")
bbox = (320, 50, 40, 20)
tracker.track_object(bbox)

# Test case 5: Object at bottom (should tilt down)
print("\n5. Object at bottom:")
bbox = (320, 400, 40, 20)
tracker.track_object(bbox)

print("\nTracker test completed!")