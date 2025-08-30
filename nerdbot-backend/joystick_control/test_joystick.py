#!/usr/bin/env python3
"""
Quick test script for joystick functionality
"""

import pygame
import time
import sys

def quick_test():
    """Quick test to verify joystick is detected"""
    pygame.init()
    pygame.joystick.init()
    
    joystick_count = pygame.joystick.get_count()
    print(f"Found {joystick_count} joystick(s)")
    
    if joystick_count == 0:
        print("No joysticks found!")
        return False
        
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Controller: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print(f"Hats: {joystick.get_numhats()}")
    
    print("\nTesting for 5 seconds - move sticks and press buttons...")
    
    start_time = time.time()
    while time.time() - start_time < 5:
        pygame.event.pump()
        
        # Show any non-zero axis values
        for i in range(joystick.get_numaxes()):
            value = joystick.get_axis(i)
            if abs(value) > 0.1:
                print(f"Axis {i}: {value:.3f}")
                
        # Show pressed buttons
        for i in range(joystick.get_numbuttons()):
            if joystick.get_button(i):
                print(f"Button {i} pressed")
                
        # Show hat (D-pad) values
        for i in range(joystick.get_numhats()):
            hat_x, hat_y = joystick.get_hat(i)
            if hat_x != 0 or hat_y != 0:
                print(f"Hat {i}: ({hat_x}, {hat_y})")
                
        time.sleep(0.1)
    
    joystick.quit()
    pygame.quit()
    print("Test complete!")
    return True

if __name__ == "__main__":
    quick_test()