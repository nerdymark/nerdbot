#!/usr/bin/env python3
"""
Interactive demo for the light_bar module
"""
import sys
import time
sys.path.insert(0, '/home/mark/nerdbot-backend')

from light_bar.light_bar import light_bar, rainbow_cycle, loading_bar, vu_meter

def show_menu():
    print("\n=== Light Bar Interactive Demo ===")
    print("1. Rainbow Cycle")
    print("2. Loading Bar")
    print("3. VU Meter Demo")
    print("4. Set Single Color")
    print("5. Clear Strip")
    print("6. Exit")
    print("==================================")

def main():
    while True:
        show_menu()
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            print("Running rainbow cycle (press Ctrl+C to stop)...")
            try:
                while True:
                    rainbow_cycle(light_bar, delay=0.2)
            except KeyboardInterrupt:
                print("\nStopped rainbow cycle")
                
        elif choice == '2':
            print("Running loading bar...")
            loading_bar(light_bar)
            
        elif choice == '3':
            print("VU Meter demo (press Ctrl+C to stop)...")
            try:
                import random
                while True:
                    volume = random.randint(0, 100)
                    vu_meter(light_bar, volume)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopped VU meter")
                
        elif choice == '4':
            try:
                r = int(input("Enter Red value (0-255): "))
                g = int(input("Enter Green value (0-255): "))
                b = int(input("Enter Blue value (0-255): "))
                light_bar.fill_strip(r, g, b)
                light_bar.update_strip()
                print(f"Set color to RGB({r}, {g}, {b})")
            except ValueError:
                print("Invalid input! Please enter numbers 0-255")
                
        elif choice == '5':
            light_bar.clear_strip()
            light_bar.update_strip()
            print("Strip cleared")
            
        elif choice == '6':
            light_bar.clear_strip()
            light_bar.update_strip()
            print("Goodbye!")
            break
            
        else:
            print("Invalid option! Please select 1-6")

if __name__ == "__main__":
    try:
        print("Starting Light Bar Interactive Demo...")
        print(f"Number of LEDs: {light_bar.num_leds}")
        main()
    except Exception as e:
        print(f"Error: {e}")
        light_bar.clear_strip()
        light_bar.update_strip()