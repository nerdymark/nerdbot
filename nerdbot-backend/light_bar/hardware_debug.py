#!/usr/bin/env python3
"""
Hardware debugging script for LED strip
Tests with different timings and signal patterns to diagnose hardware issues
"""
import sys
import time
import logging
from pi5neo import Pi5Neo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_led_slow():
    """Test single LED with slow updates to check for signal issues"""
    logger.info("=" * 60)
    logger.info("SINGLE LED SLOW TEST - Watch for ANY light activity")
    logger.info("=" * 60)
    
    light_bar = Pi5Neo('/dev/spidev0.0', 8, 800)
    
    logger.info("Setting LED 0 to BRIGHT RED (max brightness)")
    light_bar.clear_strip()
    light_bar.set_led_color(0, 255, 0, 0)  # Maximum red
    light_bar.update_strip()
    time.sleep(3)
    
    logger.info("Setting LED 0 to BRIGHT GREEN (max brightness)")
    light_bar.clear_strip()
    light_bar.set_led_color(0, 0, 255, 0)  # Maximum green
    light_bar.update_strip()
    time.sleep(3)
    
    logger.info("Setting LED 0 to BRIGHT BLUE (max brightness)")
    light_bar.clear_strip()
    light_bar.set_led_color(0, 0, 0, 255)  # Maximum blue
    light_bar.update_strip()
    time.sleep(3)
    
    logger.info("Setting LED 0 to WHITE (all channels max)")
    light_bar.clear_strip()
    light_bar.set_led_color(0, 255, 255, 255)  # Maximum white
    light_bar.update_strip()
    time.sleep(3)
    
    light_bar.clear_strip()
    light_bar.update_strip()

def test_all_leds_max_brightness():
    """Test all LEDs at maximum brightness"""
    logger.info("\n" + "=" * 60)
    logger.info("ALL LEDS MAX BRIGHTNESS TEST")
    logger.info("=" * 60)
    
    light_bar = Pi5Neo('/dev/spidev0.0', 8, 800)
    
    colors = [
        ("BRIGHT RED", (255, 0, 0)),
        ("BRIGHT GREEN", (0, 255, 0)),
        ("BRIGHT BLUE", (0, 0, 255)),
        ("BRIGHT WHITE", (255, 255, 255)),
        ("BRIGHT YELLOW", (255, 255, 0)),
        ("BRIGHT CYAN", (0, 255, 255)),
        ("BRIGHT MAGENTA", (255, 0, 255))
    ]
    
    for name, (r, g, b) in colors:
        logger.info(f"Setting ALL LEDs to {name}")
        light_bar.fill_strip(r, g, b)
        light_bar.update_strip()
        time.sleep(2)
    
    light_bar.clear_strip()
    light_bar.update_strip()

def test_different_spi_speeds():
    """Test with different SPI speeds and timing"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DIFFERENT SPI SPEEDS")
    logger.info("=" * 60)
    
    speeds = [800, 600, 400, 200, 100]
    
    for speed in speeds:
        logger.info(f"\nTesting at {speed} KHz...")
        try:
            light_bar = Pi5Neo('/dev/spidev0.0', 8, speed)
            
            # Quick flash test
            for _ in range(3):
                light_bar.fill_strip(255, 255, 255)  # White
                light_bar.update_strip()
                time.sleep(0.2)
                light_bar.clear_strip()
                light_bar.update_strip()
                time.sleep(0.2)
            
            logger.info(f"✓ {speed} KHz test completed")
            
        except Exception as e:
            logger.error(f"✗ Failed at {speed} KHz: {e}")

def test_signal_integrity():
    """Test signal integrity with rapid updates"""
    logger.info("\n" + "=" * 60)
    logger.info("SIGNAL INTEGRITY TEST")
    logger.info("=" * 60)
    
    light_bar = Pi5Neo('/dev/spidev0.0', 8, 800)
    
    logger.info("Sending rapid color changes (flashing test)...")
    for i in range(20):
        if i % 2 == 0:
            light_bar.fill_strip(255, 0, 0)  # Red
        else:
            light_bar.fill_strip(0, 0, 255)  # Blue
        light_bar.update_strip()
        time.sleep(0.1)
    
    light_bar.clear_strip()
    light_bar.update_strip()

def test_power_levels():
    """Test with different power consumption levels"""
    logger.info("\n" + "=" * 60)
    logger.info("POWER CONSUMPTION TEST")
    logger.info("=" * 60)
    
    light_bar = Pi5Neo('/dev/spidev0.0', 8, 800)
    
    logger.info("Testing with increasing number of LEDs...")
    for num_leds in range(1, 9):
        logger.info(f"Lighting {num_leds} LED(s) in white")
        light_bar.clear_strip()
        for i in range(num_leds):
            light_bar.set_led_color(i, 255, 255, 255)
        light_bar.update_strip()
        time.sleep(2)
    
    light_bar.clear_strip()
    light_bar.update_strip()

def verify_connections():
    """Display connection verification checklist"""
    logger.info("\n" + "=" * 60)
    logger.info("HARDWARE CONNECTION CHECKLIST")
    logger.info("=" * 60)
    
    logger.info("""
Please verify the following connections:

1. POWER CONNECTIONS:
   □ LED Strip 5V (usually RED wire) → Raspberry Pi 5V pin (Pin 2 or 4)
   □ LED Strip GND (usually BLACK/WHITE wire) → Raspberry Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
   
2. DATA CONNECTION:
   □ LED Strip DATA IN (usually GREEN/YELLOW wire) → GPIO 10 / MOSI (Pin 19)
   
   IMPORTANT: Make sure it's DATA IN, not DATA OUT!
   - DATA IN is the input side (usually has an arrow pointing INTO the first LED)
   - DATA OUT is on the other end of the strip

3. LED STRIP TYPE:
   □ Verify your LED strip is WS2812B/NeoPixel compatible
   □ Check the voltage rating (should be 5V)
   
4. COMMON ISSUES TO CHECK:
   □ Is the LED strip connected with correct polarity? (5V to 5V, GND to GND)
   □ Are the connections secure? (not loose)
   □ Is the data wire connected to the INPUT side of the strip?
   □ Do you have a shared ground between Pi and LED strip?
   □ Is your power supply adequate? (60mA per LED at full white)
   
5. OPTIONAL IMPROVEMENTS:
   □ Add a 470Ω resistor between GPIO 10 and DATA IN
   □ Add a 1000µF capacitor across the power supply (5V to GND)
   □ Use a level shifter if experiencing issues (3.3V GPIO to 5V data)

GPIO PINOUT REFERENCE (Physical Pin Numbers):
   Pin 2/4: 5V Power
   Pin 6/9/14/20/25/30/34/39: Ground
   Pin 19: GPIO 10 (SPI MOSI) - DATA signal
""")

def main():
    logger.info("Starting Hardware Debug Test")
    logger.info("This test will try various patterns to help diagnose hardware issues")
    
    verify_connections()
    
    logger.info("\nStarting tests in 3 seconds...")
    time.sleep(3)
    
    try:
        test_single_led_slow()
        test_all_leds_max_brightness()
        test_different_spi_speeds()
        test_signal_integrity()
        test_power_levels()
        
        logger.info("\n" + "=" * 60)
        logger.info("HARDWARE DEBUG COMPLETE")
        logger.info("=" * 60)
        
        logger.info("""
If you didn't see any lights:
1. Double-check you're connected to the DATA IN side of the strip
2. Verify 5V and GND connections with a multimeter
3. Try a different LED strip if available
4. Consider adding a logic level shifter (3.3V to 5V)
5. Check if the LED strip works with another controller (Arduino, etc.)
        """)
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()