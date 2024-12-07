"""
Control the LED strip using the Pi5Neo library
"""
from pi5neo import Pi5Neo
import time


light_bar = Pi5Neo('/dev/spidev0.0', 8, 800)

def rainbow_cycle(pixels, delay=0.1):
    """
    Create a rainbow cycle on the LED strip
    """
    colors = [
        (255, 0, 0),  # Red
        (255, 127, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (75, 0, 130),  # Indigo
        (148, 0, 211)  # Violet
    ]
    for color in colors:
        pixels.fill_strip(*color)
        pixels.update_strip()
        time.sleep(delay)


def loading_bar(pixels):
    """
    Create a loading bar on the LED strip
    """
    for i in range(pixels.num_leds):
        pixels.set_led_color(i, 0, 255, 0)  # Green loading bar
        pixels.update_strip()
        time.sleep(0.2)
    pixels.clear_strip()
    pixels.update_strip()


def vu_meter(pixels, volume):
    """
    Create a VU meter on the LED strip
    Using a volume level of 0-100 to determine the number of LEDs to light up
    Light up LEDS starting from the middle and moving outwards
    Color scheme:
        - Green: 0-10%
        - Yellow: 10-50%
        - Red: 50-100%
    """
    pixels.clear_strip()
    num_leds = pixels.num_leds
    num_leds_on = int(volume / 100 * num_leds)
    green_leds = int(num_leds_on * 0.1)
    yellow_leds = int(num_leds_on * 0.4)
    red_leds = num_leds_on - green_leds - yellow_leds

    for i in range(green_leds):
        pixels.set_led_color(num_leds // 2 + i, 0, 255, 0)
    for i in range(yellow_leds):
        pixels.set_led_color(num_leds // 2 + green_leds + i, 255, 255, 0)
    for i in range(red_leds):
        pixels.set_led_color(num_leds // 2 + green_leds + yellow_leds + i, 255, 0, 0)
    pixels.update_strip()
