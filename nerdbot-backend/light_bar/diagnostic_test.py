#!/usr/bin/env python3
"""
Comprehensive diagnostic test for LED light bar with detailed logging
Tests SPI device availability, permissions, and LED control functionality
"""
import sys
import os
import time
import logging
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/light_bar_diagnostic.log')
    ]
)
logger = logging.getLogger(__name__)

def check_spi_device():
    """Check if SPI device is available and accessible"""
    spi_device = '/dev/spidev0.0'
    logger.info("=" * 60)
    logger.info("CHECKING SPI DEVICE")
    logger.info("=" * 60)
    
    # Check if device exists
    if os.path.exists(spi_device):
        logger.info(f"✓ SPI device {spi_device} exists")
    else:
        logger.error(f"✗ SPI device {spi_device} NOT FOUND")
        return False
    
    # Check permissions
    if os.access(spi_device, os.R_OK):
        logger.info(f"✓ Read permission on {spi_device}")
    else:
        logger.error(f"✗ No read permission on {spi_device}")
        
    if os.access(spi_device, os.W_OK):
        logger.info(f"✓ Write permission on {spi_device}")
    else:
        logger.error(f"✗ No write permission on {spi_device}")
        
    # Check device stats
    stats = os.stat(spi_device)
    logger.info(f"Device permissions: {oct(stats.st_mode)[-3:]}")
    logger.info(f"Device owner UID: {stats.st_uid}, GID: {stats.st_gid}")
    
    # Check user groups
    import grp
    import pwd
    username = pwd.getpwuid(os.getuid()).pw_name
    user_groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
    logger.info(f"Current user: {username}")
    logger.info(f"User groups: {', '.join(user_groups)}")
    
    if 'spi' in user_groups:
        logger.info("✓ User is in 'spi' group")
    else:
        logger.warning("✗ User is NOT in 'spi' group - may have permission issues")
    
    return True

def test_pi5neo_import():
    """Test if Pi5Neo library can be imported"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING Pi5Neo LIBRARY")
    logger.info("=" * 60)
    
    try:
        import pi5neo
        logger.info("✓ Pi5Neo module imported successfully")
        logger.info(f"Pi5Neo module location: {pi5neo.__file__ if hasattr(pi5neo, '__file__') else 'built-in'}")
        
        # Check if Pi5Neo class exists
        if hasattr(pi5neo, 'Pi5Neo'):
            logger.info("✓ Pi5Neo class found")
            return True
        else:
            logger.error("✗ Pi5Neo class not found in module")
            return False
    except ImportError as e:
        logger.error(f"✗ Failed to import pi5neo: {e}")
        logger.info("Attempting to install pi5neo...")
        os.system("pip install pi5neo")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error importing pi5neo: {e}")
        return False

def test_light_bar_initialization():
    """Test initializing the light bar"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING LIGHT BAR INITIALIZATION")
    logger.info("=" * 60)
    
    try:
        from pi5neo import Pi5Neo
        
        # Test with different SPI speeds
        spi_speeds = [800, 400, 200]  # Different SPI speeds in KHz
        
        for speed in spi_speeds:
            logger.info(f"\nTesting with SPI speed: {speed} KHz")
            try:
                light_bar = Pi5Neo('/dev/spidev0.0', 8, speed)
                logger.info(f"✓ Light bar initialized successfully at {speed} KHz")
                logger.info(f"  Number of LEDs: {light_bar.num_leds}")
                
                # Try a simple operation
                light_bar.clear_strip()
                light_bar.update_strip()
                logger.info("✓ Clear strip command executed")
                
                return light_bar
                
            except Exception as e:
                logger.warning(f"✗ Failed at {speed} KHz: {e}")
                continue
                
        logger.error("✗ Failed to initialize at any SPI speed")
        return None
        
    except ImportError as e:
        logger.error(f"✗ Cannot import Pi5Neo: {e}")
        return None
    except Exception as e:
        logger.error(f"✗ Unexpected error during initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_basic_led_operations(light_bar):
    """Test basic LED operations with detailed logging"""
    if not light_bar:
        logger.error("No light bar object provided")
        return False
        
    logger.info("\n" + "=" * 60)
    logger.info("TESTING BASIC LED OPERATIONS")
    logger.info("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Clear all LEDs
    logger.info("\nTest 1: Clear all LEDs")
    try:
        light_bar.clear_strip()
        light_bar.update_strip()
        logger.info("✓ Clear strip successful")
        tests_passed += 1
        time.sleep(1)
    except Exception as e:
        logger.error(f"✗ Clear strip failed: {e}")
        tests_failed += 1
    
    # Test 2: Set single LED (first LED red)
    logger.info("\nTest 2: Set first LED to RED")
    try:
        light_bar.set_led_color(0, 255, 0, 0)
        light_bar.update_strip()
        logger.info("✓ Set single LED successful (LED 0 = RED)")
        tests_passed += 1
        time.sleep(2)
    except Exception as e:
        logger.error(f"✗ Set single LED failed: {e}")
        tests_failed += 1
    
    # Test 3: Set multiple LEDs individually
    logger.info("\nTest 3: Set LEDs individually (gradient)")
    try:
        light_bar.clear_strip()
        for i in range(min(8, light_bar.num_leds)):
            intensity = int(255 * (i + 1) / 8)
            light_bar.set_led_color(i, intensity, 0, 255 - intensity)
            logger.debug(f"  LED {i}: R={intensity}, G=0, B={255-intensity}")
        light_bar.update_strip()
        logger.info("✓ Set multiple LEDs successful")
        tests_passed += 1
        time.sleep(2)
    except Exception as e:
        logger.error(f"✗ Set multiple LEDs failed: {e}")
        tests_failed += 1
    
    # Test 4: Fill entire strip
    logger.info("\nTest 4: Fill entire strip (GREEN)")
    try:
        light_bar.fill_strip(0, 255, 0)
        light_bar.update_strip()
        logger.info("✓ Fill strip successful (ALL GREEN)")
        tests_passed += 1
        time.sleep(2)
    except Exception as e:
        logger.error(f"✗ Fill strip failed: {e}")
        tests_failed += 1
    
    # Test 5: Brightness test
    logger.info("\nTest 5: Brightness levels test")
    try:
        for brightness in [255, 128, 64, 32, 16]:
            light_bar.fill_strip(brightness, brightness, brightness)
            light_bar.update_strip()
            logger.info(f"  Brightness level: {brightness}/255")
            time.sleep(0.5)
        logger.info("✓ Brightness test successful")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Brightness test failed: {e}")
        tests_failed += 1
    
    # Final cleanup
    logger.info("\nFinal cleanup...")
    try:
        light_bar.clear_strip()
        light_bar.update_strip()
        logger.info("✓ Final cleanup successful")
    except Exception as e:
        logger.error(f"✗ Final cleanup failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    logger.info("=" * 60)
    
    return tests_failed == 0

def test_advanced_patterns():
    """Test the advanced patterns from light_bar module"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ADVANCED PATTERNS")
    logger.info("=" * 60)
    
    try:
        sys.path.insert(0, '/home/mark/nerdbot-backend')
        from light_bar.light_bar import light_bar, rainbow_cycle, loading_bar, vu_meter
        
        logger.info("✓ Imported light_bar functions successfully")
        
        # Test rainbow cycle
        logger.info("\nTesting rainbow cycle (quick)...")
        try:
            rainbow_cycle(light_bar, delay=0.2)
            logger.info("✓ Rainbow cycle completed")
        except Exception as e:
            logger.error(f"✗ Rainbow cycle failed: {e}")
        
        # Test loading bar
        logger.info("\nTesting loading bar...")
        try:
            loading_bar(light_bar)
            logger.info("✓ Loading bar completed")
        except Exception as e:
            logger.error(f"✗ Loading bar failed: {e}")
        
        # Test VU meter
        logger.info("\nTesting VU meter...")
        try:
            for volume in [25, 50, 75, 100]:
                logger.info(f"  Volume: {volume}%")
                vu_meter(light_bar, volume)
                time.sleep(0.5)
            logger.info("✓ VU meter completed")
        except Exception as e:
            logger.error(f"✗ VU meter failed: {e}")
        
        # Clear at the end
        light_bar.clear_strip()
        light_bar.update_strip()
        
    except ImportError as e:
        logger.error(f"✗ Failed to import light_bar module: {e}")
    except Exception as e:
        logger.error(f"✗ Unexpected error in advanced patterns: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main diagnostic function"""
    logger.info("Starting LED Light Bar Diagnostic Test")
    logger.info("Log file: /tmp/light_bar_diagnostic.log")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Step 1: Check SPI device
    if not check_spi_device():
        logger.error("SPI device check failed. Please check your hardware setup.")
        return
    
    # Step 2: Test Pi5Neo import
    if not test_pi5neo_import():
        logger.error("Pi5Neo library not available. Please install with: pip install pi5neo")
        return
    
    # Step 3: Initialize light bar
    light_bar = test_light_bar_initialization()
    if not light_bar:
        logger.error("Failed to initialize light bar. Check connections and permissions.")
        return
    
    # Step 4: Test basic operations
    if test_basic_led_operations(light_bar):
        logger.info("✓ All basic tests passed!")
    else:
        logger.warning("Some basic tests failed. Check the log for details.")
    
    # Step 5: Test advanced patterns
    test_advanced_patterns()
    
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC TEST COMPLETE")
    logger.info("Check /tmp/light_bar_diagnostic.log for full details")
    logger.info("=" * 60)
    
    # Hardware troubleshooting tips
    logger.info("\nHARDWARE TROUBLESHOOTING TIPS:")
    logger.info("1. Connections:")
    logger.info("   - 5V → LED strip 5V (red wire)")
    logger.info("   - GND → LED strip GND (black/white wire)")
    logger.info("   - GPIO 10 (MOSI) → LED strip DATA IN (green/yellow wire)")
    logger.info("2. Power supply:")
    logger.info("   - Ensure 5V supply can provide enough current (60mA per LED max)")
    logger.info("   - For 8 LEDs: ~500mA minimum")
    logger.info("3. Common issues:")
    logger.info("   - Check wire connections are secure")
    logger.info("   - Verify DATA IN (not DATA OUT) is connected")
    logger.info("   - Add 470Ω resistor between GPIO and DATA if experiencing flickering")
    logger.info("   - Add 1000µF capacitor across power supply for stability")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())