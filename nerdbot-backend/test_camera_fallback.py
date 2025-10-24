#!/usr/bin/env python3
"""
Test script to verify camera failsafe mechanism
"""
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask server URL
BASE_URL = "http://localhost:5000"

def test_camera_status():
    """Test camera status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/camera/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Camera Status: {data}")
            return data
        else:
            logger.error(f"Camera status failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Camera status error: {e}")
        return None

def test_camera_fallback():
    """Test manual camera fallback"""
    try:
        response = requests.post(f"{BASE_URL}/api/camera/fallback", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Camera Fallback Success: {data}")
            return True
        else:
            logger.error(f"Camera fallback failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Camera fallback error: {e}")
        return False

def test_camera_reset():
    """Test camera reset"""
    try:
        response = requests.post(f"{BASE_URL}/api/camera/reset", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Camera Reset Success: {data}")
            return True
        else:
            logger.error(f"Camera reset failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Camera reset error: {e}")
        return False

def main():
    """Run camera failsafe tests"""
    logger.info("Starting camera failsafe tests...")
    
    # Test 1: Check initial camera status
    logger.info("=== Test 1: Initial Camera Status ===")
    initial_status = test_camera_status()
    if initial_status:
        logger.info(f"Initial camera type: {initial_status.get('camera_type', 'unknown')}")
        logger.info(f"Hailo enabled: {initial_status.get('hailo_enabled', 'unknown')}")
    
    time.sleep(1)
    
    # Test 2: Test manual fallback
    logger.info("=== Test 2: Manual Fallback ===")
    if test_camera_fallback():
        time.sleep(2)
        fallback_status = test_camera_status()
        if fallback_status:
            logger.info(f"After fallback - camera type: {fallback_status.get('camera_type', 'unknown')}")
            logger.info(f"After fallback - hailo enabled: {fallback_status.get('hailo_enabled', 'unknown')}")
    
    time.sleep(1)
    
    # Test 3: Test camera reset
    logger.info("=== Test 3: Camera Reset ===")
    if test_camera_reset():
        time.sleep(2)
        reset_status = test_camera_status()
        if reset_status:
            logger.info(f"After reset - camera type: {reset_status.get('camera_type', 'unknown')}")
            logger.info(f"After reset - hailo enabled: {reset_status.get('hailo_enabled', 'unknown')}")
    
    logger.info("Camera failsafe tests completed")

if __name__ == "__main__":
    main()