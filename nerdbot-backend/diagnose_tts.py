#!/usr/bin/env python3
"""
Diagnostic script to test TTS pipeline reliability and identify failure points
"""

import time
import requests
import threading
import json
import os
import re
from urllib.parse import quote
from pathlib import Path

def test_text_sanitization():
    """Test the text sanitization regex"""
    print("=== Testing Text Sanitization ===")

    # This is the regex from the piper_tts function
    text_regex = re.compile(r'[^A-Za-z0-9\s\.,\'\"\-\?\!]+')

    test_cases = [
        "Hello world!",
        "This has émojis and spëcial chars",
        "Numbers 123 and symbols @#$%",
        "Quotes 'single' and \"double\"",
        "Punctuation... question? exclamation!",
        "Mixed: café naïve résumé",
        "Unicode: 😀🤖👍",
        "Empty after sanitization: 🔥💯✨",
        "",  # Empty string
        "   ",  # Just spaces
        ".",  # Single punctuation
        "123",  # Just numbers
    ]

    for text in test_cases:
        sanitized = text_regex.sub('', text)
        print(f"Original: '{text}' -> Sanitized: '{sanitized}'")
        if not sanitized.strip():
            print(f"  ⚠️  WARNING: Text becomes empty after sanitization!")

    print()

def test_url_encoding():
    """Test URL encoding issues"""
    print("=== Testing URL Encoding ===")

    test_cases = [
        "Hello world!",
        "Text with spaces and punctuation...",
        "Special chars: café naïve",
        "Quotes 'single' and \"double\"",
        "Forward/back slashes: / \\",
        "Percent signs: 50% completion",
    ]

    for text in test_cases:
        encoded = quote(text)
        print(f"Text: '{text}' -> URL encoded: '{encoded}'")

    print()

def test_concurrent_requests():
    """Test concurrent TTS requests for race conditions"""
    print("=== Testing Concurrent Requests ===")

    results = []

    def make_request(text, request_id):
        """Make a single TTS request"""
        try:
            url = f'http://localhost:5000/api/tts/{quote(text)}'
            response = requests.post(url, timeout=30)
            success = response.status_code == 200
            results.append({
                'id': request_id,
                'text': text,
                'success': success,
                'status_code': response.status_code,
                'response': response.text if not success else 'OK'
            })
            print(f"Request {request_id}: {'✓' if success else '✗'} ({response.status_code})")
        except Exception as e:
            results.append({
                'id': request_id,
                'text': text,
                'success': False,
                'error': str(e)
            })
            print(f"Request {request_id}: ✗ (Exception: {e})")

    # Test concurrent requests
    threads = []
    test_texts = [
        f"Test message number {i+1} for concurrent testing."
        for i in range(10)
    ]

    start_time = time.time()

    for i, text in enumerate(test_texts):
        thread = threading.Thread(target=make_request, args=(text, i+1))
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Small delay between starts

    # Wait for all to complete
    for thread in threads:
        thread.join()

    duration = time.time() - start_time

    # Analyze results
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nConcurrent test completed in {duration:.2f}s")
    print(f"Success: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"Failed: {failed}/{len(results)} ({failed/len(results)*100:.1f}%)")

    if failed > 0:
        print("\nFailure details:")
        for r in results:
            if not r['success']:
                print(f"  Request {r['id']}: {r.get('error', r.get('response', 'Unknown error'))}")

    print()

def test_edge_cases():
    """Test edge cases that might cause failures"""
    print("=== Testing Edge Cases ===")

    edge_cases = [
        "",  # Empty string
        " ",  # Just space
        ".",  # Single period
        "...",  # Multiple periods
        "A",  # Single character
        "A" * 1000,  # Very long text
        "Test with\nnewlines\nand\ttabs",
        "Rapid. Fire. Sentences. With. Periods.",
        "Question? After question? More questions?",
        "Mix of punctuation: Hello, world! How are you? Fine...",
    ]

    for i, text in enumerate(edge_cases):
        print(f"Testing edge case {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        try:
            url = f'http://localhost:5000/api/tts/{quote(text)}'
            response = requests.post(url, timeout=15)
            success = response.status_code == 200
            print(f"  Result: {'✓' if success else '✗'} ({response.status_code})")
            if not success:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")

        time.sleep(1)  # Small delay between tests

    print()

def check_audio_service():
    """Check if audio service is running and processing requests"""
    print("=== Checking Audio Service ===")

    # Check if audio service directory exists
    audio_dir = Path('/tmp/nerdbot_audio')
    if not audio_dir.exists():
        print("⚠️  Audio request directory doesn't exist!")
        return

    # Check for pending requests
    pending = list(audio_dir.glob('*.json'))
    print(f"Pending audio requests: {len(pending)}")

    if pending:
        print("Pending requests:")
        for req_file in pending[:5]:  # Show first 5
            try:
                with open(req_file) as f:
                    data = json.load(f)
                print(f"  {req_file.name}: {data.get('type', 'unknown')} - {data.get('timestamp', 'no timestamp')}")
            except Exception as e:
                print(f"  {req_file.name}: Error reading - {e}")

    # Test creating a direct audio request
    print("\nTesting direct audio request creation...")
    try:
        test_request = {
            'type': 'test',
            'timestamp': time.time()
        }
        test_file = audio_dir / f"test_{int(time.time())}.json"
        with open(test_file, 'w') as f:
            json.dump(test_request, f)
        print(f"✓ Created test request: {test_file}")

        # Clean up
        time.sleep(1)
        if test_file.exists():
            test_file.unlink()
            print("⚠️  Test file not processed by audio service")
        else:
            print("✓ Test file was processed by audio service")

    except Exception as e:
        print(f"✗ Failed to create test request: {e}")

    print()

def test_pipeline_components():
    """Test individual pipeline components"""
    print("=== Testing Pipeline Components ===")

    # Test if required commands exist
    import subprocess

    commands_to_test = [
        ('echo', ['echo', 'test']),
        ('piper', ['which', 'piper']),
        ('sox', ['which', 'sox']),
    ]

    for name, cmd in commands_to_test:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ {name} available")
            else:
                print(f"✗ {name} not available or failed")
        except Exception as e:
            print(f"✗ {name} test failed: {e}")

    print()

def main():
    """Run all diagnostic tests"""
    print("TTS Pipeline Diagnostic Tool")
    print("=" * 50)

    test_text_sanitization()
    test_url_encoding()
    check_audio_service()
    test_pipeline_components()

    print("Starting TTS endpoint tests...")
    print("(This will make audio - you may want to lower volume)")
    input("Press Enter to continue or Ctrl+C to stop...")

    test_edge_cases()
    test_concurrent_requests()

    print("Diagnostic complete!")

if __name__ == "__main__":
    main()