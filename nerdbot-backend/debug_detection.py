#!/usr/bin/env python3
"""
Debug script to test detection label conversion
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask_server.server import imx500_get_labels
    
    print("=== DETECTION DEBUG SCRIPT ===")
    
    # Get the labels
    labels = imx500_get_labels()
    print(f"Total labels: {len(labels)}")
    print(f"First 20 labels: {labels[:20]}")
    print(f"Labels with 'n/a': {[i for i, label in enumerate(labels) if label == 'n/a']}")
    
    # Test some common category indices
    test_categories = [0, 1, 2, 15, 16, 17]  # person, bicycle, car, bird, cat, dog
    
    for cat in test_categories:
        if 0 <= cat < len(labels):
            label = labels[cat].lower().strip()
            print(f"Category {cat} -> '{label}'")
        else:
            print(f"Category {cat} -> INVALID (out of range)")
    
    # Check for empty or problematic labels
    problem_labels = []
    for i, label in enumerate(labels):
        if not label or label.strip() in ['n/a', 'na', '', '0', 'zero', 'none', 'unknown']:
            problem_labels.append((i, label))
    
    print(f"Problematic labels found: {problem_labels}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()