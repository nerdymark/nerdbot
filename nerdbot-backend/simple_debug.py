#!/usr/bin/env python3
"""
Simple debug script to check COCO labels without initializing camera
"""

# Read the COCO labels directly
try:
    with open("assets/coco_labels.txt", "r", encoding="utf-8") as fi:
        labels = fi.read().splitlines()
    
    print("=== COCO LABELS DEBUG ===")
    print(f"Total labels: {len(labels)}")
    print(f"First 20 labels: {labels[:20]}")
    
    # Find problematic labels
    problem_labels = []
    for i, label in enumerate(labels):
        if not label or label.strip() in ['n/a', 'na', '', '0', 'zero', 'none', 'unknown']:
            problem_labels.append((i, label))
    
    print(f"Problematic labels: {problem_labels}")
    
    # Test some common category indices
    test_categories = [0, 1, 2, 15, 16, 17]  # person, bicycle, car, bird, cat, dog
    
    print("\nCategory index to label mapping:")
    for cat in test_categories:
        if 0 <= cat < len(labels):
            label = labels[cat].lower().strip()
            print(f"  Category {cat} -> '{label}'")
        else:
            print(f"  Category {cat} -> INVALID (out of range)")
            
    # Look for any labels that might be causing the "0.0" issue
    print(f"\nChecking for numeric-looking labels:")
    for i, label in enumerate(labels):
        if label and ('0' in label or label.isdigit()):
            print(f"  Index {i}: '{label}'")
    
except Exception as e:
    print(f"Error reading labels: {e}")