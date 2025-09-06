#!/usr/bin/env python3
"""
Utility script to regenerate thumbnails for all meme sounds
"""
import os
import logging
from thumbnail_generator import ThumbnailGenerator

def main():
    """Regenerate all thumbnails"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    generator = ThumbnailGenerator()
    
    print("Regenerating all meme sound thumbnails...")
    
    # Remove all existing thumbnails first
    import glob
    thumbnail_files = glob.glob('/home/mark/nerdbot-backend/assets/meme_sounds_converted/meme_thumbnails/*.png')
    print(f"Removing {len(thumbnail_files)} existing thumbnails...")
    
    for thumbnail_file in thumbnail_files:
        try:
            os.remove(thumbnail_file)
            print(f"Removed: {os.path.basename(thumbnail_file)}")
        except Exception as e:
            print(f"Error removing {os.path.basename(thumbnail_file)}: {e}")
    
    # Generate new thumbnails
    count = generator.generate_all_thumbnails()
    print(f"Successfully regenerated {count} thumbnails")

if __name__ == "__main__":
    main()