#!/usr/bin/env python3
"""
Thumbnail Generator for Meme Sounds
Uses Gemini API to generate emoji-like thumbnail images based on filename
"""
import os
import json
import logging
import re
import requests
import io
import base64
from PIL import Image

# Load config
CONFIG_FILE = '/home/mark/nerdbot-scripts/config.json'
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = json.load(f)

GOOGLE_GEMINI_KEY = config.get('google_gemini_key')
MEME_SOUNDS_FOLDER_CONVERTED = config.get('meme_sounds_folder_converted')
THUMBNAIL_FOLDER = os.path.join(os.path.dirname(MEME_SOUNDS_FOLDER_CONVERTED), 'meme_thumbnails')

# Ensure thumbnail directory exists
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

class ThumbnailGenerator:
    """Generate emoji-like thumbnails for meme sounds using Gemini AI image generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.thumbnail_size = (128, 128)  # Thumbnail sized as requested
        
    def clean_filename(self, filename):
        """Clean filename for better prompt generation"""
        # Remove extension
        name = os.path.splitext(filename)[0]
        # Replace dashes and underscores with spaces
        name = re.sub(r'[-_]', ' ', name)
        # Remove common sound effect suffixes
        name = re.sub(r'(sound effect|sound|effect|meme|hd)$', '', name, flags=re.IGNORECASE)
        return name.strip()
    
    def generate_image_with_nano_banana(self, filename, force_regenerate=False):
        """Generate thumbnail image using Google's Nano Banana API"""
        clean_name = self.clean_filename(filename)
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, f"{os.path.splitext(filename)[0]}.png")
        
        # Skip if thumbnail already exists and not forcing regeneration
        if os.path.exists(thumbnail_path) and not force_regenerate:
            self.logger.info("Thumbnail already exists for %s", filename)
            return thumbnail_path
        
        # Create a descriptive prompt for image generation
        prompt = f"""Simple emoji-style icon representing "{clean_name}". Flat design, bright colors, minimalist, no text, thumbnail size, fun and playful style suitable for a meme soundboard."""

        try:
            # Use the correct endpoint for Nano Banana (Gemini 2.5 Flash Image)
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt}
                    ]
                }]
            }
            
            response = requests.post(
                f"{url}?key={GOOGLE_GEMINI_KEY}", 
                headers=headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    parts = result['candidates'][0]['content']['parts']
                    for part in parts:
                        if 'inlineData' in part and part['inlineData']['mimeType'].startswith('image/'):
                            # Get the base64 encoded image
                            image_data = part['inlineData']['data']
                            
                            # Decode and save the image
                            image_bytes = base64.b64decode(image_data)
                            
                            # Open and resize to thumbnail size
                            img = Image.open(io.BytesIO(image_bytes))
                            img = img.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
                            img.save(thumbnail_path, 'PNG')
                            
                            self.logger.info("Generated thumbnail with Nano Banana: %s", thumbnail_path)
                            return thumbnail_path
                
                self.logger.warning("No image data received from Nano Banana, using fallback")
                return self.create_fallback_thumbnail(filename, clean_name, force_regenerate)
            else:
                self.logger.error("Nano Banana API error %d: %s", response.status_code, response.text)
                return self.create_fallback_thumbnail(filename, clean_name, force_regenerate)
            
        except Exception as e:
            self.logger.error("Error generating image with Nano Banana: %s", e)
            return self.create_fallback_thumbnail(filename, clean_name)
    
    def create_fallback_thumbnail(self, filename, clean_name, force_regenerate=False):
        """Create a colorful thumbnail as fallback"""
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, f"{os.path.splitext(filename)[0]}.png")
        
        # Skip if thumbnail already exists and not forcing regeneration
        if os.path.exists(thumbnail_path) and not force_regenerate:
            self.logger.info("Thumbnail already exists for %s", filename)
            return thumbnail_path
        
        # Create colorful image based on filename
        try:
            from colorsys import hsv_to_rgb
            import hashlib
            
            # Generate colors based on filename hash
            hash_obj = hashlib.md5(filename.encode())
            hash_hex = hash_obj.hexdigest()
            hash_val = int(hash_hex[:8], 16) % 360
            
            # Create image with light background
            img = Image.new('RGB', self.thumbnail_size, (248, 248, 248))
            
            # Generate vibrant but pleasing colors
            color1 = tuple(int(255 * x) for x in hsv_to_rgb(hash_val / 360, 0.7, 0.85))
            color2 = tuple(int(255 * x) for x in hsv_to_rgb((hash_val + 120) / 360, 0.6, 0.75))
            color3 = tuple(int(255 * x) for x in hsv_to_rgb((hash_val + 240) / 360, 0.65, 0.8))
            
            pixels = img.load()
            width, height = img.size
            
            # Create a gradient/pattern
            for x in range(width):
                for y in range(height):
                    # Create circular gradient pattern
                    center_x, center_y = width // 2, height // 2
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    max_distance = (width ** 2 + height ** 2) ** 0.5 / 2
                    
                    ratio = min(distance / max_distance, 1.0)
                    
                    # Blend colors based on position
                    if ratio < 0.33:
                        blend_ratio = ratio * 3
                        r = int(color1[0] * (1 - blend_ratio) + color2[0] * blend_ratio)
                        g = int(color1[1] * (1 - blend_ratio) + color2[1] * blend_ratio)
                        b = int(color1[2] * (1 - blend_ratio) + color2[2] * blend_ratio)
                    elif ratio < 0.66:
                        blend_ratio = (ratio - 0.33) * 3
                        r = int(color2[0] * (1 - blend_ratio) + color3[0] * blend_ratio)
                        g = int(color2[1] * (1 - blend_ratio) + color3[1] * blend_ratio)
                        b = int(color2[2] * (1 - blend_ratio) + color3[2] * blend_ratio)
                    else:
                        blend_ratio = (ratio - 0.66) * 3
                        r = int(color3[0] * (1 - blend_ratio) + color1[0] * blend_ratio)
                        g = int(color3[1] * (1 - blend_ratio) + color1[1] * blend_ratio)
                        b = int(color3[2] * (1 - blend_ratio) + color1[2] * blend_ratio)
                    
                    pixels[x, y] = (r, g, b)
            
            # Add simple geometric shape overlay
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Draw circle or square based on hash
            shape_choice = hash_val % 3
            center_x, center_y = width // 2, height // 2
            
            if shape_choice == 0:  # Circle
                radius = 60
                draw.ellipse([center_x - radius, center_y - radius, 
                            center_x + radius, center_y + radius], 
                           outline=(255, 255, 255), width=6)
            elif shape_choice == 1:  # Square
                size = 80
                draw.rectangle([center_x - size//2, center_y - size//2,
                              center_x + size//2, center_y + size//2], 
                             outline=(255, 255, 255), width=6)
            else:  # Triangle
                points = [(center_x, center_y - 60), 
                         (center_x - 52, center_y + 30),
                         (center_x + 52, center_y + 30)]
                draw.polygon(points, outline=(255, 255, 255), width=6)
            
            # Add border
            draw.rectangle([0, 0, width-1, height-1], 
                          outline=(255, 255, 255), width=4)
            
            # Save thumbnail
            img.save(thumbnail_path, 'PNG')
            self.logger.info("Created fallback thumbnail: %s", thumbnail_path)
            return thumbnail_path
            
        except Exception as e:
            self.logger.error("Error creating fallback thumbnail: %s", e)
            return None
    
    def generate_thumbnail_for_sound(self, filename, force_regenerate=False):
        """Generate a complete thumbnail for a single sound file"""
        try:
            thumbnail_path = self.generate_image_with_nano_banana(filename, force_regenerate)
            return thumbnail_path
        except Exception as e:
            self.logger.error("Error generating thumbnail for %s: %s", filename, e)
            return None
    
    def generate_all_thumbnails(self):
        """Generate thumbnails for all meme sounds"""
        if not os.path.exists(MEME_SOUNDS_FOLDER_CONVERTED):
            self.logger.error("Meme sounds folder not found: %s", MEME_SOUNDS_FOLDER_CONVERTED)
            return 0
        
        sound_files = [f for f in os.listdir(MEME_SOUNDS_FOLDER_CONVERTED) if f.endswith('.mp3')]
        self.logger.info("Found %d sound files", len(sound_files))
        
        generated_count = 0
        for sound_file in sound_files:
            self.logger.info("Processing: %s", sound_file)
            if self.generate_thumbnail_for_sound(sound_file):
                generated_count += 1
        
        self.logger.info("Generated %d thumbnails", generated_count)
        return generated_count
    
    def get_thumbnail_path(self, filename):
        """Get the path to a thumbnail for a given sound filename"""
        thumbnail_filename = f"{os.path.splitext(filename)[0]}.png"
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, thumbnail_filename)
        
        # Generate if doesn't exist
        if not os.path.exists(thumbnail_path):
            self.logger.info("Thumbnail doesn't exist for %s, generating...", filename)
            self.generate_thumbnail_for_sound(filename)
        
        return thumbnail_path if os.path.exists(thumbnail_path) else None

def main():
    """Main function for command line usage"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    generator = ThumbnailGenerator()
    
    if len(os.sys.argv) > 1:
        # Generate thumbnail for specific file
        filename = os.sys.argv[1]
        result = generator.generate_thumbnail_for_sound(filename)
        if result:
            print(f"Generated thumbnail: {result}")
        else:
            print(f"Failed to generate thumbnail for: {filename}")
    else:
        # Generate all thumbnails
        count = generator.generate_all_thumbnails()
        print(f"Generated {count} thumbnails")

if __name__ == "__main__":
    main()