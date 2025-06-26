#!/usr/bin/env python3
"""
Data Augmentation Script for Microbio AI
Generates additional training samples to improve model accuracy
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmenter:
    def __init__(self, input_dir="data/training", output_dir="data/augmented"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_samples_per_class = 50  # Target number of samples per class
        self.img_width, self.img_height = 224, 224
        
    def create_augmented_image(self, image_path, augmentation_type="random"):
        """Create an augmented version of an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            
            if augmentation_type == "random":
                # Apply random combination of augmentations
                augmentations = [
                    self._rotate_image,
                    self._adjust_brightness,
                    self._adjust_contrast,
                    self._adjust_saturation,
                    self._add_noise,
                    self._blur_image,
                    self._sharpen_image,
                    self._flip_image
                ]
                
                # Apply 2-4 random augmentations
                num_augmentations = random.randint(2, 4)
                selected_augmentations = random.sample(augmentations, num_augmentations)
                
                for aug_func in selected_augmentations:
                    img = aug_func(img)
                    
            elif augmentation_type == "rotation":
                img = self._rotate_image(img)
            elif augmentation_type == "brightness":
                img = self._adjust_brightness(img)
            elif augmentation_type == "contrast":
                img = self._adjust_contrast(img)
            elif augmentation_type == "noise":
                img = self._add_noise(img)
            elif augmentation_type == "blur":
                img = self._blur_image(img)
            elif augmentation_type == "sharpen":
                img = self._sharpen_image(img)
            elif augmentation_type == "flip":
                img = self._flip_image(img)
            
            return img
            
        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {e}")
            return None
    
    def _rotate_image(self, img):
        """Rotate image by random angle"""
        angle = random.uniform(-30, 30)
        return img.rotate(angle, fillcolor=(128, 128, 128))
    
    def _adjust_brightness(self, img):
        """Adjust image brightness"""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, img):
        """Adjust image contrast"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _adjust_saturation(self, img):
        """Adjust image saturation"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    def _add_noise(self, img):
        """Add random noise to image"""
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def _blur_image(self, img):
        """Apply slight blur to image"""
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    def _sharpen_image(self, img):
        """Sharpen image"""
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    def _flip_image(self, img):
        """Randomly flip image horizontally or vertically"""
        if random.choice([True, False]):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    def augment_class(self, class_dir):
        """Augment all images in a class directory"""
        class_name = class_dir.name
        output_class_dir = self.output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(class_dir.glob("*.jpg"))
        current_count = len(image_files)
        
        # Skip classes with no images
        if current_count == 0:
            logger.warning(f"Class {class_name} has no images, skipping augmentation")
            return 0
        
        if current_count >= self.target_samples_per_class:
            logger.info(f"Class {class_name} already has {current_count} samples, skipping augmentation")
            return current_count
        
        needed_samples = self.target_samples_per_class - current_count
        logger.info(f"Augmenting class {class_name}: {current_count} -> {self.target_samples_per_class} samples")
        
        # Copy original images
        for i, img_file in enumerate(image_files):
            img = Image.open(img_file).convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            output_path = output_class_dir / f"{class_name}_original_{i:03d}.jpg"
            img.save(output_path, quality=95)
        
        # Generate augmented images
        augmentation_types = ["random", "rotation", "brightness", "contrast", "noise", "blur", "sharpen", "flip"]
        
        for i in range(needed_samples):
            # Select random original image
            original_img = random.choice(image_files)
            augmentation_type = random.choice(augmentation_types)
            
            # Create augmented image
            augmented_img = self.create_augmented_image(original_img, augmentation_type)
            
            if augmented_img is not None:
                output_path = output_class_dir / f"{class_name}_aug_{i:03d}.jpg"
                augmented_img.save(output_path, quality=95)
        
        return self.target_samples_per_class
    
    def run_augmentation(self):
        """Run augmentation for all classes"""
        logger.info("Starting data augmentation...")
        print("Starting data augmentation...")
        
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist!")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all class directories
        class_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        print(f"Found {len(class_dirs)} classes to augment")
        
        total_original = 0
        total_augmented = 0
        
        for i, class_dir in enumerate(class_dirs):
            print(f"[{i+1}/{len(class_dirs)}] Processing {class_dir.name}...")
            
            original_count = len(list(class_dir.glob("*.jpg")))
            total_original += original_count
            
            final_count = self.augment_class(class_dir)
            total_augmented += final_count
        
        print(f"\n=== Augmentation Complete ===")
        print(f"Original samples: {total_original}")
        print(f"Final samples: {total_augmented}")
        print(f"Generated {total_augmented - total_original} new samples")
        print(f"Output directory: {self.output_dir}")
        
        return total_augmented

def main():
    augmenter = DataAugmenter()
    augmenter.run_augmentation()

if __name__ == "__main__":
    main() 