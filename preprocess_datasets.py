#!/usr/bin/env python3
"""
Microbio AI - Dataset Preprocessing Script
Converts downloaded datasets to the right format for training
"""

import os
import shutil
from PIL import Image
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    def __init__(self):
        self.base_dir = Path("data")
        self.processed_dir = self.base_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def preprocess_dibas(self):
        """Preprocess DIBaS dataset"""
        logger.info("Starting DIBaS preprocessing...")
        print("Starting DIBaS preprocessing...")
        
        dibas_source = self.base_dir / "dibas"
        dibas_processed = self.processed_dir / "dibas"
        dibas_processed.mkdir(exist_ok=True)
        
        # Get all species directories
        species_dirs = [d for d in dibas_source.iterdir() if d.is_dir()]
        total_species = len(species_dirs)
        
        print(f"Found {total_species} species to process")
        
        processed_count = 0
        total_images = 0
        
        for i, species_dir in enumerate(species_dirs, 1):
            species_name = species_dir.name
            print(f"[{i}/{total_species}] Processing {species_name}...")
            
            # Create output directory for this species
            output_dir = dibas_processed / species_name
            output_dir.mkdir(exist_ok=True)
            
            # Process all images in this species directory
            image_files = list(species_dir.glob("*.tif"))
            species_image_count = 0
            
            for image_file in image_files:
                try:
                    # Open and convert image
                    img = Image.open(image_file)
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 224x224 (standard size for many models)
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Save as JPEG with high quality
                    output_filename = image_file.stem + ".jpg"
                    output_path = output_dir / output_filename
                    img.save(output_path, "JPEG", quality=95)
                    
                    species_image_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_file}: {e}")
                    continue
            
            print(f"  ✓ Processed {species_image_count} images for {species_name}")
            total_images += species_image_count
            processed_count += 1
        
        print(f"\nDIBaS preprocessing completed!")
        print(f"Processed {processed_count}/{total_species} species")
        print(f"Total images: {total_images}")
        
        return processed_count > 0
    
    def merge_with_existing_data(self):
        """Merge new data with existing raw data"""
        logger.info("Merging with existing data...")
        print("Merging with existing data...")
        
        # Check if existing raw data exists
        existing_raw = self.base_dir / "raw"
        if not existing_raw.exists():
            print("No existing raw data found, skipping merge")
            return
        
        # Create merged directory
        merged_dir = self.processed_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        
        # Copy existing raw data
        if existing_raw.exists():
            print("Copying existing raw data...")
            for item in existing_raw.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Find the corresponding JSON file
                    json_file = item.with_suffix('.json')
                    if json_file.exists():
                        # Read the JSON to get the class
                        try:
                            import json
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                if 'classes' in data and data['classes']:
                                    class_name = data['classes'][0]
                                    class_dir = merged_dir / class_name
                                    class_dir.mkdir(exist_ok=True)
                                    
                                    # Copy and resize image
                                    img = Image.open(item)
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                                    
                                    output_path = class_dir / f"existing_{item.stem}.jpg"
                                    img.save(output_path, "JPEG", quality=95)
                        except Exception as e:
                            logger.error(f"Failed to process existing file {item}: {e}")
        
        # Copy processed DIBaS data
        dibas_processed = self.processed_dir / "dibas"
        if dibas_processed.exists():
            print("Copying processed DIBaS data...")
            for species_dir in dibas_processed.iterdir():
                if species_dir.is_dir():
                    species_name = species_dir.name
                    class_dir = merged_dir / species_name
                    class_dir.mkdir(exist_ok=True)
                    
                    for image_file in species_dir.glob("*.jpg"):
                        shutil.copy2(image_file, class_dir / f"dibas_{image_file.name}")
        
        print("Data merge completed!")
    
    def create_training_structure(self):
        """Create the final training data structure"""
        logger.info("Creating training data structure...")
        print("Creating training data structure...")
        
        # Use merged data if available, otherwise use processed DIBaS
        source_dir = self.processed_dir / "merged"
        if not source_dir.exists():
            source_dir = self.processed_dir / "dibas"
        
        training_dir = self.base_dir / "training"
        training_dir.mkdir(exist_ok=True)
        
        # Copy all data to training directory
        for species_dir in source_dir.iterdir():
            if species_dir.is_dir():
                species_name = species_dir.name
                output_dir = training_dir / species_name
                output_dir.mkdir(exist_ok=True)
                
                for image_file in species_dir.glob("*.jpg"):
                    shutil.copy2(image_file, output_dir / image_file.name)
        
        # Count total images
        total_images = sum(len(list(d.glob("*.jpg"))) for d in training_dir.iterdir() if d.is_dir())
        total_species = len([d for d in training_dir.iterdir() if d.is_dir()])
        
        print(f"Training data structure created!")
        print(f"Total species: {total_species}")
        print(f"Total images: {total_images}")
        
        return training_dir
    
    def run(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting dataset preprocessing...")
        print("Starting dataset preprocessing...")
        
        # Preprocess DIBaS
        dibas_success = self.preprocess_dibas()
        
        # Merge with existing data
        self.merge_with_existing_data()
        
        # Create final training structure
        training_dir = self.create_training_structure()
        
        logger.info("Dataset preprocessing completed!")
        
        # Print summary
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"DIBaS Preprocessing: {'✓ Completed' if dibas_success else '✗ Failed'}")
        print(f"Training data location: {training_dir.absolute()}")
        
        # List all species
        species_list = [d.name for d in training_dir.iterdir() if d.is_dir()]
        print(f"Species available: {len(species_list)}")
        print("Species list:")
        for i, species in enumerate(sorted(species_list), 1):
            image_count = len(list((training_dir / species).glob("*.jpg")))
            print(f"  {i:2d}. {species} ({image_count} images)")
        
        print("="*50)

def main():
    """Main function"""
    preprocessor = DatasetPreprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main() 