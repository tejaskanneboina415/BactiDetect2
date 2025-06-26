#!/usr/bin/env python3
"""
Microbio AI - Expanded Model Training Script
Trains a model on the expanded dataset with 38+ species
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your existing model
from model import build_bacterial_classifier

class ExpandedModelTrainer:
    def __init__(self, data_dir="data/training"):
        self.data_dir = Path(data_dir)
        self.img_width, self.img_height = 224, 224
        self.batch_size = 32
        self.epochs = 50  # Increased epochs for better learning
        self.learning_rate = 0.001
        
    def load_data(self):
        """Load data from the training directory, filtering out classes with <4 images"""
        logger.info("Loading training data...")
        print("Loading training data...")
        
        image_paths = []
        labels = []
        class_names = []
        excluded_classes = []
        
        # Get all species directories
        species_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        species_dirs.sort()  # Sort for consistent ordering
        
        print(f"Found {len(species_dirs)} species directories")
        
        valid_species_dirs = []
        for i, species_dir in enumerate(species_dirs):
            species_name = species_dir.name
            image_files = list(species_dir.glob("*.jpg"))
            if len(image_files) < 4:
                excluded_classes.append((species_name, len(image_files)))
                continue
            class_names.append(species_name)
            valid_species_dirs.append((species_dir, len(class_names)-1))
            print(f"[{len(class_names)}/{len(species_dirs)}] {species_name}: {len(image_files)} images")
        
        for species_dir, class_idx in valid_species_dirs:
            for image_file in species_dir.glob("*.jpg"):
                image_paths.append(str(image_file))
                labels.append(class_idx)
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        print(f"\nData loading completed!")
        print(f"Total images: {len(image_paths)}")
        print(f"Total classes: {self.num_classes}")
        if excluded_classes:
            print("\nExcluded classes with fewer than 4 images:")
            for cname, count in excluded_classes:
                print(f"  - {cname}: {count} image(s)")
        
        return image_paths, labels
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img) / 255.0
            return img_array
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def prepare_data(self, image_paths, labels):
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        print("Preparing data for training...")
        
        # Preprocess all images
        processed_images = []
        valid_labels = []
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            if i % 50 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}")
            
            img = self.preprocess_image(path)
            if img is not None:
                processed_images.append(img)
                valid_labels.append(label)
        
        X = np.array(processed_images)
        y = to_categorical(np.array(valid_labels), num_classes=self.num_classes)
        
        print(f"Data preparation completed!")
        print(f"Final dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """Create data generators with augmentation"""
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=self.batch_size)
        validation_generator = val_datagen.flow(X_val, y_val, batch_size=self.batch_size)
        
        return train_generator, validation_generator
    
    def build_advanced_model(self):
        """Build an advanced model for the expanded dataset"""
        logger.info("Building advanced model...")
        print("Building advanced model...")
        
        # Use your existing model as base
        base_model = build_bacterial_classifier(num_classes=self.num_classes)
        
        # Unfreeze some layers for fine-tuning
        # Find the base ResNet50 layer
        for layer in base_model.layers:
            if 'resnet50' in layer.name.lower() or 'conv2d' in layer.name.lower():
                layer.trainable = True
                break
        
        # Compile with different learning rates
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        base_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return base_model
    
    def train_model(self, model, train_generator, validation_generator, X_train, X_val):
        """Train the model"""
        logger.info("Starting model training...")
        print("Starting model training...")
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // self.batch_size
        validation_steps = len(X_val) // self.batch_size
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Training for {self.epochs} epochs...")
        
        # Callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_expanded.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model"""
        logger.info("Evaluating model...")
        print("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_true_classes)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=self.class_names))
        
        return y_pred_classes, y_true_classes
    
    def save_model_info(self):
        """Save model information"""
        model_info = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        }
        
        with open('model_info_expanded.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("Model information saved to model_info_expanded.json")
    
    def run(self):
        """Run the complete training pipeline"""
        logger.info("Starting expanded model training...")
        print("="*60)
        print("MICROBIO AI - EXPANDED MODEL TRAINING")
        print("="*60)
        
        # Load data
        image_paths, labels = self.load_data()
        
        if len(image_paths) == 0:
            print("No data found! Please check your data directory.")
            return
        
        # Prepare data
        X, y = self.prepare_data(image_paths, labels)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Training: {len(X_train)} images")
        print(f"Validation: {len(X_val)} images")
        print(f"Test: {len(X_test)} images")
        
        # Create data generators
        train_generator, validation_generator = self.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Build model
        model = self.build_advanced_model()
        
        # Train model
        history = self.train_model(model, train_generator, validation_generator, X_train, X_val)
        
        # Evaluate model
        y_pred_classes, y_true_classes = self.evaluate_model(model, X_test, y_test)
        
        # Save model and info
        model.save_weights('expanded_bacterial_classifier.weights.h5')
        self.save_model_info()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Model saved as: expanded_bacterial_classifier.weights.h5")
        print(f"Model info saved as: model_info_expanded.json")
        print(f"Best model saved as: best_model_expanded.h5")
        print("="*60)

def main():
    """Main function"""
    trainer = ExpandedModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main() 