#!/usr/bin/env python3
"""
Microbio AI - Improved Model Training Script
Advanced training with multiple accuracy enhancement strategies
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedModelTrainer:
    def __init__(self, data_dir="data/training"):
        self.data_dir = Path(data_dir)
        self.img_width, self.img_height = 224, 224
        self.batch_size = 16  # Reduced batch size for better gradient updates
        self.epochs = 100  # More epochs with early stopping
        self.learning_rate = 0.001
        self.fold_count = 5  # For cross-validation
        
    def load_data(self):
        """Load data from the training directory, filtering out classes with <8 images"""
        logger.info("Loading training data...")
        print("Loading training data...")
        
        image_paths = []
        labels = []
        class_names = []
        excluded_classes = []
        
        # Get all species directories
        species_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        species_dirs.sort()
        
        print(f"Found {len(species_dirs)} species directories")
        
        valid_species_dirs = []
        for i, species_dir in enumerate(species_dirs):
            species_name = species_dir.name
            image_files = list(species_dir.glob("*.jpg"))
            if len(image_files) < 8:  # Increased minimum threshold
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
            print("\nExcluded classes with fewer than 8 images:")
            for cname, count in excluded_classes:
                print(f"  - {cname}: {count} image(s)")
        
        return image_paths, labels
    
    def advanced_preprocess_image(self, image_path):
        """Advanced preprocessing with OpenCV"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype(np.float32) / 255.0
            return img
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def prepare_data(self, image_paths, labels):
        """Prepare data for training with advanced preprocessing"""
        logger.info("Preparing data for training...")
        print("Preparing data for training...")
        
        processed_images = []
        valid_labels = []
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            if i % 50 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}")
            
            img = self.advanced_preprocess_image(path)
            if img is not None:
                processed_images.append(img)
                valid_labels.append(label)
        
        X = np.array(processed_images)
        y = to_categorical(np.array(valid_labels), num_classes=self.num_classes)
        
        print(f"Data preparation completed!")
        print(f"Final dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def build_improved_resnet50(self):
        """Build improved ResNet50 model with better architecture"""
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.img_width, self.img_height, 3))
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Use different learning rates for different layers
        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_efficientnet(self):
        """Build EfficientNet model"""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(self.img_width, self.img_height, 3))
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        optimizer = Adam(learning_rate=self.learning_rate * 0.5)  # Lower learning rate for EfficientNet
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def cosine_annealing_schedule(self, epoch):
        """Cosine annealing learning rate schedule"""
        initial_lr = self.learning_rate
        min_lr = 1e-7
        max_epochs = self.epochs
        
        if epoch < max_epochs // 4:
            return initial_lr
        else:
            return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch - max_epochs // 4) / (max_epochs - max_epochs // 4))) / 2
    
    def create_advanced_data_generators(self, X_train, y_train, X_val, y_val):
        """Create advanced data generators with sophisticated augmentation"""
        # Advanced training augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='reflect',
            preprocessing_function=lambda x: x + np.random.normal(0, 0.01, x.shape)  # Add noise
        )
        
        # Validation data (minimal augmentation)
        val_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: x + np.random.normal(0, 0.005, x.shape)  # Minimal noise
        )
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=self.batch_size)
        validation_generator = val_datagen.flow(X_val, y_val, batch_size=self.batch_size)
        
        return train_generator, validation_generator
    
    def train_with_cross_validation(self, X, y):
        """Train model with k-fold cross-validation"""
        logger.info("Starting cross-validation training...")
        print("Starting cross-validation training...")
        
        skf = StratifiedKFold(n_splits=self.fold_count, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, np.argmax(y, axis=1))):
            print(f"\n=== Training Fold {fold + 1}/{self.fold_count} ===")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create data generators
            train_generator, validation_generator = self.create_advanced_data_generators(
                X_train, y_train, X_val, y_val
            )
            
            # Build model
            model = self.build_improved_resnet50()
            
            # Calculate steps
            steps_per_epoch = len(X_train) // self.batch_size
            validation_steps = len(X_val) // self.batch_size
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=8,
                    min_lr=1e-8,
                    verbose=1
                ),
                tf.keras.callbacks.LearningRateScheduler(self.cosine_annealing_schedule, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    f'best_model_fold_{fold+1}.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(val_accuracy)
            
            print(f"Fold {fold + 1} - Validation Accuracy: {val_accuracy:.4f}")
            
            # Save fold results
            self.save_fold_results(fold + 1, history, val_accuracy)
        
        # Calculate average performance
        avg_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        
        print(f"\n=== Cross-Validation Results ===")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in fold_scores]}")
        
        return avg_accuracy, fold_scores
    
    def save_fold_results(self, fold_num, history, val_accuracy):
        """Save results for each fold"""
        results = {
            'fold': fold_num,
            'val_accuracy': float(val_accuracy),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        
        with open(f'fold_{fold_num}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def run(self):
        """Main training pipeline"""
        print("=== Improved Microbio AI Training Pipeline ===")
        
        # Load data
        image_paths, labels = self.load_data()
        
        if len(image_paths) == 0:
            print("No valid images found!")
            return
        
        # Prepare data
        X, y = self.prepare_data(image_paths, labels)
        
        # Train with cross-validation
        avg_accuracy, fold_scores = self.train_with_cross_validation(X, y)
        
        # Save final results
        final_results = {
            'average_accuracy': float(avg_accuracy),
            'fold_accuracies': [float(acc) for acc in fold_scores],
            'num_classes': self.num_classes,
            'total_images': len(X),
            'class_names': self.class_names
        }
        
        with open('improved_training_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n=== Training Complete ===")
        print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
        print(f"Results saved to improved_training_results.json")
        
        return avg_accuracy

def main():
    trainer = ImprovedModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main() 