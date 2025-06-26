import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
from model import build_bacterial_classifier # Assuming model.py is in the same directory or accessible

# Define global constants
IMAGE_DIR = 'data/raw/'
IMG_WIDTH, IMG_HEIGHT = 224, 224
# NUM_CLASSES and CLASS_NAMES will be determined dynamically
BATCH_SIZE = 32 # You can tune this
EPOCHS = 20 # You can tune this

def load_and_discover_classes(image_dir):
    """
    Loads image paths and their corresponding labels from JSON files,
    dynamically discovering all unique class names.
    """
    image_paths = []
    temp_labels_str = [] # Store string labels temporarily
    
    all_class_names_set = set()
    
    json_files = [f for f in os.listdir(image_dir) if f.endswith('.json')]

    # First pass: discover all unique class names
    for json_file_name in json_files:
        json_path = os.path.join(image_dir, json_file_name)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if data.get('classes') and len(data['classes']) > 0:
                    # Assuming the first class in the list is the primary label
                    all_class_names_set.add(data['classes'][0])
        except Exception as e:
            print(f"Warning: Error reading or parsing {json_file_name} during class discovery: {e}")

    if not all_class_names_set:
        return [], [], [], 0

    CLASS_NAMES = sorted(list(all_class_names_set))
    NUM_CLASSES = len(CLASS_NAMES)
    class_to_int_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    print(f"Discovered {NUM_CLASSES} classes: {CLASS_NAMES}")

    # Second pass: load image paths and assign integer labels
    for json_file_name in json_files:
        image_file_name = json_file_name.replace('.json', '.jpg')
        image_path = os.path.join(image_dir, image_file_name)
        json_path = os.path.join(image_dir, json_file_name)
        
        if os.path.exists(image_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if data.get('classes') and len(data['classes']) > 0:
                        class_label_str = data['classes'][0]
                        if class_label_str in class_to_int_map: # Should always be true now
                            image_paths.append(image_path)
                            temp_labels_str.append(class_label_str) # Store string label
                        # No need for 'else' here as all classes are known
                    else:
                        print(f"Warning: No 'classes' found or empty in {json_file_name} during data loading. Skipping.")
            except Exception as e:
                print(f"Error reading or parsing {json_file_name} during data loading: {e}")
        else:
            print(f"Warning: Corresponding image {image_file_name} not found for {json_file_name}. Skipping.")
    
    # Convert string labels to integers
    labels_int = [class_to_int_map[s_label] for s_label in temp_labels_str]
            
    return image_paths, labels_int, CLASS_NAMES, NUM_CLASSES

def preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Loads and preprocesses a single image."""
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

if __name__ == '__main__':
    # Create a mapping from class name string to integer index
    # class_to_int_map = {name: i for i, name in enumerate(CLASS_NAMES)} # This will be done in load_and_discover_classes

    print("Loading data and discovering classes...")
    image_paths, labels_int, CLASS_NAMES, NUM_CLASSES = load_and_discover_classes(IMAGE_DIR)
    
    if not image_paths:
        print("No data loaded. Please check your data directory and JSON files.")
        exit()
        
    print(f"Loaded {len(image_paths)} images.")

    # Preprocess images
    processed_images = []
    valid_labels = []
    for i, path in enumerate(image_paths):
        img = preprocess_image(path)
        if img is not None:
            processed_images.append(img)
            valid_labels.append(labels_int[i])
    
    if not processed_images:
        print("No images could be processed. Exiting.")
        exit()

    X = np.array(processed_images)
    y = to_categorical(np.array(valid_labels), num_classes=NUM_CLASSES) # Use dynamic NUM_CLASSES
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if np.sum(y, axis=0).min() > 1 else None) # Stratify if possible
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator() # No augmentation for validation

    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    validation_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # Build and compile model
    print("Building model...")
    model = build_bacterial_classifier(num_classes=NUM_CLASSES) # Use dynamic NUM_CLASSES

    # Optional: Fine-tuning - unfreeze some layers of the base model
    # base_model = model.layers[...name or index of base_model...] # Need to identify the base model layer correctly
    # base_model.trainable = True
    # for layer in base_model.layers[:-10]: # Example: unfreeze last 10 layers
    #     layer.trainable = False
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower learning rate for fine-tuning
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    print("Training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(X_val) // BATCH_SIZE
    )
    
    print("Training complete.")
    
    # Save the model
    model.save_weights('updated_bacterial_classifier.weights.h5')
    print("Saved updated model weights to updated_bacterial_classifier.weights.h5")

    # Further steps:
    # 1. Evaluate the model on a hold-out test set (if you have one).
    # 2. Plot training history (accuracy, loss).
    # 3. Update app.py to use 'updated_bacterial_classifier_weights.h5'. 