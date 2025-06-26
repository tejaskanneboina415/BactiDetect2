from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os
import json
from model import build_bacterial_classifier

app = Flask(__name__)

# Load model information
MODEL_INFO_FILE = 'model_info_expanded.json'
EXPANDED_WEIGHTS_FILE = 'expanded_bacterial_classifier.weights.h5'
FALLBACK_WEIGHTS_FILE = 'updated_bacterial_classifier.weights.h5'

# Try to load expanded model info, fallback to original
if os.path.exists(MODEL_INFO_FILE):
    with open(MODEL_INFO_FILE, 'r') as f:
        model_info = json.load(f)
    CLASS_NAMES = model_info['class_names']
    NUM_CLASSES = model_info['num_classes']
    print(f"Loaded expanded model with {NUM_CLASSES} classes")
else:
    # Fallback to original classes
    CLASS_NAMES = ['B.subtilis', 'C.albicans', 'E.coli', 'P.aeruginosa', 'S.aureus']
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Using fallback model with {NUM_CLASSES} classes")

# Build model
model = build_bacterial_classifier(num_classes=NUM_CLASSES)

# Try to load expanded weights, fallback to original
if os.path.exists(EXPANDED_WEIGHTS_FILE):
    print(f"Loading expanded weights from {EXPANDED_WEIGHTS_FILE}")
    model.load_weights(EXPANDED_WEIGHTS_FILE)
elif os.path.exists(FALLBACK_WEIGHTS_FILE):
    print(f"Loading fallback weights from {FALLBACK_WEIGHTS_FILE}")
    model.load_weights(FALLBACK_WEIGHTS_FILE)
else:
    print("Warning: No weights file found. Model will use initial weights.")

def preprocess_image(image_stream):
    """Preprocesses a single image stream."""
    image = Image.open(image_stream)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Handle RGBA images by taking only RGB
        image = image[..., :3]
    if len(image.shape) == 2:  # Handle grayscale images by converting to RGB
        image = np.stack((image,)*3, axis=-1)
    return np.expand_dims(image, axis=0)

@app.route('/', methods=['GET'])
def home():
    """Renders the landing page."""
    return render_template('index.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier_tool():
    predictions_list = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file')
        if not uploaded_files or not uploaded_files[0].filename:
            predictions_list.append({'filename': 'N/A', 'error': 'No file selected or uploaded.'})
            return render_template('classifier_tool.html', predictions=predictions_list)

        for file_storage in uploaded_files:
            if file_storage and file_storage.filename:
                filename = file_storage.filename
                try:
                    processed_image = preprocess_image(file_storage.stream)
                    preds = model.predict(processed_image)
                    
                    # Get top 3 predictions
                    top_3_indices = np.argsort(preds[0])[-3:][::-1]
                    top_3_predictions = []
                    
                    for idx in top_3_indices:
                        confidence = preds[0][idx] * 100
                        species_name = CLASS_NAMES[idx]
                        top_3_predictions.append({
                            'species': species_name,
                            'confidence': f"{confidence:.2f}%"
                        })
                    
                    predictions_list.append({
                        'filename': filename,
                        'top_predictions': top_3_predictions,
                        'best_match': top_3_predictions[0]
                    })
                    
                except Exception as e:
                    print(f"Error processing file {filename} internally: {e}")
                    base_error_message = "we could not process this image. It might be an unsupported file type (please use JPG, PNG) or the file may be corrupted."
                    if "cannot identify image file" in str(e).lower() or ("no such file or directory" in str(e).lower() and ".heic" in filename.lower()):
                         user_friendly_error = f"Unfortunately, we could not process '{filename}'. HEIC/HEIF files are not currently supported. Please convert to JPG or PNG."
                    elif "image file is truncated" in str(e).lower():
                        user_friendly_error = f"Unfortunately, we could not process '{filename}'. The image file seems to be incomplete or corrupted."
                    else:
                        user_friendly_error = f"Unfortunately, {base_error_message}" 

                    predictions_list.append({
                        'filename': filename,
                        'error': user_friendly_error
                    })
                
    return render_template('classifier_tool.html', predictions=predictions_list)

@app.route('/howto')
def howto():
    return render_template('howto.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sources')
def sources():
    return render_template('sources.html')

@app.route('/species')
def species():
    """Show all available species"""
    species_info = []
    for i, species_name in enumerate(CLASS_NAMES):
        species_info.append({
            'id': i + 1,
            'name': species_name,
            'category': get_species_category(species_name)
        })
    
    return render_template('species.html', species=species_info)

def get_species_category(species_name):
    """Categorize species by type"""
    if 'candida' in species_name.lower():
        return 'Fungi'
    elif any(genus in species_name.lower() for genus in ['lactobacillus', 'bifidobacterium']):
        return 'Probiotic Bacteria'
    elif any(genus in species_name.lower() for genus in ['staphylococcus', 'streptococcus', 'enterococcus']):
        return 'Gram-Positive Cocci'
    elif any(genus in species_name.lower() for genus in ['escherichia', 'pseudomonas', 'neisseria']):
        return 'Gram-Negative Bacteria'
    elif any(genus in species_name.lower() for genus in ['clostridium', 'listeria']):
        return 'Anaerobic Bacteria'
    else:
        return 'Other Bacteria'

if __name__ == '__main__':
    print(f"Starting Microbio AI with {NUM_CLASSES} species")
    print(f"Available species: {', '.join(CLASS_NAMES[:10])}{'...' if len(CLASS_NAMES) > 10 else ''}")
    app.run(host='0.0.0.0', port=5001, debug=True) 