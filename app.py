from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os # Added for checking weights file existence
from model import build_bacterial_classifier

app = Flask(__name__)

# Updated class names based on training output
# Original: ['E_coli', 'S_aureus', 'Contaminated', 'Clean_plate', 'Other']
CLASS_NAMES = ['B.subtilis', 'C.albicans', 'E.coli', 'P.aeruginosa', 'S.aureus']
NUM_CLASSES = len(CLASS_NAMES)

model = build_bacterial_classifier(num_classes=NUM_CLASSES)

# Path to the new weights file
NEW_WEIGHTS_FILE = 'updated_bacterial_classifier.weights.h5'

if os.path.exists(NEW_WEIGHTS_FILE):
    print(f"Loading weights from {NEW_WEIGHTS_FILE}")
    model.load_weights(NEW_WEIGHTS_FILE)
else:
    print(f"Warning: Weights file {NEW_WEIGHTS_FILE} not found. Model will use initial weights.")
    # Optionally, you could load default weights or raise an error here
    # For example, if you have an old weights file: model.load_weights('old_weights.h5')

# class_names = ['E_coli', 'S_aureus', 'Contaminated', 'Clean_plate', 'Other'] # Old class names

def preprocess_image(image_stream):
    """Preprocesses a single image stream."""
    image = Image.open(image_stream)
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4: # Handle RGBA images by taking only RGB
        image = image[..., :3]
    if len(image.shape) == 2: # Handle grayscale images by converting to RGB
        image = np.stack((image,)*3, axis=-1)
    return np.expand_dims(image, axis=0)

@app.route('/', methods=['GET'])
def home():
    """Renders the new landing page."""
    return render_template('index.html') # This will be the new landing page template

@app.route('/classifier', methods=['GET', 'POST'])
def classifier_tool():
    predictions_list = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file')
        if not uploaded_files or not uploaded_files[0].filename: # Check if any file was actually uploaded
            predictions_list.append({'filename': 'N/A', 'error': 'No file selected or uploaded.'})
            return render_template('classifier_tool.html', predictions=predictions_list)

        for file_storage in uploaded_files:
            if file_storage and file_storage.filename: # Ensure it's a valid FileStorage object with a filename
                filename = file_storage.filename
                try:
                    processed_image = preprocess_image(file_storage.stream)
                    preds = model.predict(processed_image)
                    idx = np.argmax(preds)
                    prediction_text = f"{CLASS_NAMES[idx]} ({preds[0][idx]*100:.2f}%)"
                    predictions_list.append({
                        'filename': filename,
                        'prediction_text': prediction_text
                    })
                except Exception as e:
                    print(f"Error processing file {filename} internally: {e}") # Server-side log
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
            # else: # This case should be handled by the initial check, but good for robustness
                # predictions_list.append({'filename': 'N/A', 'error': 'Invalid file uploaded.'})
                
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)