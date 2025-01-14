from flask import Flask, request, render_template, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import logging

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/predicted'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
MODEL_PATH = "./runs/classify/train/weights/best.pt"
CLASS_NAMES = ['airplane', 'bicycles', 'cars', 'motorbikes', 'ships']
CONFIDENCE_THRESHOLD = 0.9

# Configure Flask app
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    OUTPUT_FOLDER=OUTPUT_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH
)

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize YOLO model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path):
    """Load and validate image from path."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to load image at {image_path}")
        return img
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise

def download_image(url):
    """Download image from URL and save it temporarily."""
    try:
        filename = secure_filename(os.path.basename(url))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        urllib.request.urlretrieve(url, filepath)
        return filepath
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise

def predict_image(image_path, is_upload=True):
    """Predict image class and return results"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Unable to load image at {image_path}"}, None

        # Perform prediction
        results = model.predict(source=img, save=False)

        # Process results
        if len(results) > 0 and hasattr(results[0], "probs"):
            probs = results[0].probs
            predicted_class_idx = probs.top1
            confidence = probs.top1conf.item()

            # Determine class label
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class_name = "Not in class"
            else:
                predicted_class_name = CLASS_NAMES[predicted_class_idx] if predicted_class_idx < len(CLASS_NAMES) else "Unknown class"

            # Annotate image
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_text = f"Class: {predicted_class_name}, Conf: {confidence:.2f}"
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.putText(img, label_text, (10, 30), font, 0.8, color, 2)

            # Save annotated image
            if is_upload:
                filename = os.path.basename(image_path)
            else:
                # For path input, create a new filename
                filename = f"predicted_{os.path.basename(image_path)}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            cv2.imwrite(output_path, img)

            return {
                "class": predicted_class_name,
                "confidence": confidence,
                "output_image": f"/static/predicted/{filename}"
            }, None

        return {"error": "No valid predictions found"}, None

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}, None

def annotate_image(img, class_name, confidence):
    """Annotate image with prediction results."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_text = f"Class: {class_name}, Conf: {confidence:.2f}"
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)

    # Create a copy to avoid modifying the original
    annotated = img.copy()
    cv2.putText(annotated, label_text, (10, 30), font, 0.8, color, 2)
    return annotated

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from various sources."""
    try:
        if 'file' in request.files:
            return handle_file_upload(request.files['file'])
        elif 'url' in request.form:
            return handle_url_input(request.form['url'])
        elif 'path' in request.form:
            return handle_path_input(request.form['path'])
        else:
            return jsonify({"error": "No file, URL, or path provided"}), 400

    except Exception as e:
        logger.error(f"Prediction request error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def handle_file_upload(file):
    """Handle file upload prediction requests."""
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result, error = predict_image(filepath, is_upload=True)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

def handle_url_input(url):
    """Handle URL-based prediction requests."""
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    filepath = download_image(url)
    result, error = predict_image(filepath, is_upload=True)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

def handle_path_input(image_path):
    """Handle local path prediction requests."""
    if not os.path.exists(image_path):
        return jsonify({"error": "File path does not exist"}), 400
        
    if not allowed_file(image_path):
        return jsonify({"error": "File type not allowed"}), 400
    image_path = load_image (image_path)
    result, error = predict_image(image_path, is_upload=False)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error."""
    return jsonify({"error": "File size exceeded the limit (16MB)"}), 413

if __name__ == '__main__':
    app.run(debug=True)
