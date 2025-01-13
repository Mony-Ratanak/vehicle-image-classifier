from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/predicted'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# YOLO configuration
class_names = ['airplane', 'bicycles', 'cars', 'motorbikes', 'ships']
confidence_threshold = 0.9
model_path = "./runs/classify/train/weights/best.pt"
model = YOLO(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            if confidence < confidence_threshold:
                predicted_class_name = "Not in class"
            else:
                predicted_class_name = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else "Unknown class"

            # Annotate image
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_text = f"Class: {predicted_class_name}, Conf: {confidence:.2f}"
            color = (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255)
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
        return {"error": str(e)}, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result, error = predict_image(filepath, is_upload=True)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Handle path input
    elif 'path' in request.form:
        image_path = request.form['path']
        if not os.path.exists(image_path):
            return jsonify({"error": "File path does not exist"}), 400
        
        if not allowed_file(image_path):
            return jsonify({"error": "File type not allowed"}), 400
        
        try:
            result, error = predict_image(image_path, is_upload=False)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        return jsonify({"error": "No file or path provided"}), 400

    if error:
        return jsonify({"error": error}), 500
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)