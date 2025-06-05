from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the absolute path to the model file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'braintumor.h5')
logger.info(f"Model path: {MODEL_PATH}")

# Load model with error handling
try:
    logger.info("Attempting to load model...")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Configure upload folder with absolute path
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
logger.info(f"Upload folder: {UPLOAD_FOLDER}")

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define your class names
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    
    if model is None:
        logger.error("Model is not loaded")
        return render_template('error.html', error="Model not loaded properly. Please contact administrator.")
        
    if 'file' not in request.files:
        logger.error("No file in request")
        return render_template('error.html', error="No file uploaded")
        
    file = request.files['file']
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    
    logger.info(f"Processing request for patient: {patient_name}, age: {patient_age}")
    
    if file.filename == '' or not patient_name or not patient_age:
        logger.error("Missing required fields")
        return render_template('error.html', error="Missing required fields")
        
    try:
        if file:
            # Save file with secure filename
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")
            file.save(filepath)

            # Process image
            logger.info("Loading and processing image")
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            logger.info("Making prediction")
            predictions = model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            result = class_names[predicted_index]
            confidence = f"{predictions[predicted_index] * 100:.2f}%"
            image_path = f"uploads/{filename}"
            logger.info(f"Prediction result: {result} with confidence: {confidence}")

            # Save to history.csv
            history_file = os.path.join(BASE_DIR, 'history.csv')
            logger.info(f"Saving to history file: {history_file}")
            with open(history_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([patient_name, patient_age, filename, result, confidence, image_path])

            return render_template(
                'result.html',
                result=result,
                confidence=confidence,
                image_path=image_path,
                patient_name=patient_name,
                patient_age=patient_age,
                photo_name=filename
            )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return render_template('error.html', error=f"An error occurred during prediction: {str(e)}")

@app.route('/history')
def history():
    records = []
    history_file = os.path.join(BASE_DIR, 'history.csv')
    logger.info(f"Reading history from: {history_file}")
    if os.path.exists(history_file):
        try:
            with open(history_file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                records = list(reader)
        except Exception as e:
            logger.error(f"Error reading history: {str(e)}", exc_info=True)
            return render_template('error.html', error="Error reading history file")
    return render_template('history.html', records=records)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    try:
        row_id = int(request.form['row_id'])
        records = []
        history_file = os.path.join(BASE_DIR, 'history.csv')
        logger.info(f"Deleting history record {row_id} from: {history_file}")
        
        if os.path.exists(history_file):
            with open(history_file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                records = list(reader)
            if 0 <= row_id < len(records):
                image_path = os.path.join('static', records[row_id][5])
                if os.path.exists(image_path):
                    os.remove(image_path)
                records.pop(row_id)
            with open(history_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(records)
        return redirect(url_for('history'))
    except Exception as e:
        logger.error(f"Error deleting history: {str(e)}", exc_info=True)
        return render_template('error.html', error="Error deleting history record")

@app.route('/form')
def form_page():
    return render_template('form.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
    