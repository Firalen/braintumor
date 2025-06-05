from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import csv

app = Flask(__name__)

# Configure paths for deployment
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'braintumor.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.csv')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define your class names
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Load model with error handling
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('error.html', error="Model not loaded properly. Please contact administrator.")
        
    if 'file' not in request.files:
        return render_template('error.html', error="No file uploaded")
        
    file = request.files['file']
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    
    if file.filename == '' or not patient_name or not patient_age:
        return render_template('error.html', error="Missing required fields")
        
    try:
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            result = class_names[predicted_index]
            confidence = f"{predictions[predicted_index] * 100:.2f}%"
            image_path = f"uploads/{file.filename}"

            # Save to history.csv
            with open(HISTORY_FILE, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([patient_name, patient_age, file.filename, result, confidence, image_path])

            return render_template(
                'result.html',
                result=result,
                confidence=confidence,
                image_path=image_path,
                patient_name=patient_name,
                patient_age=patient_age,
                photo_name=file.filename
            )
    except Exception as e:
        return render_template('error.html', error=f"An error occurred: {str(e)}")

@app.route('/history')
def history():
    records = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, newline='') as csvfile:
                reader = csv.reader(csvfile)
                records = list(reader)
        except Exception as e:
            return render_template('error.html', error=f"Error reading history: {str(e)}")
    return render_template('history.html', records=records)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    try:
        row_id = int(request.form['row_id'])
        records = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, newline='') as csvfile:
                reader = csv.reader(csvfile)
                records = list(reader)
            if 0 <= row_id < len(records):
                image_path = os.path.join('static', records[row_id][5])
                if os.path.exists(image_path):
                    os.remove(image_path)
                records.pop(row_id)
            with open(HISTORY_FILE, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(records)
        return redirect(url_for('history'))
    except Exception as e:
        return render_template('error.html', error=f"Error deleting history: {str(e)}")

@app.route('/form')
def form_page():
    return render_template('form.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)