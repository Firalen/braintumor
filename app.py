from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import csv


app = Flask(__name__)
model = load_model("braintumor.h5")  # Make sure this is a valid Keras .h5 model

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define your class names at the top of your file
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

@app.route('/')
def index():
    return render_template('index.html')

import csv

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    if file.filename == '' or not patient_name or not patient_age:
        return redirect(request.url)
    if file:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Get prediction probabilities for each class
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        result = class_names[predicted_index]
        confidence = f"{predictions[predicted_index] * 100:.2f}%"
        image_path = f"uploads/{file.filename}"

        # Save to history.csv
        with open('history.csv', 'a', newline='') as csvfile:
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

@app.route('/history')
def history():
    records = []
    if os.path.exists('history.csv'):
        with open('history.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            records = list(reader)
    return render_template('history.html', records=records)
@app.route('/delete_history', methods=['POST'])
def delete_history():
    import csv, os
    row_id = int(request.form['row_id'])
    records = []
    if os.path.exists('history.csv'):
        with open('history.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            records = list(reader)
        if 0 <= row_id < len(records):
            # Optionally delete the image file as well
            image_path = os.path.join('static', records[row_id][5])
            if os.path.exists(image_path):
                os.remove(image_path)
            records.pop(row_id)
        with open('history.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(records)
    return redirect(url_for('history'))
@app.route('/form')
def form_page():
    return render_template('form.html')
if __name__ == "__main__":
    app.run(debug=True)