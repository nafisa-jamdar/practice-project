import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from keras.models import model_from_json

# Initialize Flask app
app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the model
with open('model_vgg.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Fix potential issue with softmax layer
loaded_model_json = loaded_model_json.replace('softmax_v2', 'softmax')

# Load model from JSON
model = model_from_json(loaded_model_json)
model.load_weights("model_vgg.weights.h5")

# Constants
IMAGE_SIZE = 64
LABELS = ['glaucoma', 'normal', 'cataract', 'Diabetic retina']

# Preprocess image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    image = tf.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    return image

# Prediction function
def classify(image_path):
    processed = preprocess_image(image_path)
    prob = model.predict(processed)[0]
    label_index = np.argmax(prob)
    return LABELS[label_index], float(prob[label_index])

# Routes
@app.route("/")
def home():
    return render_template("home.html")

# @app.route("/classify", methods=["POST"])
# def upload_file():
#     if "image" not in request.files:
#         return "No file uploaded.", 400

#     file = request.files["image"]
#     if file.filename == "":
#         return "No file selected.", 400

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     label, prob = classify(file_path)
#     prob_percent = round(prob * 100, 2)

#     return render_template("classify.html", image_file_name=file.filename, label=label, prob=prob_percent)
@app.route("/classify", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return "No file uploaded.", 400

    files = request.files.getlist("image")  # Get list of uploaded files
    if not files or files[0].filename == "":
        return "No file selected.", 400

    results = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        label, prob = classify(file_path)
        prob_percent = round(prob * 100, 2)

        results.append((file.filename, label, prob_percent))

    return render_template("classify.html", results=results)

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()
