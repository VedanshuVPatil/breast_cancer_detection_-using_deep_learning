from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("breast_cancer_vgg19_finetuned.h5")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)[0][0]
    result = "Malignant" if prediction > 0.5 else "Benign"
    
    return jsonify({"prediction": result, "confidence": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
