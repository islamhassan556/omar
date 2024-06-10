# from flask import Flask, request, jsonify
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the trained model
# model = load_model('asl_model_39_classes.keras')

# # Define the class labels (replace with your actual class labels)
# class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
#                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
#                 'U', 'V', 'W', 'X', 'Y', 'Z', 'best of luck', 'i love you', 'space']

# def preprocess_image(image_path):
#     """Preprocess the input image to be compatible with the model."""
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or unable to read")

#     # Resize the image to the target size (64x64)
#     image = cv2.resize(image, (64, 64))
#     # Normalize the image
#     image = image / 255.0
#     # Expand dimensions to match the input shape of the model (1, 64, 64, 3)
#     image = np.expand_dims(image, axis=0)
#     return image

# def predict_image(image_path):
#     """Predict the class label of the input image."""
#     preprocessed_image = preprocess_image(image_path)
#     predictions = model.predict(preprocessed_image)
#     predicted_class = np.argmax(predictions)
#     predicted_label = class_labels[predicted_class]
#     return predicted_label

# @app.route('/predict', methods=['POST'])
# def predict():
#     print(request.files)
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file in request"}), 400
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#     # Continue with processing the file

#     # Save the file to a temporary location
#     image_path = "temp_image.png"
#     file.save(image_path)

#     try:
#         predicted_label = predict_image(image_path)
#         return jsonify({"predicted_label": predicted_label})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('sign_language_cnn_model.joblib')

# Define a route to handle image predictions
@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")  # Debug: Confirm request received

    if 'file' not in request.files:
        print("No file part in the request")  # Debug: File not in request
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    if file.filename == '':
        print("No file selected for uploading")  # Debug: No file selected
        return jsonify({'error': 'No file selected'}), 400

    print(f"File received: {file.filename}")  # Debug: File received

    try:
        # Ensure file is loaded correctly
        image = load_img(file, target_size=(64, 64))
        print("Image loaded")  # Debug: Image loaded

        # Convert image to array and preprocess
        image = img_to_array(image)
        print("Image converted to array")  # Debug: Image to array
        image = np.expand_dims(image, axis=0)
        print("Image expanded dimensions")  # Debug: Image expanded
        image = image / 255.0  # Rescale
        print("Image rescaled")  # Debug: Image rescaled

        # Make a prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        print(f"Prediction made: {predicted_class}")  # Debug: Prediction made
        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        print(f"Error processing file: {e}")  # Debug: Error processing file
        return jsonify({'error': 'File processing error'}), 500

if __name__ == '__main__':
    print("Starting Flask server...")  # Debug: Starting server
    app.run(debug=True)




