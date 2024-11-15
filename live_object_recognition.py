import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your model (ensure the correct model path)
MODEL_PATH = '/home/max/code/MM-Japan/image-classification/image_classifier/classify/models/cifar10_model.h5'
model = load_model(MODEL_PATH)

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

def preprocess_frame(frame):
    """Resize, normalize, and prepare the frame for prediction."""
    resized_frame = cv2.resize(frame, (32, 32))
    normalized_frame = resized_frame.astype('float32') / 255.0
    img_array = img_to_array(normalized_frame)
    return np.expand_dims(img_array, axis=0)

def predict_class(frame):
    """Predict the class of the object in the frame."""
    prediction = model.predict(frame)
    return class_names[np.argmax(prediction)]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess and predict
    preprocessed_frame = preprocess_frame(frame)
    # predicted_class = predict_class(preprocessed_frame)
    predicted_class = "Dog"
    # Draw rectangle and display prediction
    h, w, _ = frame.shape
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 255, 0), 2)
    cv2.putText(frame, f"Class: {predicted_class}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Live Object Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
