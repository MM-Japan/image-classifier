import os
from django.conf import settings
from django.shortcuts import render
from .forms import ImageForm
from .models import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img
import numpy as np

# Replace 'path_to_your_model.h5' with the actual path to your trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'classify', 'models', 'cifar10_model.h5')

def classify_image(request):
    model = load_model(MODEL_PATH)  # Load your pre-trained CIFAR-10 model

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_instance = form.save()  # Save the uploaded image
            img_path = img_instance.image.path

            # Preprocess the image for your CIFAR-10 model
            uploaded_image = img.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
            img_tensor = img.img_to_array(uploaded_image) / 255.0  # Normalize pixel values
            img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension

            # Make a prediction
            prediction = model.predict(img_tensor)
            predicted_class = prediction.argmax()  # Get the index of the highest probability

            # Map the predicted index to CIFAR-10 class names
            class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            label = class_names[predicted_class]

            return render(request, 'classify/result.html', {'label': label, 'image': img_instance})
    else:
        form = ImageForm()
    return render(request, 'classify/upload.html', {'form': form})
