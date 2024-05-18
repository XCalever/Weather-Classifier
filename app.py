import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    model_path = 'model_weather.hdf5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    model = tensorflow.keras.models.load_model(model_path)
    return model

# Function to preprocess the image for prediction
def preprocess_image(image_data):
    img = Image.open(image_data)
    img = img.convert('RGB')  # Ensure image is RGB (3 channels)
    img = img.resize((244, 244))  # Resize image to match model's expected sizing
    img = np.asarray(img)  # Convert PIL image to numpy array
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the weather class from the image
def predict_weather(image_data, model):
    if model is None:
        return "Model not loaded"
    img = preprocess_image(image_data)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

weather_labels = {
    0: 'Cloudy',
    1: 'Rainy',
    2: 'Shine',
    3: 'Sunrise'}

def main():
    st.title('Weather Classifier System')

    # Debug: Print the current working directory and model path
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Model path: model_weather.hdf5")

    model = load_model()
    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        image_display = Image.open(file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)

        predicted_class = predict_weather(file, model)
        if predicted_class == "Model not loaded":
            st.error(predicted_class)
        else:
            predicted_label = weather_labels.get(predicted_class, 'Unknown')
            st.write(f"### Prediction: {predicted_label}")

if __name__ == '__main__':
    main()
