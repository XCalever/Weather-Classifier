import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import gdown

# Constants
file_id = '17Puq3cl919vPg8RHbQwQCeLZIeQF02rN'
url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'model_weather.hdf5'

# Download the model file if it doesn't exist
if not os.path.exists(model_path):
    st.write(f"Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((244, 244))
    img = np.asarray(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict weather
def predict_weather(image_data, model):
    img = preprocess_image(image_data)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Label mapping
weather_labels = {
    0: 'Cloudy',
    1: 'Rainy',
    2: 'Shine',
    3: 'Sunrise'
}

# Main function
def main():
    st.title('Weather Classifier System')
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Model path: {model_path}")

    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return

    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        image_display = Image.open(file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)

        predicted_class = predict_weather(image_display, model)
        predicted_label = weather_labels.get(predicted_class, 'Unknown')
        st.write(f"### Prediction: {predicted_label}")

if __name__ == '__main__':
    main()
