import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model():
    model_path = 'model_weather.hdf5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_data):
    try:
        img = Image.open(image_data)
        img = img.convert('RGB')
        img = img.resize((244, 244))  # Assuming the model expects 244x244 images
        img = np.asarray(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error in image preprocessing: {e}")
        return None

def predict_weather(image_data, model):
    if model is None:
        return "Model not loaded"
    img = preprocess_image(image_data)
    if img is None:
        return "Error in image preprocessing"
    try:
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "Error during model prediction"

weather_labels = {
    0: 'Cloudy',
    1: 'Rainy',
    2: 'Shine',
    3: 'Sunrise'
}

def main():
    st.title('Weather Classifier System')
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Model path: model_weather.hdf5")

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please ensure the model file exists and is accessible.")
        return

    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        try:
            image_display = Image.open(file)
            st.image(image_display, caption='Uploaded Image', use_column_width=True)

            predicted_class = predict_weather(file, model)
            if predicted_class == "Model not loaded" or predicted_class == "Error in image preprocessing" or predicted_class == "Error during model prediction":
                st.error(predicted_class)
            else:
                predicted_label = weather_labels.get(predicted_class, 'Unknown')
                st.write(f"### Prediction: {predicted_label}")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

if __name__ == '__main__':
    main()
