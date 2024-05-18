import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import traceback

@st.cache_resource
def load_model():
    model_path = 'model_weather.hdf5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_data):
    try:
        img = Image.open(image_data)
        img = img.convert('RGB')
        img = img.resize((244, 244))
        img = np.asarray(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        st.error(traceback.format_exc())
        return None

def predict_weather(image_data, model):
    if model is None:
        return "Model not loaded"
    
    img = preprocess_image(image_data)
    if img is None:
        return "Image preprocessing error"
    
    try:
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error(traceback.format_exc())
        return "Prediction error"

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
    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        image_display = Image.open(file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)

        if model:
            predicted_class = predict_weather(file, model)
            if predicted_class == "Model not loaded":
                st.error(predicted_class)
            elif predicted_class == "Prediction error" or predicted_class == "Image preprocessing error":
                st.error(predicted_class)
            else:
                predicted_label = weather_labels.get(predicted_class, 'Unknown')
                st.write(f"### Prediction: {predicted_label}")
        else:
            st.error("Model not loaded")

if __name__ == '__main__':
    main()
