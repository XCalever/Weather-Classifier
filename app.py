import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_weather.hdf5')
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
    img = preprocess_image(image_data)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

weather_labels = {
    0: 'Cloudy',
    1: 'Rainy',
    2: 'Sunny',
    3: 'Sunrise'}

def main():
    st.title('Weather Classifier System')
    model = load_model()
    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        image_display = Image.open(file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)

        predicted_class = predict_weather(file, model)
        predicted_label = weather_labels.get(predicted_class, 'Unknown')

        st.write("### Prediction:")
        st.write(f"The predicted weather is: {predicted_label}")

if __name__ == '__main__':
    main()
