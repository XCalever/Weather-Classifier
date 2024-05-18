import streamlit as st

# Attempt to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from PIL import Image
    import numpy as np
except ImportError:
    st.error("Failed to import TensorFlow. Please check your environment setup.")

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_weather.hdf5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Main Streamlit app
def main():
    st.title('Weather Classifier System')

    # Load the model
    model = load_model()

    # File uploader for image
    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Display the uploaded image
        image_display = Image.open(file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        predicted_class = predict_weather(file, model)
        predicted_label = weather_labels.get(predicted_class, 'Unknown')

        # Display the prediction result
        st.write("### Prediction:")
        st.write(f"The predicted weather is: {predicted_label}")

# Execute the main function
if __name__ == '__main__':
    main()
