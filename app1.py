import streamlit as st

# Attempt to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from PIL import Image
    import numpy as np
    st.success("TensorFlow imported successfully!")
except ImportError as e:
    st.error(f"Failed to import TensorFlow. Please check your environment setup: {e}")

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_weather.hdf5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")

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

# Mapping from predicted class index to weather class labels
weather_labels = {
    0: 'cloudy',
    1: 'rainy',
    2: 'sunny'
    # Add more classes as per your model's output
}

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
