import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="CNN Img Classification",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add custom CSS to set the background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.shutterstock.com/image-vector/cute-animals-watercolor-effect-600nw-1687590076.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model_path = os.path.join("C:\\Users\\farhan\\Desktop\\Data science\\mini project 1\\models\\imageclassifier.h5")
if not os.path.exists(model_path):
    st.error("Model file not found. Ensure 'models/imageclassifier.h5' exists.")
else:
    model = tf.keras.models.load_model(model_path)

    # Function to preprocess the image
    def preprocess_image(image):
        img_array = np.array(image)
        resize = tf.image.resize(img_array, (256,256))
        return resize.numpy().astype('float32') / 255.0

    # Title of the app
    st.title('Puppy and Kitten Image Classifier')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp"],accept_multiple_files=False,)

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False,width=300)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        yhat = model.predict(np.expand_dims(processed_image, axis=0))

        # Display the prediction
        if yhat > 0.5:
            st.write(f'Predicted class is Puppies')
        else:
            st.write(f'Predicted class is Kitten')
