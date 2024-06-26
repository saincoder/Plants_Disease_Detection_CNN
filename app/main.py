import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Define the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")

st.markdown(
    """
    <style>
    .main {
        background-color: #0d1016;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(to right, #11998e, #38ef7d);
        border: none;
        padding: 0.5em 1em;
        border-radius: 4px;
        font-size: 1em;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #38ef7d, #11998e);
    }
    .stFileUploader>div>div {
        border: 2px dashed #11998e;
        padding: 1em;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🌿 Plant Disease Classifier')
st.write("Developed by **Shahid Hussain**")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    footer:after {
        content:'Developed by Shahid Hussain';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
