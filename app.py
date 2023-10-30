import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

def predict(image):
    loaded_model = tf.keras.models.load_model("./assets/digits_model.h5")

    # Resize the image using Pillow
    image = image.resize((28, 28), Image.LANCZOS)
    image = image.convert('L')
    image = np.array(image)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.bitwise_not(image)
    image = image / 255.0

    prediction = np.argmax(loaded_model.predict(image[np.newaxis, :, :, np.newaxis]))
    return prediction

st.title("Recognize Digit")

uploaded_file = st.file_uploader("Upload hand written digit image")
if uploaded_file is not None:
    # display image
    image = Image.open(uploaded_file)
    st.write("You uploaded this")
    st.image(image)
    st.divider()

    if st.button("What digit am I?"):
        prediction = predict(image)
        st.header(f"It's {prediction}.")
