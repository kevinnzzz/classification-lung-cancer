import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model CNN
model_cnn = tf.keras.models.load_model('model_cnn_baru_2.keras')

# Fungsi preprocessing
def preprocessing_image(image_array, resize_dim=(128, 128), blur_kernel=(5, 5), blur_sigma=0,
                         erosion_kernel=(3, 3), erosion_iterations=1, lung_threshold=100):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_kernel)
    resized = cv2.resize(image_array, resize_dim)
    blurred = cv2.GaussianBlur(resized, blur_kernel, blur_sigma)
    eroded = cv2.erode(blurred, kernel, iterations=erosion_iterations)
    gray = eroded if len(eroded.shape) == 2 else cv2.cvtColor(eroded, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, lung_threshold, 255, cv2.THRESH_BINARY)
    masked_img = cv2.bitwise_and(gray, gray, mask=mask)
    return masked_img

# Fungsi prediksi CNN
def predict_image_cnn(image_array):
    preprocessed_img = preprocessing_image(image_array)
    preprocessed_img_cnn_input = np.expand_dims(preprocessed_img, axis=0)
    preprocessed_img_cnn_input = np.expand_dims(preprocessed_img_cnn_input, axis=-1)
    preprocessed_img_cnn_input = preprocessed_img_cnn_input.astype(np.float32) / 255.0
    predictions = model_cnn.predict(preprocessed_img_cnn_input)
    prediction_label_index = np.argmax(predictions, axis=1)[0]
    if prediction_label_index == 0:
        return "Normal"
    elif prediction_label_index == 1:
        return "Jinak"
    elif prediction_label_index == 2:
        return "Ganas"
    else:
        return "Unknown"

# Judul
st.title("Klasifikasi Kanker Paru-paru CNN")
st.markdown("Upload gambar paru-paru atau pilih contoh gambar, lalu sistem akan memprediksi apakah **Normal**, **Jinak**, atau **Ganas**.")

# Pilihan gambar contoh
st.subheader("Pilih Contoh Gambar")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Gambar Normal"):
        selected_image_path = 'normal.jpg'
        selected_image = np.array(Image.open(selected_image_path))
        st.image(selected_image, caption="Normal", use_column_width=True)
        result = predict_image_cnn(selected_image)
        st.success(f"Hasil Prediksi: **{result}**")

with col2:
    if st.button("Gambar Jinak"):
        selected_image_path = 'jinak.jpg'
        selected_image = np.array(Image.open(selected_image_path))
        st.image(selected_image, caption="Jinak", use_column_width=True)
        result = predict_image_cnn(selected_image)
        st.success(f"Hasil Prediksi: **{result}**")

with col3:
    if st.button("Gambar Ganas"):
        selected_image_path = 'ganas.jpg'
        selected_image = np.array(Image.open(selected_image_path))
        st.image(selected_image, caption="Ganas", use_column_width=True)
        result = predict_image_cnn(selected_image)
        st.success(f"Hasil Prediksi: **{result}**")

st.markdown("---")

# Upload gambar
st.subheader("Upload Gambar")
uploaded_file = st.file_uploader("Pilih file gambar (format: JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    result = predict_image_cnn(image_array)
    st.success(f"Hasil Prediksi: **{result}**")

# Footer
st.markdown("---")
st.markdown("📱 Website ini responsif di laptop maupun HP ✅")

# Supaya responsive
st.markdown("""
    <style>
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)
