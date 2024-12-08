import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

# LOAD MODEL
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# LOAD CLASS LABELS
def load_class_labels(train_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    return list(generator.class_indices.keys())

# PREPROCESS IMAGE
def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# PREDICT IMAGE CLASS
def predict_image_class(img_array, class_labels):
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    return predicted_class, confidence

# STREAMLIT APP
st.title("Animal Classifier")
st.write("Upload gambar hewan untuk memprediksi kelasnya.")

# INPUT FILE UPLOAD
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    # Load Class Labels
    TRAIN_DIR = "Dataset/Training"  # Update path sesuai dataset Anda
    class_labels = load_class_labels(TRAIN_DIR)

    # Preprocess and Predict
    img_array = preprocess_image(uploaded_file)
    predicted_class, confidence = predict_image_class(img_array, class_labels)

    # Display Result
    st.success(f"Prediksi: **{predicted_class}**")
    st.info(f"Kepercayaan: **{confidence * 100:.2f}%**")
