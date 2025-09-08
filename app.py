import streamlit as st
import tensorflow as tf
from PIL import Image
from pymongo import MongoClient
from datetime import datetime
import numpy as np

# Load Trained Model (CNN)
model = tf.keras.models.load_model('ImageClassifierModel.h5')
Class_Names = ['Cat', 'Dog']

# MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['ImageClassifier']
collection = db['Predictions']

st.title('Image Classifier')
st.write('Upload an image')

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((64, 64))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    x = np.array(img) / 255.0  # normalize
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_name = Class_Names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.subheader(f'Prediction: {class_name} ({confidence:.2f}%)')

    # Save prediction to local MongoDB
    record = {
        "predicted_class": class_name,
        "confidence": confidence,
        "timestamp": datetime.now()
    }
    collection.insert_one(record)
    st.success("Prediction saved to MongoDB!")