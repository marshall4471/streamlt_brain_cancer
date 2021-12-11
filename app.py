import streamlit as st
st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import keras
import cv2
model=load_model('model.h5')
model=load_model('model.h5')
uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("Uploaded")
        st.write("Classifying...")
        def prepare(image):
            IMG_SIZE=384
            img_array = cv2.imread(image, cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 3)
     
            
        prediction=model.predict([prepare(image)])
                
        label = teachable_machine_classification(image, prediction, 'model.h5')
        if prediction <= 0.5:
            st.write("The MRI scan detected a brain tumor")
        else:
            st.write("The MRI scan shows is healthy brain")
