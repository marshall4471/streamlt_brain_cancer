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
from io import BytesIO
model=load_model('model.h5')
uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)  

        img2 = image.crop((1,20,50,80))

        b = BytesIO()
        img2.save(b,format="jpeg")
        new_image = Image.open(b)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("Uploaded")
        st.write("Classifying...")
        def prepare(new_image):
            IMG_SIZE=384
            img_array = cv2.imread(new_image, cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 3)
     
            
        prediction=model.predict([prepare(new_image)])
                
        label = teachable_machine_classification(image, prediction, 'model.h5')
        if prediction <= 0.5:
            st.write("The MRI scan detected a brain tumor")
        else:
            st.write("The MRI scan shows is healthy brain")
