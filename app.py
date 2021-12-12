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
uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = load_img(image, target_size=(384,384))

        img = img_to_array(img)

        img = np.expand_dims(img,axis = 0)
        st.image(img, caption='Uploaded MRI.', use_column_width=True)
        st.write("Uploaded")
        st.write("Classifying...")
        
        
     
            
        prediction=model.predict(image)
                
        
        if prediction <= 0.5:
            st.write("The MRI scan detected a brain tumor")
        else:
            st.write("The MRI scan shows is healthy brain")
