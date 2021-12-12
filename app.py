import streamlit as st
st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or Healthy Brain")
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras

     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = load_img(image, color_mode="rgb", target_size=(384, 384))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    st.write("")
    label = teachable_machine_classification(image, 'model.h5')
    if label <= 0.5:
       st.write("The MRI scan detects a brain tumor")
    else:
       st.write("The MRI scan shows an healthy brain")
   
        
        
