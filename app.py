import streamlit as st
st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras
import cv2
model=load_model('model.h5')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    def predict(image_path): 
        image = load_img(image_path, target_size=(384, 384))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        label = decode_predictions(yhat)
     
        label = label[0][0]
        return label 
label = model.predict(uploaded_file)
if label <= 0.5:
   st.write("The MRI scan has a brain tumor")
else:
   st.write("The MRI scan is healthy")
   
        
        
