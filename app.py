import streamlit as st
st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("Uploaded")
        st.write("Classifying...")
       
        label = teachable_machine_classification(image, 'model.h5')
        final_prediction = predictions(prediction, label)
        if final_prediction <= 0.5:
            st.write("The MRI scan detected a brain tumor")
        else:
            st.write("The MRI scan shows is healthy brain")
