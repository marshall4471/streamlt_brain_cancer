import streamlit as st
st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
from img_classification import teachable_machine_classification
uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("Uploaded")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'model.h5')
        if label == 0:
            st.write("The MRI scan has a brain tumor")
        else:
            st.write("The MRI scan is healthy")
