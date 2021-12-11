import keras
from PIL import Image, ImageOps
import numpy as np
import cv2
def teachable_machine_classification(img, weights_file):
    
    model = keras.models.load_model(weights_file)
    image = img
    def prepare(image):
        IMG_SIZE=384
        img_array = cv2.imread(image, cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 3)
     
        def predictions(prediction):
            prediction=model.predict([prepare(image)])
            return prediction
