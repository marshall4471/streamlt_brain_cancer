import keras
from PIL import Image, ImageOps
import numpy as np
import cv2
def teachable_machine_classification(img, prediction, weights_file):
    
    model = keras.models.load_model(weights_file)
    image = img
