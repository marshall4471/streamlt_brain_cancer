import keras
from PIL import Image, ImageOps
def teachable_machine_classification(img, prediction, weights_file):
    
    model = keras.models.load_model(weights_file)
    image = img
