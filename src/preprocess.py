import cv2
import numpy as np

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype(np.float32) / 255.0
    return image