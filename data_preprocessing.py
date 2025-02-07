import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #BGR to RGB
    return image

def preprocess_image(image, size=(640, 640)):
    image_resized = cv2.resize(image, size)
    image_normalized = image_resized / 255.0  #Normalize pixel values to 0-1
    return image_normalized

def load_data(data_dir):
    data = []
    for img_file in Path(data_dir).rglob("*.jpg"):  #dataset
        img = load_image(str(img_file))
        img = preprocess_image(img)
        data.append(img)
    return np.array(data)

# Test the function (example)
data = load_data('data/')
print(f"Loaded {len(data)} images for training.")