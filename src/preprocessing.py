import numpy as np
from scipy.io import loadmat
import pandas as pd
import cv2 as cv

# Loads data file and splits the data into images and their respective labels
# TODO implement entire system with the preprocessed data because R-CNN needs bounding boxes
def load(file_name, preprocessed=False):
    if preprocessed:
        file = loadmat(file_name)
        
        # Images are a 4D array with [row, column, channel, image] as dimensions
        raw_images = file['X']
        raw_labels = file['y']
        
        # Restructure image data into an array of images 
        images = np.transpose(raw_images, (3, 0, 1, 2))
        
        # Restructure label data into an array of labels
        labels = np.array([label[0] for label in raw_labels])
        
        return [images, labels]
    else:
        images = []
        
def resize_img(original, min_side):
    width, height = original
    resized_size = (-1, -1)
    scaling_factor = -1
    
    # Recalcuate size
    if width >= height: # Shortest side is height
        scaling_factor = float(min_side) / height
        resized_size = (int(scaling_factor * width), min_side)
    else: # Shortest side is width
        scaling_factor = float(min_side) / width
        resized_size = (min_side, int(scaling_factor * height))
        
    if scaling_factor >= 1: # Image is enlarged
        resized = cv.resize(original, resized_size, interpolation=cv.INTER_CUBIC)
    else: # Image is shrunk
        resize = cv.resize(original, resized_size, interpolation=cv.INTER_AREA)