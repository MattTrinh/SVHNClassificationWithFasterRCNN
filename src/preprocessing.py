import numpy as np
from scipy.io import loadmat
import pandas as pd

# Loads data file and splits the data into images and their respective labels
# TODO: Handle case where data does not contain labels
def load(file_name):
    file = loadmat(file_name)
    
    # Images are a 4D array with [row, column, channel, image] as dimensions
    raw_images = file['X']
    raw_labels = file['y']
    
    # Restructure image data into an array of images 
    images = np.transpose(raw_images, (3, 0, 1, 2))
    
    # Restructure label data into an array of labels
    labels = np.array([label[0] for label in raw_labels])
    
    return [images, labels]
