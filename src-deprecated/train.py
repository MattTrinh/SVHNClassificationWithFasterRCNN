import tensorflow as tf
import numpy as np
import pandas as pd

from config import Config
from rpn import *

from keras.layers import Input

def build_model(img_size):
    config = Config()
    
    # Input layer(s)
    img_input = Input(shape=img_size)
    
    # TODO Put pre-trained CNN layers here
    trained_cnn = ""
    
    # RPN Layers
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = create_rpn(num_anchors, trained_cnn)
    
    # TODO Put R-CNN classification and bounding box layers here
    classifier = ""
    
    # Build model
    model = Model()
    model.compile(optimizer='adam')
    
def train():
    return 0