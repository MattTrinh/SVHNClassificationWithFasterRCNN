import numpy as np
import tensorflow as tf
from keras.layers import Conv2D

# Creates and returns the layers for the RPN
def create_rpn(num_anchors, shared_layers):
    # Convolution layer of RPN
    conv_layer = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal')(shared_layers)
    
    # Binary category prediction layer (i.e. whether region contains an object we're interested in or is a background)
    # In original paper, this is a softmax activation, but since it is binary sigmoid can be used
    bin_layer = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform')(conv_layer)
    
    # Bounding box prediction layer
    box_layer = Conv2D(num_anchors * 4, (1,1), activation='linear', kernel_initializer='zero')(conv_layer)
    
    return bin_layer, box_layer
    
# Wrapper for Tensorflow's non-maximum suppression that returns the indices of boxes and scores to keep
# TODO test performance with fast NMS found below
# https://github.com/you359/Keras-FasterRCNN/blob/master/keras_frcnn/roi_helpers.py
def nms(boxes, scores, max_proposals, overlap_thresh):
    indices = tf.image.non_max_suppression(boxes, scores, max_proposals, overlap_thresh)
    return indices