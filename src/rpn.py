import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
from keras import backend as K

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
    
# Loss function for bounding boxes
def rpn_loss_box(num_anchors):
    # Need to pass symbolic function that takes the true labels and predictions as arguments
    # See: https://keras.io/losses/
    def rpn_loss_box_fixed_num(y_true, y_pred):
        balanced_weight = 1.0
        ground_truth = y_true[:, :, :, :4 * num_anchors]
        n_reg = K.sum(1e-4 + ground_truth)
        
        # Calculate Huber Loss
        # Note that Huber Loss is a robust loss function
        error = y_pred - y_true[:, :, :, 4 * num_anchors:]
        abs_error = K.abs(error)
        quadratic = K.minimum(abs_error, 1.0)
        linear = abs_error - quadratic
        huber_loss = 0.5 * K.square(quadratic) + 1.0 * linear
        
        return balanced_weight * K.sum(ground_truth * huber_loss) / n_reg
    return rpn_loss_box_fixed
    
# Loss function for positive/negative regions (i.e. background or object)
def rpn_loss_bin(num_anchors):
    # Need to pass symbolic function that takes the true labels and predictions as arguments
    # See: https://keras.io/losses/
    def rpn_loss_bin_fixed_num(y_true, y_pred):
        ground_truth = y_true[:, :, :, :num_anchors]
        predictions = y_pred[:, :, :, :]
        n_cls = K.sum(1e-4 + ground_truth)
        
        # Note that binary cross entropy loss is a log loss function
        return K.sum(K.binary_crossentropy(predictions, ground_truth)) / n_cls