from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

# Needed to write custom RoI Pooling layer as Keras does not have
# one of these
# The RoI Pooling layer uses the output of the CNN and RPN as inputs
class RoiPoolingLayer(Layer):
    def __init__(self, num_rois, pool_size, **kwargs):
        self.num_rois = num_rois
        self.pool_size = pool_size
        
        super(RoiPoolingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.num_channels = input_shape[0][3]
        
    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.num_channels, self.pool_size, self.pool_size
    
    def call(self, x):
        # Get image and proposed regions
        image, regions = x
        outputs = []
        
        # Resize all RoIs to the pooling size
        # Assuming regions are in the format (x, y, w, h)
        for region in regions[0]:
            region = K.cast(region, 'int32')
            x, y, w, h = region
        
            resize = tf.image.resize_images(image[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)
        
        # Combine all the RoIs
        pooled_rois = K.concatenate(outputs, axis=0)
        
        # Reshape to fit output shape
        pooled_rois = K.reshape(pooled_rois, (1, self.num_rois, self.num_channels, self.pool_size, self.pool_size)
        pooled_rois = K.permute_dimensions(pooled_rois, (0, 1, 2, 3, 4))
        
        return pool_rois