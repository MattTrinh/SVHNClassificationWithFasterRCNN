import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import h5py

import keras_rcnn.datasets.svhn as ds
import keras_rcnn.models as m
import keras_rcnn.preprocessing as pp

TRAINING_DIR = "keras_rcnn/data/svhn/train"
TEST_DIR = "keras_rcnn/data/svhn/test"

def main():
    training, testing = ds.load_data()

    categories = {str(i):i for i in range(10)}
    
    generator = pp.ObjectDetectionGenerator()
    generator = generator.flow_from_dictionary(
        dictionary=training,
        categories=categories,
        target_size=(256,256)
    )
    
    validation_data = pp.ObjectDetectionGenerator()
    validation_data = validation_data.flow_from_dictionary(
        dictionary=testing,
        categories=categories,
        target_size=(256,256)
    )
    
    K.set_learning_phase(1)
    
    model = m.RCNN(
        categories=categories,
        dense_units=512,
        input_shape=(256, 256, 3)
    )
    
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer)
    model.fit_generator(
        epochs=1,
        steps_per_epoch=200,
        generator=generator,
        validation_data=validation_data,
        validation_steps=200
    )

if __name__ == "__main__":
    main()