import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import h5py
from datetime import datetime
import os
from matplotlib import pyplot
import keras_rcnn
import keras_rcnn.datasets.svhn as ds
import keras_rcnn.callbacks._tensorboard as callback
import keras_rcnn.models as m
import keras_rcnn.preprocessing as pp
import keras_rcnn.metrics.mean_average_precision as met

TRAINING_DIR = "keras_rcnn/data/svhn/train"
TEST_DIR = "keras_rcnn/data/svhn/test"
LOG_DIR = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    start = datetime.now()
    print("Run started at: "+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    WEIGHT_DIR = "saved_models"

    # Load in data
    training, testing = ds.load_data()
    categories = {str(i):i for i in range(10)}

    def schedule(epoch_index):
        return 0.1 * np.power(0.5, np.floor((1 + epoch_index) / 1.0))

    
    
    # Augment data via data generators
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
    
    #K.set_learning_phase(1)
    
    # Show bounding boxes
    """ target, _ = generator.next()
    target_bounding_boxes, target_categories, target_images, target_masks, target_metadata = target
    target_bounding_boxes = np.squeeze(target_bounding_boxes)
    target_images = np.squeeze(target_images)
    target_categories = np.argmax(target_categories, -1)
    target_categories = np.squeeze(target_categories)
    keras_rcnn.utils.show_bounding_boxes(target_images, target_bounding_boxes, target_categories)
         """
    # Build R-CNN
    model = m.RCNN(
        categories=categories,
        dense_units=512,
        input_shape=(256, 256, 3)
    )
    
    # Train R-CNN
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer, metrics=['acc'])
    history = model.fit_generator(
        epochs=1,
        steps_per_epoch=20,
        generator=generator,
        validation_data=validation_data,
        validation_steps=20,
        verbose=2
        #callbacks = [
        #    keras.callbacks.LearningRateScheduler(schedule)
        #]
    )
    pyplot.plot(history.history['acc'])
    pyplot.show()
    # Save model weights
    # Do not use "model.save()" because this implementation is considered a 'subclassed model'
    # refer to documentation here (https://www.tensorflow.org/guide/keras/save_and_serialize#saving_subclassed_models) 
    # for more details
    model.save_weights(WEIGHT_DIR + "/epoch_1_images_50.h5")
    timeElapsed = datetime.now()-start
    print(("Run completed in %4d" % timeElapsed.seconds) + " seconds at " + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
def load(file_name):
    # Build model
    categories = {str(i):i for i in range(10)}
    
    generator = pp.ObjectDetectionGenerator()
    generator = generator.flow_from_dictionary(
        dictionary=training,
        categories=categories,
        target_size=(256,256)
    )
    
    model = m.RCNN(
        categories=categories,
        dense_units=512,
        input_shape=(256, 256, 3)
    )
    
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer, metrics=['accuracy', met.evaluate])
    
    # Initialize variables used by optimizers
    history = model.fit_generator(
        epochs=1,
        steps_per_epoch=1,
        generator=generator
    )
    
    # Load weights from file
    model.load_weights(WEIGHT_DIR + "/" + file_name)
    
    return model

    
    
    


if __name__ == "__main__":
    main()
