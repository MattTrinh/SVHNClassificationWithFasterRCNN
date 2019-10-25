import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import h5py
from datetime import datetime

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
    training, testing = ds.load_data()

    categories = {str(i):i for i in range(10)}

    def schedule(epoch_index):
        return 0.1 * np.power(0.5, np.floor((1 + epoch_index) / 1.0))

    
    
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
    model.compile(
        optimizer
        #metrics=['accuracy', met.evaluate]
    )
    history = model.fit_generator(
        epochs=1,
        steps_per_epoch=50,
        callbacks = [
            keras.callbacks.LearningRateScheduler(schedule)
        ],
        generator=generator,
        validation_data=validation_data,
        validation_steps=50
    )

    
    
    runTime = datetime.now() - start

    print("Run completed in " + runTime.seconds + " seconds at " + datetime.now.strftime("%Y%m%d-%H%M%S"))

if __name__ == "__main__":
    main()