import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
import cv2
import keras.backend as K
from keras import models, optimizers, losses, activations, callbacks
from keras.layers import *
from PIL import Image

# import tensorflow as tf


class Recognizer (object):

    def __init__(self):

        # tf.logging.set_verbosity(tf.logging.ERROR)

        self.__DIMEN = 96
        self.faceCascade = cv2.CascadeClassifier("C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\cascade\\frontalFace10\\haarcascade_frontalface_alt2.xml")


        input_shape = (self.__DIMEN**2,)
        convolution_shape = (self.__DIMEN, self.__DIMEN,1)
        kernel_size_1 = (4, 4)
        kernel_size_2 = (3, 3)
        pool_size_1 = (3, 3)
        pool_size_2 = (2, 2)
        strides = 1

        seq_conv_model = [
            # reshaping the input
            Reshape(input_shape=input_shape, target_shape=convolution_shape),
            # conv layers
            Conv2D(32, kernel_size=kernel_size_1, strides=strides,
                   activation=activations.relu),
            Conv2D(32, kernel_size=kernel_size_1, strides=strides,
                   activation=activations.relu),
            MaxPooling2D(pool_size=pool_size_1, strides=strides),
            Conv2D(64, kernel_size=kernel_size_2, strides=strides,
                   activation=activations.relu),
            Conv2D(64, kernel_size=kernel_size_2, strides=strides,
                   activation=activations.relu),
            MaxPooling2D(pool_size=pool_size_2, strides=strides),
            # flatten layer
            Flatten(),
            # dense layers
            Dense(64, activation=activations.sigmoid),
        ]

        seq_model = keras.Sequential(seq_conv_model)
        
        input_x1 = Input(shape=input_shape)
        input_x2 = Input(shape=input_shape)

        output_x1 = seq_model(input_x1)
        output_x2 = seq_model(input_x2)

        distance_euclid = Lambda(lambda tensors: K.abs(
            tensors[0] - tensors[1]))([output_x1, output_x2])
        outputs = Dense(1, activation=activations.sigmoid)(distance_euclid)
        self.__model = models.Model([input_x1, input_x2],outputs)

        self.__model.compile(loss=losses.binary_crossentropy,
                             optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])

    def fit(self, X, Y,  hyperparameters):
        initial_time = time.time()
        self.__model.fit(X, Y,
                         batch_size=hyperparameters['batch_size'],
                         epochs=hyperparameters['epochs'],
                         callbacks=hyperparameters['callbacks'],
                         validation_data=hyperparameters['val_data'],
                         verbose=1
                         )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary()
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(
            hyperparameters['epochs'], eta, time_unit))

    def prepare_images_from_dir(self, dir_path, padding=5, flatten=True):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        images = list()
        images_names = os.listdir(dir_path)
        for imageName in images_names:
            image = cv2.imread(str(dir_path + imageName))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                new_img = image[y+padding:y+h+padding, x+padding:x+w+padding]
                image = Image.fromarray(new_img).resize([self.__DIMEN, self.__DIMEN])
                image_data = np.array(image)
                grayscale_image = np.dot(image_data[..., :3], rgb_weights)
                image = grayscale_image/255
                image = np.array(np.reshape(image, (self.__DIMEN, self.__DIMEN)))
                
                images.append(image)

        if flatten:
            images = np.array(images)
            return images.reshape((images.shape[0], self.__DIMEN**2)).astype(np.float32)
        else:
            return np.array(images)

    def evaluate(self, test_X, test_Y):
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions

    def summary(self):
        self.__model.summary()

    def save_model(self, file_path):
        self.__model.save(file_path)

    def load_model(self, file_path):
        self.__model = models.load_model(file_path)
