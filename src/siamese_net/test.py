from recognizer import Recognizer
from PIL import Image
import numpy as np
import time
import os

recognizer = Recognizer()
# recognizer.load_model('models/model.h5')
recognizer.summary()

custom_images = recognizer.prepare_images_from_dir('custom_images/')
hk_images = recognizer.prepare_images_from_dir('images/hk/')
mz_images = recognizer.prepare_images_from_dir('images/mz/')
custom_dir = os.listdir('custom_images/')

print(custom_dir)

data = list()
data.append(custom_images[0].reshape((1, -1)))
data.append(mz_images[0].reshape((1, -1)))
result = recognizer.predict(data)

print(result[0])