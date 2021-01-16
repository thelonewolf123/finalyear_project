from recognizer import Recognizer
from PIL import Image
import numpy as np
import time
import os

recognizer = Recognizer()
recognizer.load_model('models/model.h5')
recognizer.summary()

custom_images = recognizer.prepare_images_from_dir('custom_images/')
hk_images = recognizer.prepare_images_from_dir('images/hk/')
mz_images = recognizer.prepare_images_from_dir('images/mz/')
# print("Test images")
# print(custom_images)
# print("Harish Kumar K")
# print(hk_images)
names = [
    "Harish Kumar K",
    "Mz"
]

test_images = os.listdir('custom_images/')

scores = list()
labels = list()
for image in custom_images:
    label = list()
    score = list()
    for sample in hk_images:
        image, sample = image.reshape((1, -1)), sample.reshape((1, -1))
        score.append(recognizer.predict([image, sample])[0])
        label.append(0)
    for sample in mz_images:
        image, sample = image.reshape((1, -1)), sample.reshape((1, -1))
        score.append(recognizer.predict([image, sample])[0])
        label.append(1)
    labels.append(label)
    scores.append(score)

scores = np.array(scores)
labels = np.array(labels)
print(scores)
print(labels)
for i in range(custom_images.shape[0]):
    index = np.argmax(scores[i])
    label_ = labels[i][index]
    print('IMAGE {} is {} with confidence of {}'.format(test_images[i], names[label_], scores[i][index][0]))