from recognizer import Recognizer
from PIL import Image
import numpy as np
import time
import os

data_dimension = 48
TRAIN = False

x = input('You want to train the model? [y/n] ')


if x == 'y':
    TRAIN = True

if TRAIN:
    X1 = np.load('processed_data/x1.npy')
    X2 = np.load('processed_data/x2.npy')
    Y = np.load('processed_data/y.npy')

    X1 = X1.reshape((X1.shape[0], data_dimension**2)).astype(np.float32)
    X2 = X2.reshape((X2.shape[0], data_dimension**2)).astype(np.float32)

    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

recognizer = Recognizer()
if not TRAIN:
    recognizer.load_model('models/model.h5')

parameters = {
    'batch_size': 256,
    'epochs': 4,
    # [ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
    'callbacks': None,
    'val_data': None
}
if TRAIN:
    recognizer.fit([X1, X2], Y, hyperparameters=parameters)
    recognizer.save_model('models/model.h5')

custom_images = recognizer.prepare_images_from_dir('custom_images/')
hk_images = recognizer.prepare_images_from_dir('images/hk/')
mz_images = recognizer.prepare_images_from_dir('images/mz/')

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
    print('IMAGE {} is {} with confidence of {}'.format(
        test_images[i], names[label_], scores[i][index][0]))
