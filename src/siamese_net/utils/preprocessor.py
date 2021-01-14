import os
import face_recognition
import numpy as np

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt


rgb_weights = [0.2989, 0.5870, 0.1140]
dim_size = 48

def preprocess_photos(path_str, new_path_str):

    X_data, y_data = [], []

    path = Path(path_str)
    new_path = Path(new_path_str)

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    sub_dir = os.listdir(path)
    n = len(sub_dir)

    for cat, index in zip(sub_dir, range(0, n)):
        for img_path in os.listdir(path/cat):
            image = face_recognition.load_image_file(path/cat/img_path)
            face_loc = face_recognition.face_locations(image)
            if len(face_loc) == 1:
                top, right, bottom, left = face_loc[0]
                new_img = image[top:bottom, left:right]
                grayscale_image = np.dot(new_img[..., :3], rgb_weights)
                gimage = Image.fromarray(grayscale_image).resize([64, 64])
                data = np.array(gimage)/255
                X_data.append(data)
                y_data.append(index)

    np.save(f'{new_path}/X_data.npy',np.array(X_data))
    np.save(f'{new_path}/y_data.npy',np.array(X_data))


preprocess_photos('./images', './processed_data')


def pre_process_image(path_str):

    image = face_recognition.load_image_file(path_str)
    face_loc = face_recognition.face_locations(image)
    if len(face_loc) == 1:
        top, right, bottom, left = face_loc[0]
        new_img = image[top:bottom, left:right]
        grayscale_image = np.dot(new_img[..., :3], rgb_weights)
        gimage = Image.fromarray(grayscale_image).resize([dim_size,dim_size])
        data = np.array(gimage)/255
        print(data)


pre_process_image(
    '/home/lonewolf/projects/hobby/machine_learning/siamese_network_tf/custom_images/IMG_20200604_112705.jpg')
