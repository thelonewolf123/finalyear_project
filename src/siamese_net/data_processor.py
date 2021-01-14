import numpy as np
import os
from PIL import Image

dimen = 48

dir_path = "./Preprocess"
out_path = "./processed_data"

rgb_weights = [0.2989, 0.5870, 0.1140]

sub_dir_list = os.listdir(dir_path)
images = list()
labels = list()
for i in range(len(sub_dir_list)):
    label = i
    image_names = os.listdir(os.path.join(dir_path, sub_dir_list[i]))
    for image_path in image_names:
        path = os.path.join(dir_path, sub_dir_list[i], image_path)
        try:
            image = Image.open(path)
            resize_image = image.resize((dimen, dimen))
            array_ = list()
            for x in range(dimen):
                sub_array = list()
                for y in range(dimen):
                    sub_array.append(resize_image.load()[x, y])
                array_.append(sub_array)
            image_data = np.array(array_)
            grayscale_image = np.dot(image_data[..., :3], rgb_weights)
            image = grayscale_image/255
            images.append(image)
            labels.append(label)
        except:
            print('WARNING : File {} could not be processed.'.format(path))

images = np.array(images)

samples_1 = list()
samples_2 = list()
labels = list()

for i in range(8):
    for j in range(20):
        samples_1.append(images[i])
        samples_2.append(images[j])
        if i < 3:
            if j < 3:
                labels.append(1)
            else:
                labels.append(0)
        else:
            if j > 2:
                labels.append(1)
            else:
                labels.append(0)

for image in images:
    pass

X1 = np.array(samples_1)
X2 = np.array(samples_2)
Y = np.array(labels)

np.save('{}/x1.npy'.format(out_path), X1)
np.save('{}/x2.npy'.format(out_path), X2)
np.save('{}/y.npy'.format(out_path), Y)

print('Done !!')
