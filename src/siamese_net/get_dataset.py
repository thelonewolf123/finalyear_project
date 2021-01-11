import face_recognition
import os
import json

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
from fastbook import *
 

actors = ['vijay', 'ajith', 'kamal', 'rajini', 'modi',
          'nayanthara', 'anushka', 'mia khalifa', 'sunny leone']

if not os.path.exists(f'./face_dt'):
    os.mkdir(f'./face_dt')

for name in actors:
    if not os.path.exists(f'./face_dt/{name.replace(" ","_")}'):
        os.mkdir(f'./face_dt/{name.replace(" ","_")}')
    urls = search_images_ddg(name, max_images=10)
    for url, index in zip(urls[0:10], range(0, 10)):
        download_url(url, f'face_dt/{name.replace(" ","_")}/{name}{index}.jpg')
        print(f'done {name}{index}.jpg')


def preprocess_photos(path_str, new_path_str):

    X_data, y_data = [], []
    config = {}
    path = Path(path_str)
    new_path = Path(new_path_str)

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    sub_dir = os.listdir(path)
    n = len(sub_dir)

    for cat, index in zip(sub_dir, range(0, n)):
        config[cat] = index
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
    
    with open(f'{new_path}/mapping.conf','w') as fileobj:
        fileobj.write(json.dump(config))

preprocess_photos(Path('./face_dt/'), Path('./Preprocess'))


print('Done !!')
