import face_recognition
import os
import json

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
from fastbook import *
 

actors = ['vijay', 'ajith', 'kamal', 'rajini', 'modi',
          'nayanthara', 'anushka', 'mia khalifa', 'sunny leone']

location = "C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\siamese_net\\face_dt"

if not os.path.exists(f'{location}'):
    os.mkdir(f'{location}')

for name in actors:
    if not os.path.exists(f'{location}/{name.replace(" ","_")}'):
        os.mkdir(f'{location}/{name.replace(" ","_")}')
    urls = search_images_ddg(name, max_images=10)
    for url, index in zip(urls[0:10], range(0, 10)):
        download_url(url, f'{location}/{name.replace(" ","_")}/{name}{index}.jpg')
        print(f'done {name}{index}.jpg')

print('Done !!')
