import random
import os
import pathlib
import numpy as np
import cv2

from PIL import Image

class MakePairs:
    def __init__(self, path, new_path='./processed_data', positive_ratio=0.5, dim=48, dataset_len=100):
        self.path = path
        self.new_path = new_path
        self.dataset_len = dataset_len
        self.dim = dim
        self.dir_list, self.keys = self.get_dir_list()
        self.rgb_weights = [0.2989, 0.5870, 0.1140]
        self.positive_ratio = positive_ratio

        self.sample_1 = []
        self.sample_2 = []
        self.result = []

        self.faceCascade = cv2.CascadeClassifier("C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\cascade\\frontalFace10\\haarcascade_frontalface_alt2.xml")


    def get_dir_list(self):
        path = pathlib.Path(self.path)
        dir_list = {}
        keys = os.listdir(path)

        for subdir in keys:
            dir_list[subdir] = os.listdir(path/subdir)

        return dir_list, keys

    def get_image_array(self,obj_index = None,padding=5):

        if obj_index == None:
            obj_index = random.randint(0, len(self.keys)-1)
            
        obj_class = self.keys[obj_index]

        index = random.randint(0, len(self.dir_list[obj_class])-1)

        path = pathlib.Path(self.path) / self.keys[obj_index]/self.dir_list[obj_class][index]

        image = cv2.imread(str(path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            new_img = image[y+padding:y+h+padding, x+padding:x+w+padding]
            image = Image.fromarray(new_img).resize([self.dim, self.dim])
            img_array = np.array(image)

            grayscale_image = np.dot(img_array[..., :3], self.rgb_weights)
            image = grayscale_image/255
            # print(f'({len(self.result)+1} /{self.dataset_len}) {path} done!!')

            return image, obj_index

        return None,None

    def make_pairs(self):

        positive = 0
        negative = 0
        i = 0
        image2, obj_class2 = None, None

        while(self.dataset_len > i):

            image1, obj_class1 = self.get_image_array()
            
            if negative >= (self.dataset_len*self.positive_ratio)//1:
                image2, obj_class2 = self.get_image_array(obj_class1)
            else:
                image2, obj_class2 = self.get_image_array()

            if obj_class1 != None and  obj_class2 != None:

                self.sample_1.append(image1)
                self.sample_2.append(image2)

                print(f'({len(self.result)+1} /{self.dataset_len}) {self.keys[obj_class1]} done!!')
                print(f'({len(self.result)+1} /{self.dataset_len}) {self.keys[obj_class2]} done!!')

                if obj_class1 == obj_class2:
                    self.result.append(1)
                    positive += 1
                    i += 1
                else:
                    self.result.append(0)
                    negative += 1
                    i += 1

    def save_data(self):
        if not (len(self.sample_1) and len(self.sample_2) and len(self.result)):
            self.make_pairs()

        if not os.path.exists(self.new_path):
            os.mkdir(self.new_path)

        np.save('{}/x1.npy'.format(self.new_path), np.asarray(self.sample_1,dtype=np.float32))
        # print(self.sample_1)
        np.save('{}/x2.npy'.format(self.new_path), np.asarray(self.sample_2,dtype=np.float32))
        np.save('{}/y.npy'.format(self.new_path), np.asanyarray(self.result,dtype=np.uint8))



if __name__ == "__main__":
    dataset = "C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\siamese_net\\face_dt"
    location = "C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\siamese_net\\processed_data"
    make_pairs = MakePairs(path=dataset, new_path=location dim=96,dataset_len=1500,positive_ratio=0.58)
    make_pairs.save_data()

    positive = 0
    negative = 0

    for data in make_pairs.result:
        if data == 1:
            positive += 1
        else:
            negative += 1

    print(f'Positive: {positive}, Negative: {negative}')
    # print(res)
