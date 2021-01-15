# import face_recognition
import cv2
import random
import threading
import requests
import numpy as np

from recognizer import Recognizer
from PIL import Image

class Gods_eye:

    def __init__(self,dim=128):
        
        self.dim = dim
        self.count = 0
        self.rgb_weights = [0.2989, 0.5870, 0.1140]

        self.webcam = cv2.VideoCapture(0)
        
        self.known_people = []
        self.known_face_encodings = []
        
        self.faceCascade = cv2.CascadeClassifier("C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\cascade\\frontalFace10\\haarcascade_frontalface_alt2.xml")
        
        self.recognizer = Recognizer()
        self.recognizer.load_model('C:\\Users\\hk471\\Documents\\Machine_learning\\finalyear_project\\src\\siamese_net\\models\\model.h5')


    def run(self):
        self.init_setup()
        self.capture_frames()

    def init_setup(self):
        image = Image.open("people/face0.jpg").resize(([self.dim, self.dim]))
        image_array = np.asarray(image)

        grayscale_image = np.dot(image_array[..., :3], self.rgb_weights)
        image = grayscale_image/255

        self.known_face_encodings.append(image)
        self.known_people.append("Harish Kumar")

    def check_person(self, person):
        score = list()

        image = Image.fromarray(person).resize([self.dim, self.dim])
        img_array = np.array(image)
        grayscale_image = np.dot(img_array[..., :3], self.rgb_weights)
        image = grayscale_image/255
        
        for sample in self.known_face_encodings:
            image, sample = image.reshape((1, -1)), sample.reshape((1, -1))
            score.append(self.recognizer.predict([image, sample])[0])

        scores = np.array(score)
        person_index = np.argmin(scores)

        if scores[person_index] > 0.5:
            print(f"Found {self.known_people[person_index]}, with confidence {scores[person_index]}")


    def extract_faces(self, frame):
        padding = 5
        people = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        # print("Found {0} faces!".format(len(faces)))

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w] #slice the face from the image
            # cv2.imwrite(f'face{self.count}.jpg', face) #save the image
            # count+=1
            self.check_person(face)

        return faces



    def capture_frames(self):
        padding = 5
        while True:
            ret, frame = self.webcam.read()
            faces = self.extract_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x+padding, y+padding), (x+w+padding, y+h+padding), (0, 255, 0), 2)

            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def init_connection(self,server):
        pass

    def send_request(self, person):
        pass

if __name__ == "__main__":
    gods_eye = Gods_eye()
    gods_eye.run()
