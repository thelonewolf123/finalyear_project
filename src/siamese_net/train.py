from recognizer import Recognizer
from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np
import time
import os
import pandas as pd 

data_dimension = 128
TRAIN = True

def extract_data(data):
    sample1 = [ x[0] for x in data.values.tolist() ]
    sample2 = [ x[1] for x in data.values.tolist() ]
    print(len(sample2),len(sample2[0]))
    return np.asarray(sample1),np.asarray(sample2)

x = int(input('Enter the epochs value '))



X1 = np.load('processed_data/x1.npy',allow_pickle=True)
X2 = np.load('processed_data/x2.npy',allow_pickle=True)
Y = np.load('processed_data/y.npy',allow_pickle=True)
print(X1.shape)

X1 = X1.reshape((X1.shape[0], data_dimension**2)).astype(np.float32)
X2 = X2.reshape((X2.shape[0], data_dimension**2)).astype(np.float32)

dataset = df = pd.DataFrame(list(zip(X1,X2)), columns =['Image 1', 'Image 2'])


print(X1.shape)
print(X2.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(dataset, Y, test_size=0.33, random_state=42)

X1,X2 = extract_data(X_train)
x1,x2 = extract_data(X_test)

recognizer = Recognizer()
recognizer.load_model('models/model.h5')

parameters = {
    'batch_size': 8,
    'epochs': x,
    # [ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
    'callbacks': None,
    'val_data': [[x1,x2],y_test]
}
recognizer.fit([X1,X2], y_train, hyperparameters=parameters)
recognizer.save_model('models/model.h5')