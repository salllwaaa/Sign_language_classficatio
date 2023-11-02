#Common libraries
import pandas as pd
import numpy as np
# import seaborn as sns
import random
from keras.models import load_model

#Data Preprocessing

# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split


#For Modle
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from keras import backend as K
import warnings
import cv2
# from sklearn.metrics import accuracy_score


#Visualization
import matplotlib.pyplot as plt
from keras.utils import plot_model
# from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg





def getLetter(result):
    classLabels={0:'A',
                 1:'B',
                 2:'C',
                 3:'D',
                 4:'E',
                 5:'F',
                 6:'G',
                 7:'H',
                 8:'I',
                 9:'K',
                 10:'L',
                 11:'M',
                 12:'N',
                 13:'O',
                 14:'P',
                 15:'Q',
                 16:'R',
                 17:'S',
                 18:'T',
                 19:'U',
                 20:'V',
                 21:'W',
                 22:'X',
                 23:'Y'}
    try:
        res=int(result)
        return classLabels[res]
    except:
        return "ERROR"
    






model = load_model("D:/taks4_INTERN/Sign_language_Classfication.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

    roi = roi.reshape(1, 28, 28, 1)

    result = model.predict(roi, verbose=0)
    predicted_class_index = np.argmax(result[0])
    predicted_class_name = getLetter(predicted_class_index)
    
    cv2.putText(copy, predicted_class_name, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


# result=str(model.predict_classes(roi,1,verbose=0)[0])