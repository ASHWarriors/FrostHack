
#Importing required libraries 
import numpy as np
import cv2
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

#Face Detection .xml file
face_detect = cv2.CascadeClassifier('C:/Users/K MANGOTRA/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#Mask Detection Model
mod = load_model('model-best')

np.set_printoptions(suppress=True)

# Type of Mask Detection Model
model = tensorflow.keras.models.load_model('keras_model.h5',compile=False)

source = cv2.VideoCapture(0)

while 1:
    not_to_use, image = source.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(224,244))
    faces = face_detect.detectMultiScale(gray, 1.5, 1)
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = cv2.resize(image, (224, 224))
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+w, x:x+h]
        resized_image = cv2.resize(face_roi, (100, 100))
        normalized_image = resized_image/255
        reshaped_face = np.reshape(normalized_image, (1, 100, 100, 1))
        result = mod.predict(reshaped_face)[0]
        if result[0] >  result[1]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image,". ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if(prediction[0][0]>0.75):
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "N95 Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif(prediction[0][1]>0.75):
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "Not N95", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif(prediction[0][2]>0.75):
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "Incorrectly Wearing!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(image,"Please Reposition yourself", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        elif result[1] >  result[0]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Not Safe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    
        
    cv2.imshow('Catch95', image)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
source.release()