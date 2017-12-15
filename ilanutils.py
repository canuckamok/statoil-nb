import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda, Conv2D, MaxPooling2D, BatchNormalization
from keras import applications
import os.path
import pandas as pd
import random
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2


def get_scaled_images(batch):
    img = []
    
    for i,rows in batch.iterrows():
        # Resize
        band_1 = np.array(row['band_1']).astype('float32').reshape(img_width, img_height)
        band_2 = np.array(row['band_2']).astype('float32').reshape(img_width, img_height)
        band_3 = band_1 + band_2
        
        # Rescale
        a = (band_1 - band_1.mean())/(band_1.max() - band_1.min())
        b = (band_2 - band_2.mean())/(band_2.max() - band_2.min())
        c = (band_3 - band_3.mean())/(band_3.max() - band_3.min())
        
        img.append(np.dstack((a,b,c)))
        
        return np.array(img)

def get_more_images(batch):
    more_images = []
    vert_images = []
    hor_images = []

    for i in range(0,batch.shape[0]):
        a = batch[i,:,:,0]
        b = batch[i,:,:,1]
        c = batch[i,:,:,2]

        av = cv2.flip(a,1)
        ah = cv2.flip(a,0)
        bv = cv2.flip(b,1)
        bh = cv2.flip(b,0)
        cv = cv2.flip(c,1)
        ch = cv2.flip(c,0)

        vert_images.append(nd.dstack((av,bv,cv)))
        hor_images.append(nd.dstack((ah,bh,ch)))

    v = np.array(vert_images)
    h = np.array(hor_images)

    more_images = np.concatenate((batch,v,h))

    return more_images

def one_hot(x):
	return to_categorical(x)

 
