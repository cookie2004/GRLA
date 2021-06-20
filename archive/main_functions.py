# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

#Custom libs
import ctypes_functions as ctypes_functions
import models
#import cv_lib

model = keras.models.load_model('201201.h5')
test = np.expand_dims(models.screenGrab(),0)
print (model(test))
print (models.softmax(model(test)))
print (tf.keras.activations.softmax(model(test)))

#class_names = ['UP','DOWN', 'LEFT', 'RIGHT', 'TURNLEFT', 'TURNRIGHT', 'TURNUP', 'TURNDOWN']

def act_on(key_index):
    keys = [ctypes_functions.KEY_W,
            ctypes_functions.KEY_S,
            ctypes_functions.KEY_A,
            ctypes_functions.KEY_D,
            ctypes_functions.VK_NUMPAD4,
            ctypes_functions.VK_NUMPAD6,
            ctypes_functions.VK_NUMPAD8,
            ctypes_functions.VK_NUMPAD2]  
    ctypes_functions.SendInput(ctypes_functions.Keyboard(keys[key_index]))


while True:
    test = np.expand_dims(models.screenGrab(),0)
    test = np.argmax(tf.keras.activations.softmax(model(test)))
    act_on(test)
