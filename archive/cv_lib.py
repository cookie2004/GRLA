import ctypes_functions as ctypes_functions
import models

import keyboard
import cv2
import time
import numpy as np
from mss import mss
import os
import tensorflow as tf
import matplotlib.pylab as plt
from PIL import ImageGrab

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

monitor = {'top':0, 'left':0, 'width':1920, 'height':1080}
mon = (0, 0, 1920, 1080)
BATCHSIZE = 32

def screenGrab():
    #with mss() as sct:
    #    img  = sct.grab(monitor)
    #    plt.imshow(img)
    #    plt.show()  
    #return np.array(img)[...,:3]
    img = np.asarray(ImageGrab.grab(bbox=mon))
    plt.imshow(np.array(img)[...,:3])
    return np.array(img)[...,:3]

    
def stats():
    #Health bar monitor. Health bar starts at 122x1010. It encompasses pixels 121 to 426

    #Grab screen.
    data = screenGrab()
    left = 119
    right = 425
    vertical = 1009
    healthbar = 0
    radbar = 0
    apleft = 1495
    apright = 1799
    APbar = 0
    for i in data[vertical][left:right]:
        if np.sum(i) == 713:
            healthbar += 1
        elif np.sum(i) == 449:
            radbar += 1
        else:
            None
    for i in data[vertical][apleft:apright]:
        if np.sum(i) == 713:
            APbar += 1
        else:
            None
    healthbar = healthbar/(right-left)
    radbar = radbar/(right-left)
    APbar = APbar/(apright-apleft)
    print ('Health at ' + str((round(healthbar,2)*100)) + '%')
    print ('Rads at ' + str((round(radbar,2)*100)) + '%')
    print ('AP at ' + str((round(APbar,2)*100)) + '%')

def Fetchkeys():
#class_names = ['UP','DOWN', 'LEFT', 'RIGHT', 'TURNLEFT', 'TURNRIGHT', 'TURNUP', 'TURNDOWN']
    keyspressed = np.zeros(8)
    img = screenGrab()
    if keyboard.is_pressed('2'):
        ctypes_functions.direction(0,25)
        keyspressed[2] = 1
        
    elif keyboard.is_pressed('4'):
        ctypes_functions.direction(-25,0)
        keyspressed[4] = 1
        
    elif keyboard.is_pressed('6'):
        ctypes_functions.direction(25,0)
        keyspressed[6] = 1
        
    elif keyboard.is_pressed('8'):
        ctypes_functions.direction(0,-25)
        keyspressed[7] = 1
        
    elif keyboard.is_pressed('w'):
        keyspressed[0] = 1
        
    elif keyboard.is_pressed('s'):
        keyspressed[1] = 1
        
    elif keyboard.is_pressed('a'):
        keyspressed[2] = 1
        
    elif keyboard.is_pressed('d'):
        keyspressed[3] = 1
        
    else:
        None   
    time.sleep(0.1)
    resized = cv2.resize(img, dsize=(320,180), interpolation=cv2.INTER_CUBIC)
    return np.array([resized]), np.array([keyspressed])     

while True:
    workingBatch, keys = Fetchkeys()
    for i in range(320):
        sampleimg, samplekeys = Fetchkeys()
        workingBatch = np.append(workingBatch, sampleimg, axis=0)
        keys = np.append(keys, samplekeys, axis=0)
        if i%10 == 0: print ('Step :', i)
    #workingBatch = np.delete(workingBatch, 0, axis=0)
    #keys = np.delete(keys, 0, axis=0)

    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0,
        reduction="auto",
        name="categorical_crossentropy",
    )

    epochs = 2
    for epoch in range(epochs):
        print ('Start of epoch %d' % (epoch,))
        with tf.GradientTape() as tape:
            logits = models.WalkerModel(workingBatch, training=True)
            loss_value = loss_fn(keys, logits)
        grads = tape.gradient(loss_value, models.WalkerModel.trainable_weights)
        optimizer.apply_gradients(zip(grads, models.WalkerModel.trainable_weights))
    models.WalkerModel.save('201201.h5')
    print ("Training loss: ", float(loss_value))

##while True:
##    time.sleep(5)
##    ctypes.SendInput(ctypes.Keyboard(ctypes.KEY_W))
##    time.sleep(10)
##    ctypes.SendInput(ctypes.Keyboard(ctypes.KEY_W,ctypes.KEYEVENTF_KEYUP))
##    time.sleep(1)
##    #ctypes.fire()
