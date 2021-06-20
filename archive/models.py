# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Image handling libraries
import os
import PIL
import PIL.Image
from PIL import ImageGrab
import pathlib
from os import listdir
from os.path import isfile, join


#Text recognition functions:
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

#text = pytesseract.image_to_string(data)

monitor = {'top':0, 'left':0, 'width':1920, 'height':1080}
mon = (0, 0, 1920, 1080)
BATCHSIZE = 32

def softmax(Ht):
  output = []
  for j in Ht:
    numerator = np.exp(j)
    demoninator = 0
    for i in range(len(Ht)):
      demoninator += np.exp(Ht[i])
    output.append(numerator/demoninator)
  return output

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

checkpoint_path = ""
img_width = 1920
img_height = 1080
batch_size = 1

class_names = ['UP','DOWN', 'LEFT', 'RIGHT', 'TURNLEFT', 'TURNRIGHT', 'TURNUP', 'TURNDOWN']
num_classes = 8
data_dir = pathlib.Path("STASH")
# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)

loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy",
)

WalkerModel = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(180,320),
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='linear')
])

WalkerModel.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    loss=loss_fn,
    metrics=[tf.metrics.SparseCategoricalAccuracy()])





