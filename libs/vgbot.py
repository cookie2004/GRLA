import numpy as np
import cv2
from PIL import ImageGrab #Use ImageGrab in Pillow for Windows or MacOS environment.
import win32api
import win32gui
import win32con
import matplotlib.pylab as plt
#import pyscreenshot as ImageGrab #Use pyscreenshot in Linux.
from win32api import GetSystemMetrics
import time

#Input libraries
import ctypes_functions

#ML libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##########COMPUTER VISION ROUTINES
WIDTH = GetSystemMetrics(0)
HEIGHT = GetSystemMetrics(1)
W_ACTUAL = 1920
H_ACTUAL = 1080

def screenCapture(windowName):
    
    cv2.imshow(windowName)

def _windowEnumerationHandler(hwnd, resultList):
    '''Pass to win32gui.EnumWindows() to generate list of window handle,
    window text, window class tuples.'''
    resultList.append((hwnd,
                       win32gui.GetWindowText(hwnd),
                       win32gui.GetClassName(hwnd)))
def findWindowHandle(WindowTitle):
    output = []
    topWindows = []
    win32gui.EnumWindows(_windowEnumerationHandler, topWindows)
    for i in topWindows:
        if WindowTitle in i[1]: output.append(i)
        else: None
    return output

def windowCoordinates(windowHandle):
    #win32gui.SetWindowPos(windowHandle, win32con.HWND_TOPMOST, 0,0,0,0,
    #win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    x, y, x1, y1 = win32gui.GetClientRect(windowHandle)
    return  np.array([win32gui.ClientToScreen(windowHandle, (x,y)), win32gui.ClientToScreen(windowHandle, (x1, y1))]).flatten() #win32gui.GetWindowRect(windowHandle) #win32gui.GetWindowPlacement(windowHandle)

def screenGrab(coord):
    return np.array(ImageGrab.grab(tuple(coord)))

def preprocess(image):
    output = np.average(np.array(image), axis=2)[:,123:640]
    return cv2.resize(output, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

##########INPUT CONTROL
VK_CODE = {'backspace':0x08,
           'tab':0x09,
           'clear':0x0C,
           'enter':0x0D,
           'shift':0x10,
           'ctrl':0x11,
           'alt':0x12,
           'pause':0x13,
           'caps_lock':0x14,
           'esc':0x1B,
           'spacebar':0x20,
           'page_up':0x21,
           'page_down':0x22,
           'end':0x23,
           'home':0x24,
           'left_arrow':0x25,
           'up_arrow':0x26,
           'right_arrow':0x27,
           'down_arrow':0x28,
           'select':0x29,
           'print':0x2A,
           'execute':0x2B,
           'print_screen':0x2C,
           'ins':0x2D,
           'del':0x2E,
           'help':0x2F,
           '0':0x30,
           '1':0x31,
           '2':0x32,
           '3':0x33,
           '4':0x34,
           '5':0x35,
           '6':0x36,
           '7':0x37,
           '8':0x38,
           '9':0x39,
           'a':0x41,
           'b':0x42,
           'c':0x43,
           'd':0x44,
           'e':0x45,
           'f':0x46,
           'g':0x47,
           'h':0x48,
           'i':0x49,
           'j':0x4A,
           'k':0x4B,
           'l':0x4C,
           'm':0x4D,
           'n':0x4E,
           'o':0x4F,
           'p':0x50,
           'q':0x51,
           'r':0x52,
           's':0x53,
           't':0x54,
           'u':0x55,
           'v':0x56,
           'w':0x57,
           'x':0x58,
           'y':0x59,
           'z':0x5A,
           'numpad_0':0x60,
           'numpad_1':0x61,
           'numpad_2':0x62,
           'numpad_3':0x63,
           'numpad_4':0x64,
           'numpad_5':0x65,
           'numpad_6':0x66,
           'numpad_7':0x67,
           'numpad_8':0x68,
           'numpad_9':0x69,
           'multiply_key':0x6A,
           'add_key':0x6B,
           'separator_key':0x6C,
           'subtract_key':0x6D,
           'decimal_key':0x6E,
           'divide_key':0x6F,
           'F1':0x70,
           'F2':0x71,
           'F3':0x72,
           'F4':0x73,
           'F5':0x74,
           'F6':0x75,
           'F7':0x76,
           'F8':0x77,
           'F9':0x78,
           'F10':0x79,
           'F11':0x7A,
           'F12':0x7B,
           'F13':0x7C,
           'F14':0x7D,
           'F15':0x7E,
           'F16':0x7F,
           'F17':0x80,
           'F18':0x81,
           'F19':0x82,
           'F20':0x83,
           'F21':0x84,
           'F22':0x85,
           'F23':0x86,
           'F24':0x87,
           'num_lock':0x90,
           'scroll_lock':0x91,
           'left_shift':0xA0,
           'right_shift ':0xA1,
           'left_control':0xA2,
           'right_control':0xA3,
           'left_menu':0xA4,
           'right_menu':0xA5,
           'browser_back':0xA6,
           'browser_forward':0xA7,
           'browser_refresh':0xA8,
           'browser_stop':0xA9,
           'browser_search':0xAA,
           'browser_favorites':0xAB,
           'browser_start_and_home':0xAC,
           'volume_mute':0xAD,
           'volume_Down':0xAE,
           'volume_up':0xAF,
           'next_track':0xB0,
           'previous_track':0xB1,
           'stop_media':0xB2,
           'play/pause_media':0xB3,
           'start_mail':0xB4,
           'select_media':0xB5,
           'start_application_1':0xB6,
           'start_application_2':0xB7,
           'attn_key':0xF6,
           'crsel_key':0xF7,
           'exsel_key':0xF8,
           'play_key':0xFA,
           'zoom_key':0xFB,
           'clear_key':0xFE,
           '+':0xBB,
           ',':0xBC,
           '-':0xBD,
           '.':0xBE,
           '/':0xBF,
           '`':0xC0,
           ';':0xBA,
           '[':0xDB,
           '\\':0xDC,
           ']':0xDD,
           "'":0xDE,
           '`':0xC0}

def press(*args):
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.05)
        win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)

def pressAndHold(*args):
    '''
    press and hold. Do NOT release.
    accepts as many arguments as you want.
    e.g. pressAndHold('left_arrow', 'a','b').
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.05)
           
def pressHoldRelease(*args):
    '''
    press and hold passed in strings. Once held, release
    accepts as many arguments as you want.
    e.g. pressAndHold('left_arrow', 'a','b').

    this is useful for issuing shortcut command or shift commands.
    e.g. pressHoldRelease('ctrl', 'alt', 'del'), pressHoldRelease('shift','a')
    '''
    for i in args:
        win32api.keybd_event(VK_CODE[i], 0,0,0)
        time.sleep(.05)
            
    for i in args:
            win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)
            time.sleep(.1)
            
        

def release(*args):
    '''
    release depressed keys
    accepts as many arguments as you want.
    e.g. release('left_arrow', 'a','b').
    '''
    for i in args:
           win32api.keybd_event(VK_CODE[i],0 ,win32con.KEYEVENTF_KEYUP ,0)
            
    #NES controller configuration

def action(key_index):
    keys = [ctypes_functions.KEY_A,
            ctypes_functions.KEY_D,
            ctypes_functions.KEY_V,
            ctypes_functions.KEY_B]
    #win32api.keybd_event(ctypes_functions.VK_OEM_5, 0, 0, 0)
    for key in key_index:
        win32api.keybd_event(keys[key],0 ,0 ,0)
    time.sleep(0.01)
    #win32api.keybd_event(ctypes_functions.VK_OEM_5, 0, win32con.KEYEVENTF_KEYUP, 0)
    for key in key_index:
        win32api.keybd_event(keys[key],0 ,win32con.KEYEVENTF_KEYUP ,0)

def trade_off(action):
    keys = [[ctypes_functions.KEY_A, 0],
            [ctypes_functions.KEY_D, 0],
            [ctypes_functions.KEY_V, 0],
            [ctypes_functions.KEY_B, 0],
            [ctypes_functions.KEY_A, win32con.KEYEVENTF_KEYUP],
            [ctypes_functions.KEY_D, win32con.KEYEVENTF_KEYUP],
            [ctypes_functions.KEY_V, win32con.KEYEVENTF_KEYUP],
            [ctypes_functions.KEY_B, win32con.KEYEVENTF_KEYUP]]
    win32api.keybd_event(keys[action][0],0 ,keys[action][1] ,0)

def reset_input():
    keys =  [[ctypes_functions.KEY_A, win32con.KEYEVENTF_KEYUP],
            [ctypes_functions.KEY_D, win32con.KEYEVENTF_KEYUP],
            [ctypes_functions.KEY_V, win32con.KEYEVENTF_KEYUP],
            [ctypes_functions.KEY_B, win32con.KEYEVENTF_KEYUP]]
    for i in keys:
        win32api.keybd_event(i[0],0 ,i[1] ,0)   
    
def load_save_state():
    win32api.keybd_event(ctypes_functions.KEY_P,0 ,0 ,0)
    time.sleep(0.01)
    win32api.keybd_event(ctypes_functions.KEY_P,0 ,win32con.KEYEVENTF_KEYUP ,0)
    
def load_random_state():
    states = [ctypes_functions.KEY_1,
              ctypes_functions.KEY_2,
              ctypes_functions.KEY_3,
              ctypes_functions.KEY_4,
              ctypes_functions.KEY_5,
              ctypes_functions.KEY_6,
              ctypes_functions.KEY_7,
              ctypes_functions.KEY_8,
              ctypes_functions.KEY_9]
    chosen_state = np.random.choice(states)
    win32api.keybd_event(chosen_state,0 ,0 ,0)
    win32api.keybd_event(ctypes_functions.KEY_P,0 ,0 ,0)
    win32api.keybd_event(chosen_state,0 ,win32con.KEYEVENTF_KEYUP ,0)
    win32api.keybd_event(ctypes_functions.KEY_P,0 ,win32con.KEYEVENTF_KEYUP ,0)
    
#def action(key_index):  #Backup of full controller.
#keys = [ctypes_functions.KEY_W,
#        ctypes_functions.KEY_S,
#        ctypes_functions.KEY_A,
#        ctypes_functions.KEY_D,
#        ctypes_functions.KEY_V,
#        ctypes_functions.KEY_B]
#win32api.keybd_event(ctypes_functions.VK_OEM_5, 0, 0, 0)
#win32api.keybd_event(keys[key_index],0 ,0 ,0)
#time.sleep(0.05)
#win32api.keybd_event(ctypes_functions.VK_OEM_5, 0, win32con.KEYEVENTF_KEYUP, 0)
#win32api.keybd_event(keys[key_index],0 ,win32con.KEYEVENTF_KEYUP ,0)
    
##########MEMORY ACCESS ROUTINES



##########AI ROUTINES

class ANN():
  def __init__(self):
    num_actions = 8
    #NN layers
    image_input = layers.Input(shape=(4,100,100))
    vector_input = layers.Input(shape=(4,))
    preprocessor = layers.experimental.preprocessing.Resizing(100, 100, interpolation='bilinear', name=None)(image_input)
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(preprocessor)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    concat_layers = layers.Concatenate()([vector_input, layer4])
    layer5 = layers.Dense(512, activation="relu")(concat_layers)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    #Define NN parameters.
    self.toymodel = keras.Model(inputs=[image_input, vector_input], outputs=action)
    self.loss_fn = tf.keras.losses.Huber()
    self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    self.toymodel.compile(self.optimizer, self.loss_fn)

  def trainStep(self, sample_X, sample_Y):
    with tf.GradientTape() as tape:
      old_q = self.toymodel(sample_X, training=True)
      loss_value = self.loss_fn(sample_Y, old_q)
    grads = tape.gradient(loss_value, self.toymodel.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.toymodel.trainable_weights))
    return loss_value.numpy()

  def train(self, x_input, y_input, batchsize=64):
    loss_history = []
    dataset = tf.data.Dataset.from_tensor_slices((x_input, y_input))
    dataset = dataset.shuffle(buffer_size=1024).batch(batchsize)
    for steps, (x, y) in enumerate(dataset):
      loss_history.append(self.trainStep(x,y))
    return loss_history

  def forward(self, x_input):
    return self.toymodel(x_input)
