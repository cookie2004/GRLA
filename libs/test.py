from vgbot_lib import *
import pymem
import pymem.process

def F76_preprocess(image):
    return np.average(np.array(image), axis=2)

def movement(key_index):
    keys = [ctypes_functions.KEY_W,
        ctypes_functions.KEY_S,
        ctypes_functions.KEY_A,
        ctypes_functions.KEY_D]
    ctypes_functions.SendInput(ctypes_functions.Keyboard(keys[key_index]))
    time.sleep(0.05)
    ctypes_functions.SendInput(ctypes_functions.Keyboard(keys[key_index]))
    time.sleep(0.05)

def push_button(key_index):
    keys = [0x57,
        0x53,
        0x41,
        0x44]
    win32api.keybd_event(keys[key_index], 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(keys[key_index], 0, win32con.KEYEVENTF_KEYUP, 0)

