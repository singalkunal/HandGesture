import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG_SIZE = 64

def preprocess_img(frame, IMG_SIZE):
    img_np = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    img_np = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img_np * (1./255)


def normalize(frame):
    h, w,_ = frame.shape
    if not h == w == 64:
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame

def draw_trajectory(trajec):
    plt.plot(trajec[0], trajec[1])
    plt.show()
