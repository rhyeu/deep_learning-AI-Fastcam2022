import imghdr
import math
import os
import re
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def zero_pad(img,img_size):
    """
    Add black padding to the image.
    for each side of the image, each colour channel shall be padded with 0s of size
    (512 - image_width/height)/2 on each end, so that the image stays in the center,
    and is surrounded with black.
    """
    result = np.zeros((img_size, img_size, img.shape[2]))
    horz_start = int((img_size - img.shape[0]) / 2)
    horz_cord = range(horz_start, horz_start + img.shape[0])

    vert_start = int((img_size - img.shape[1]) / 2)
    vert_cord = range(vert_start, vert_start + img.shape[1])

    result[np.ix_(horz_cord, vert_cord, range(img.shape[2]))] = img.reshape(
            (img.shape[0], img.shape[1], img.shape[2])
        )
    return result


def load_image(img_path,img_size,is_grayscale=False):
    """
    Load a single image into a ndarray.
    """
    im = keras.preprocessing.image.load_img(img_path, grayscale=is_grayscale)
    im = keras.preprocessing.image.img_to_array(im)  # converts image to numpy array
    return zero_pad(im,img_size)

def save_img(img, filename):
    """
    Utility method that convert a ndarray into image and save to a image file.
    """
    img = keras.preprocessing.image.array_to_img(img)
    try:
        img.save(filename)
    except FileNotFoundError:
        util.create_dir(os.path.dirname(filename))
        img.save(filename)


def resize_img(img, size):
    """
    Given a list of images in ndarray, resize them into target size.
    """
    img = cv2.resize(img, (512, 512))
    if len(img.shape) == 2:
        img = img.reshape((size, size, 1))
    return img
