"""
General purposed utility methods shared by all models.

To use, simply import the file and start making calls.
"""
import imghdr
import math
import os
import re
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pick_bpart(df, bpart):
    """
    Create a sub dataset of particular body part.
    :param df: dataframe to process
    :param bpart: body part to extract
    :return: trimmed dataframe
    """
    if bpart == "all":
        return df
    return df[df["body_part"] == bpart].reset_index()


def pick_n_per_patient(df, num):
    """
    Create a sub dataset that pick first n images from each patient. Will return error
    if num is greater than the minial count
    :param df: dataframe to process
    :param num: number of images to pick from each patient. if set to 0, then pick all.
    :return: trimmed dataframe
    """
    if num == 0:
        return df
    min_count = df.groupby("study")["path"].count().min()

    if num > min_count:
        raise ValueError("num is greater than minimum count of images per patient: {}".format(
            min_count
        ))

    result = pd.DataFrame()
    for study in df["study"].unique():
        result = result.append(df[df["study"] == study][:num])

    return result.reset_index()


def zero_pad(img,img_size):
    """
    Add black padding to the image.
    for each side of the image, each colour channel shall be padded with 0s of size
    (512 - image_width/height)/2 on each end, so that the image stays in the center,
    and is surrounded with black.
    :param img: Image to process in nparray.
    :return: Processed image.
    """
    result = np.zeros((img_size, img_size, img.shape[2])) #center 넣고 나머지 zero padding
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
    Args:
        img_path:
            path to the image
        is_grayscale:
            if load the image to grayscale or RGB
    Returns: image as ndarray
    """
    im = keras.preprocessing.image.load_img(img_path, grayscale=is_grayscale)
    im = keras.preprocessing.image.img_to_array(im)     # converts image to numpy array
 #   plt.imshow(im[:, :, 0], cmap='gray')
    return zero_pad(im,img_size)


def plot_first_n_img(imgs, num=9):
    """
    Plot first n images from the given list.
    :param imgs: ndarry of images
    :param num: number of images to show
    :return:
    """
    n_row = int(math.sqrt(num))
    n_col = math.ceil(math.sqrt(num))
    plt.figure(1)
    plt.tight_layout()
    for i in range(num):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap='gray')


def save_img(img, filename):
    """
    Utility method that convert a ndarray into image and save to a image file.
    Args:
        img: image in ndarray
        filename: target filename including path
    Returns:
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
    Args:
        img: Input image in ndarray
        size: Target image size
    Returns: Resized images in ndarray
    """

   

    img = cv2.resize(img, (512, 512)) #512로 바꿈
    if len(img.shape) == 2:
        img = img.reshape((size, size, 1))
    return img
