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

