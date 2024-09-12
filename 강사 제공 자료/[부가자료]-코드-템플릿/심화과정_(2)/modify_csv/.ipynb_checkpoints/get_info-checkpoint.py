import imghdr
import math
import os
import re
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BPARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]
def classify_bpart(data):
    """
    Divide TRAIN_LABELED into sub-sets based on the body parts in the image.
    Also add body part as a new feature of the dataset.
    :param data: dataset to process.
    :return:
    """
    for bpart in BPARTS:
        data.loc[data["path"].str.contains(bpart.upper()), "body_part"] = bpart


def complete_path(data_dir,data, column):
    """
    Convert relative image path to absolute path so that the execution does not depend
    on working directory. Also clean up the patient name
    :param data: dataset to process.
    :param column: column to perform the operation.
    :return:
    """
    data[column] = np.where(
        data[column].str.startswith("MURA-v1.1"),
        data[column].str.replace("MURA-v1.1", data_dir),
        data[column]
    )
    
def get_patient(row):
    """
    Call back function to check if the image column is a valid path,
    and grab the parent directory if it is.
    :param row: a row from processing table
    :return:
    """
    try:
        img_type = imghdr.what(row["path"])
    except IsADirectoryError:
        img_type = None

    if img_type:
        return os.path.dirname(row["path"]) + "/"
    return row["patient"]