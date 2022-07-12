import imghdr
import math
import os
import re
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "./MURA-v1.1" 

BPARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


def load_dataframe():
    """
     Import csv files into Dataframes.
    """
    train_labeled = pd.read_csv(
        os.path.join(DATA_DIR, "train_labeled_studies.csv"),
        names=["patient", "label"]
    )

    valid_labeled = pd.read_csv(
        os.path.join(DATA_DIR, "valid_labeled_studies.csv"),
        names=["patient", "label"]
    )

    # import image paths
    train_path = pd.read_csv(
        os.path.join(DATA_DIR, "train_image_paths.csv"),
        names=["path"]
    )

    valid_path = pd.read_csv(
        os.path.join(DATA_DIR, "valid_image_paths.csv"),
        names=["path"]
    )

    return train_labeled, valid_labeled, train_path, valid_path




