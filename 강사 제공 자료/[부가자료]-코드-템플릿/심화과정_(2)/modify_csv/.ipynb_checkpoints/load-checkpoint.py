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
ROOT_DIR = os.path.dirname(DATA_DIR)                         


TRAIN_DIR = os.path.join(DATA_DIR, "train")

VALID_DIR = os.path.join(DATA_DIR, "valid")

BPARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


def load_dataframe():
    """
     Import csv files into Dataframes.
    :return:
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


def classify_bpart(data):
    """
    Divide TRAIN_LABELED into sub-sets based on the body parts in the image.
    Also add body part as a new feature of the dataset.
    :param data: dataset to process.
    :return:
    """
    for bpart in BPARTS:
        data.loc[data["path"].str.contains(bpart.upper()), "body_part"] = bpart


def complete_path(data, column):
    """
    Convert relative image path to absolute path so that the execution does not depend
    on working directory. Also clean up the patient name
    :param data: dataset to process.
    :param column: column to perform the operation.
    :return:
    """
    data[column] = np.where(
        data[column].str.startswith("MURA-v" + DATA_VIR),
        data[column].str.replace("MURA-v" + DATA_VIR, DATA_DIR),
        data[column]
    )


def extract_study(row):
    """
    Callback function to generate a column for unique patient-study combo.
    :param row: a row from processing table
    :return:
    """
    match = re.search("study\d+", row["path"])
    if match:
        study = match.group()
        return "{}-{}-{}".format(row["patient"], row["body_part"], study)
    else:
        raise ValueError("study not found in " + row["path"])


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


def build_dataframe(df_label, df_path):
    """
    Build datasets by combining image paths with labels, so that we have a dataframe
    where each row is an image and has the patient it belongs to, as well as the label
    :param df_label: labeled dataset.
    :param df_path: image paths.
    :return: training table, validation table
    """
    df_label = df_label.copy(deep=True)
    df_path = df_path.copy(deep=True)

    complete_path(df_path, "path")
    complete_path(df_label, "patient")

    # Apply a transformation over each row to save image directory as a new column
    df_path["patient"] = df_path.apply(get_patient, axis=1)

    # Merge two table on patient column
    result = df_path.merge(df_label, on="patient") # 2개 df path df label 합치기

    classify_bpart(result) #bodypart로 되어있는거 합치기

    # change .../patient00001/... to patient00001
    result["patient"] = result["patient"].str.extract("(patient\d{5})") #patient number 경로에서 추출

    # Apply a transformation over each row to create a column for unique
    # patient-bpart-study combo
    result["study"] = result.apply(extract_study, axis=1) #STUDY - ~- -~ 형태 추가
    return result


def preprocess():
    """
    Preprocess datasets.
    :return: training set, validation set
    """
    train_labeled, valid_labeled, train_path, valid_path = load_dataframe()
    df_train = build_dataframe(train_labeled, train_path)
    df_valid = build_dataframe(valid_labeled, valid_path)

    return df_train, df_valid