import imghdr
import math
import os
import re
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modify_csv.get_info import *
        
def build_dataframe(df_label, df_path, data_dir):
    """
    Build datasets by combining image paths with labels, so that we have a dataframe
    where each row is an image and has the patient it belongs to, as well as the label
    :param df_label: labeled dataset.
    :param df_path: image paths.
    :return: training table, validation table
    """
    df_label = df_label.copy(deep=True)
    df_path = df_path.copy(deep=True)

    complete_path(data_dir,df_path, "path")
    complete_path(data_dir,df_label, "patient")

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
