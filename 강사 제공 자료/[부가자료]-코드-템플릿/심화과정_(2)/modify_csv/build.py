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
    """
    df_label = df_label.copy(deep=True)
    df_path = df_path.copy(deep=True)

    complete_path(data_dir,df_path, "path")
    complete_path(data_dir,df_label, "patient")

    # Apply a transformation over each row to save image directory as a new column
    df_path["patient"] = df_path.apply(get_patient, axis=1)

    # Merge two table on patient column
    result = df_path.merge(df_label, on="patient")

    classify_bpart(result)

    # change .../patient00001/... to patient00001
    result["patient"] = result["patient"].str.extract("(patient\d{5})")

    # Apply a transformation over each row to create a column for unique
    # patient-bpart-study combo
    result["study"] = result.apply(extract_study, axis=1)
    return result
