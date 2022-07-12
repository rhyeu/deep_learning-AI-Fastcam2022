import abc
import argparse
import datetime
import math
import os
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import time
from preprocess import preprocess_img
from mura.generator_class import Generator

class Bonemodel(Generator):
 
    def __init__(self, model_root_path,img_size, resize=True, grayscale=False, **kwargs):
        self.img_size_origin = 512
        self.img_size = img_size
        self.img_resized = resize
        self.img_grayscale = grayscale
        self.color_channel = 1 if grayscale else 3
        self.model_root_path = model_root_path                                    
        self.weight_path = os.path.join(self.ROOT_PATH, "weights")                
        self.model_save_path = os.path.join(self.model_root_path, "saved_models")   
        self.result_path = os.path.join(self.model_root_path, "results")          
       # self.cache_path = os.path.join(self.ROOT_PATH, "cache")
        self.log_path = os.path.join(self.model_root_path, "logs")              
        self.model = None

    def load_and_process_image(self, path, imggen=None):
        """
        Load and preprocess a single image
        """
        img = preprocess_img.load_image(path,self.img_size,self.img_grayscale)
        if self.img_resized:
            img=img/255
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            img = (img- imagenet_mean) / imagenet_std
            img= preprocess_img.zero_pad(img,self.img_size)
        if imggen:
            img = imggen.random_transform(img)
        return img

    def load_imgs(self, df, imggen=None):
        """
        Generator that loads all the images from the dataframe and and return them
        as ndarrays
        """
        imgs = []
        labels = []
        paths = []
        for index, row in df.iterrows():
            img = self.load_and_process_image(row["path"], imggen)
        
            imgs.append(img)
            labels.append(row["label"])
            paths.append(row["path"])

        return np.asarray(imgs), np.asarray(labels), np.asarray(paths)
    
    def load_validation(self, valid_df):
        """
        Load Validation images into memory.
        """
        img_valid, label_valid, path_valid = self.load_imgs(valid_df)

        return img_valid, label_valid, path_valid

    def write_prediction(self,  valid_df, batch_size,):
        """
        Run prediction using given model on a list of images,
        and write the result to a csv file.
        """
        predictions = self.model.predict_generator(
            self.img_generator(valid_df, batch_size),
            steps=math.ceil(valid_df.shape[0] / batch_size)
        )
        util.create_dir(self.result_path)
        for i in range(len(predictions)):
            valid_df.at[i, "prediction"] = predictions[i]

        result_path = os.path.join(self.result_path, "densenet_{:%Y-%m-%d-%H%M}.csv".format(
            datetime.datetime.now()
        )
                                   )
        valid_df.to_csv(result_path)

        return result_path
