import abc
import argparse
import datetime
import math
import os

from tensorflow import keras
import numpy as np
import pandas as pd
import dataset_util
import loss
import metric
import util
import random
import time


class MuraModel(abc.ABC):
    """
    An abstract Model object that designed to work with MURA dataset.
    """
    @classmethod
    @abc.abstractmethod
    def train_from_cli(cls, **kwargs):
        pass

    # Define argument parser so that the script can be executed directly
    # from console.
    # Global Configspr
    ROOT_PATH = os.path.abspath("./MURA-v1.1/mura_model.py")  # ?/mura_model.py
    ROOT_PATH = os.path.dirname(ROOT_PATH)  # ?/

    @classmethod
    def train_from_cli(cls):
        args = cls.ARG_PARSER.parse_args()
        arg_dict = {k: v for k, v in vars(args).items() if v is not None}
        model = cls(**arg_dict)
        model.train(**arg_dict)

    def __init__(self, model_root_path,img_size, resize=True, grayscale=False, **kwargs):
        self.img_size_origin = 512
        self.img_size = img_size
        self.img_resized = resize
        self.img_grayscale = grayscale
        self.color_channel = 1 if grayscale else 3
        self.model_root_path = model_root_path                                      # ?/models/{model_name}
        self.weight_path = os.path.join(self.ROOT_PATH, "weights")                  # ?/weights/
        self.model_save_path = os.path.join(self.model_root_path, "saved_models")   # ?/models/{model_name}/saved_models
        self.result_path = os.path.join(self.model_root_path, "results")            # ?/models/{model_name}/results
        self.cache_path = os.path.join(self.ROOT_PATH, "cache")
        print(self.cache_path)# ?/cache
        self.log_path = os.path.join(self.model_root_path, "logs")                  # /models/{model_name}/logs
        self.model = None

    def load_and_process_image(self, path, imggen=None):
        """
        Load and preprocess a single image
        Args:
            path: path to image file.
            imggen: Image Generator for performing image perturbation
        Returns:
            image in ndarray
        """
     #   img = dataset.load_image(path, self.img_grayscale)
        img = preprocess_img.load_image(path,self.img_size,self.img_grayscale)
        if self.img_resized:
           # img = resize_img(img, self.img_size)
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
        Args:
            df: Dataframe that contains all the images need to be loaded
            imggen: Image Generator for performing image perturbation
        Yields: list of resized images in ndarray, list of labels, list of path
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
    

    def input_generator(self, df, batch_size, imggen=None):
        """
        Generator that yields a batch of images with their labels
        Args:
            df: Dataframe that contains all the images need to be loaded
            batch_size: Maximum number of images in each batch
            imggen: ImageDataGenerator used to apply perturbation. If None,
                then no perturbation is applied.
        Yields: List of training images and labels in a batch
        """
        while True:
            # loop once per epoch
            df = df.sample(frac=1).reset_index(drop=True)
            for g, batch in df.groupby(np.arange(len(df)) // batch_size):
                imgs, labels, _ = self.load_imgs(batch, imggen)

                yield imgs, labels

    def img_generator(self, df, batch_size):
        """
        Generator that yields a batch of  images
        Args:
            df: Dataframe that contains all the images need to be loaded
            batch_size: Maximum number of images in each batch
        Yields: List of training images and labels in a batch
        """
        while True:
            # loop once per epoch
            for g, batch in df.groupby(np.arange(len(df)) // batch_size):
                imgs, _, _ = self.load_imgs(batch)
                yield imgs

    def load_validation(self, valid_df):
        """
        Load Validation images into memory.
        TODO: Remove once the bug with validation_data=generator is resolved.
        Args:
            valid_df:
        Returns:
        """
        img_valid, label_valid, path_valid = self.load_imgs(valid_df[:2000])

        return img_valid, label_valid, path_valid

    def prepare_imggen(self, df):
        """
        Prepare Image Generator responsible for image perturbation.
        Args:
            df: Dataframe that contains all the images to sample from.
        Returns:
        """
        imggen_args = dict(
            rotation_range=30,
            fill_mode="constant",
            cval=0,
            horizontal_flip=True
        )
        imggen = keras.preprocessing.image.ImageDataGenerator(**imggen_args)
        samples, _, _ = self.load_imgs(df.sample(1000))

        imggen.fit(np.asarray(samples))
        return imggen

    def write_prediction(self,  valid_df, batch_size,):
        """
        Run prediction using given model on a list of images,
        and write the result to a csv file.
        :param batch_size: number of inputs in each batch.
        :param valid_df: validation dataset table
        :return:
            path to result result csv
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
