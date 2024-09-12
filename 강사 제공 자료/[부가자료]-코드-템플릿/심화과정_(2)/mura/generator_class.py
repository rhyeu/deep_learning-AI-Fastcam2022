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


class Generator(abc.ABC):
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
    ROOT_PATH = os.path.abspath("./MURA-v1.1/mura_model.py")
    ROOT_PATH = os.path.dirname(ROOT_PATH)

    @classmethod
    def train_from_cli(cls):
        args = cls.ARG_PARSER.parse_args()
        arg_dict = {k: v for k, v in vars(args).items() if v is not None}
        model = cls(**arg_dict)
        model.train(**arg_dict)    

    def input_generator(self, df, batch_size, imggen=None):
        """
        Generator that yields a batch of images with their labels
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
        """
        while True:
            # loop once per epoch
            for g, batch in df.groupby(np.arange(len(df)) // batch_size):
                imgs, _, _ = self.load_imgs(batch)
                yield imgs

    def prepare_imggen(self, df):
        """
        Prepare Image Generator responsible for image perturbation.
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

