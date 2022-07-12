import imghdr
import math
import os
import re
import cv2
from tensorflow import keras

        
def selectmodel(model,img_size):
    if model=="densenet121":
        inputs = keras.layers.Input(shape=(img_size, img_size, 3))
        densenet_model=keras.applications.DenseNet121(include_top=False,input_tensor=inputs,weights=None,pooling="avg")
        output = keras.layers.Dense(1,activation="sigmoid",name="predictions")(densenet_model.output)
        model = keras.models.Model(inputs=inputs,outputs=[output],name="DenseNet121")
        return model
    if model=="densenet169":
        inputs = keras.layers.Input(shape=(img_size, img_size, 3))
        densenet_model=keras.applications.DenseNet169(include_top=False,input_tensor=inputs,weights=None,pooling="avg")
        output = keras.layers.Dense(1,activation="sigmoid",name="predictions")(densenet_model.output)
        model = keras.models.Model(inputs=inputs,outputs=[output],name="DenseNet169")
        return model
    if model=="densenet201":
        inputs = keras.layers.Input(shape=(img_size, img_size, 3))
        densenet_model=keras.applications.DenseNet201(include_top=False,input_tensor=inputs,weights=None,pooling="avg")
        output = keras.layers.Dense(1,activation="sigmoid",name="predictions")(densenet_model.output)
        model = keras.models.Model(inputs=inputs,outputs=[output],name="DenseNet201")
        return model
    if model=="resnet50":
        inputs = keras.layers.Input(shape=(img_size, img_size, 3))
        densenet_model=keras.applications.ResNet50(include_top=False,input_tensor=inputs,weights=None,pooling="avg")
        output = keras.layers.Dense(1,activation="sigmoid",name="predictions")(densenet_model.output)
        model = keras.models.Model(inputs=inputs,outputs=[output],name="ResNet50")
        return model
    if model=="resnet101":
        inputs = keras.layers.Input(shape=(img_size, img_size, 3))
        densenet_model=keras.applications.ResNet101(include_top=False,input_tensor=inputs,weights=None,pooling="avg")
        output = keras.layers.Dense(1,activation="sigmoid",name="predictions")(densenet_model.output)
        model = keras.models.Model(inputs=inputs,outputs=[output],name="ResNet101")
        return model
    if model=="resnet152":
        inputs = keras.layers.Input(shape=(img_size, img_size, 3))
        densenet_model=keras.applications.ResNet152(include_top=False,input_tensor=inputs,weights=None,pooling="avg")
        output = keras.layers.Dense(1,activation="sigmoid",name="predictions")(densenet_model.output)
        model = keras.models.Model(inputs=inputs,outputs=[output],name="ResNet152")
        return model