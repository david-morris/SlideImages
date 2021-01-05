#!/bin/env python
### A Keras classifier for the first level of the hierarchy
import keras
import tensorflow as tf
import os
from filesystem_datastream import FilesystemDatastream
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from summary_tools import summary

## Parameters
dataset_root = "/home/morris/datasets/NOA_classification/testSplit/"
train = False
modelName = 'testTwoGroups.h5'

## Helpers


## Model: pre-trained application which fine-tunes fast
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation='softmax')(x)
model = keras.Model(inputs = base_model.input, outputs = predictions)


## Dataset: NOA dataset
training_generator = NoaDataGenerator(dataset_root, "train")
validation_generator = NoaDataGenerator(dataset_root, "val", shuffle=False)

## Train the model

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
if train:
    model.fit_generator(generator=training_generator,
                        validation_data = validation_generator,
                        use_multiprocessing=False, workers=6)
    model.save(modelName)
else:
    model = load_model(modelName)

summary(model, validation_generator)
