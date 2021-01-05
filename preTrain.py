#!/bin/env python
### A Keras classifier for the first level of the hierarchy
import keras
import tensorflow as tf
import os
from filesystem_datastream import FilesystemDatastream
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, callbacks
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import keras.backend as K
from summary_tools import summary

## Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("learning_rate")
parser.add_argument("decay")
args = parser.parse_args()

dataset_root = "/home/morris/datasets/NOA_classification/testSplit/"
import socket
if socket.gethostname() == 'devbox1.research.tib.eu':
    dataset_root = "/data/noa/testSplit/testSplit/"
train = True
#modelName = 'pretrain-sgd-mobilenet.h5'
modelName = args.output + ".h5"
class_names =  ["composite", "Diagrame", "Imaging", "Photos", "VisualisierungenUndModelle"]

lr = float(args.learning_rate)
decay = float(args.decay)
# lr = 1E-4
momentum=0.9
# decay=1E-4

## Model: pre-trained application which fine-tunes fast
base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256,256,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(5, activation='softmax')(x)
model = keras.Model(inputs = base_model.input, outputs = predictions)


## Dataset: NOA dataset
augment = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.05, height_shift_range=0.05, fill_mode="constant", cval = 0.0,
    brightness_range=[0.8, 1.2], zoom_range=[0.95, 1.5])
training_generator = FilesystemDatastream(dataset_root+ "train/", class_names = class_names,
                                          batch_size=64, augment=augment)
validation_generator = FilesystemDatastream(dataset_root+ "val/", class_names = class_names)

class LRTensorBoard(callbacks.TensorBoard):
    def __init__(self, log_dir="./logs", write_graph=False):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'sgd_lr': K.eval(
            self.model.optimizer.lr *
            (1./
             (1. + model.optimizer.decay * K.cast(model.optimizer.iterations,
                                                  K.dtype(model.optimizer.decay))))),
                     'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

## Train the model

# Train upper layers first
for layer in model.layers[:-2]:
    layer.trainable = False

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False, workers=6, epochs=1,
                    callbacks=[callbacks.ModelCheckpoint("frozen_" + modelName, save_best_only=True),
                               LRTensorBoard(log_dir="frozen_" + args.output)])

# Restore the best weights and train the whole network
model = load_model("frozen_" + modelName)
for layer in model.layers:
    layer.trainable = True

sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=decay)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy', metrics=['accuracy'])



model.fit_generator(generator=training_generator,
                    validation_data = validation_generator,
                    use_multiprocessing=False, workers=6, epochs=10,
                    callbacks=[callbacks.ModelCheckpoint(modelName, save_best_only=True),
                               LRTensorBoard(log_dir=args.output)])
