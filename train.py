#!/bin/env python
### A Keras classifier for the first level of the hierarchy
from callbacks import TuneScheduler
import keras
import tensorflow as tf
import os
from filesystem_datastream import FilesystemDatastream,SlideFigureNoSlides,DocFigureEquivalents,CombinedDatasets
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, callbacks
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D, Concatenate
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from summary_tools import summary
import keras.backend as K

## Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("pretrain")
parser.add_argument("output")
#parser.add_argument("learning_rate", type=float)
#parser.add_argument("decay", type=float)
parser.add_argument("--load_frozen", action="store_true")
parser.add_argument("--frozen_path", default=None)
parser.add_argument("--dataset", default="multi_full")
args = parser.parse_args()


momentum=0.9
import socket
if socket.gethostname() == 'devbox1.research.tib.eu':
    dataset_root = "/data/multisource_classification/"
    headless = True
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
else:
    if args.dataset == "docfigure":
        dataset_root = "/home/morris/datasets/docfigure/docfigure"
        train_root = os.path.join(dataset_root, "train")
        val_root = os.path.join(dataset_root, "val")
    else:
        dataset_root = "/home/morris/datasets/multisource_classification/"
        train_root = os.path.join(dataset_root, "train")
        val_root = os.path.join(dataset_root, "val")
    headless = False

model_name = args.output+".h5"


## Model: pre-trained application which fine-tunes fast
# base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
base_model = load_model(args.pretrain)
base_model.layers.pop()
base_model.layers.pop()
x = base_model.output
x = Dropout(0.1, name='dropout_-1')(x)
#x = GlobalAveragePooling2D()(x)
#x = Dropout(0.7)(x)
x = Dense(512, activation='relu', name='dense_-1')(x)
if args.dataset == "multi_full" or args.dataset == "combined":
    predictions = Dense(9, activation='softmax', name="flat_predictions")(x)
else:
    predictions = Dense(8, activation='softmax', name="flat_predictions")(x)

model = keras.Model(inputs = base_model.input, outputs = predictions)


augment = keras.preprocessing.image.ImageDataGenerator(
    fill_mode="constant", cval = 0.0,
    width_shift_range=0.05, height_shift_range=0.05,
    brightness_range=[0.8, 1.2], channel_shift_range=0.2,
    zoom_range=[0.95, 1.1])
# training_generator = augment.flow_from_directory("./simpleDataset", batch_size=64, shuffle=False,
#                                                  target_size=(256,256),
#                                                  class_mode="categorical")
if args.dataset == "multi_full":
    training_generator = FilesystemDatastream(train_root, batch_size = 32, augment=augment)
    validation_generator = FilesystemDatastream(val_root, shuffle=False, batch_size=1)
elif args.dataset == "multi_slim":
    training_generator   = SlideFigureNoSlides(train_root, batch_size = 32, augment=augment)
    validation_generator = SlideFigureNoSlides(val_root, shuffle=False, batch_size=1)
elif args.dataset == "docfigure":
    training_generator   = DocFigureEquivalents(train_root, batch_size = 32, augment=augment)
    validation_generator = DocFigureEquivalents(val_root, shuffle=False, batch_size=1)
elif args.dataset == "combined":
    training_generator = CombinedDatasets(augment=augment)
    validation_generator = FilesystemDatastream(val_root, shuffle=False, batch_size=1)

else:
    raise ValueError("Bad dataset name")


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
# freeze layers first
if not args.load_frozen:
    for layer in model.layers[:-2]:
        layer.trainable = False

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=training_generator,
                        validation_data = validation_generator,
                        use_multiprocessing=False, epochs=5,
                        callbacks=[callbacks.ModelCheckpoint("frozen_" + model_name, save_best_only=True),
                                    LRTensorBoard(log_dir="frozen_" + args.output)])
# restore the best checkpoint
if args.load_frozen and args.frozen_path is not None:
    model = load_model(args.frozen_path)
else:
    model = load_model("frozen_" + model_name)
# unfreeze layers and train more
for layer in model.layers:
    layer.trainable = True

sgd = optimizers.SGD(lr=1e-3, momentum=0.9)
model.compile(optimizer="adam",
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(generator=training_generator,
                    validation_data = validation_generator,
                    use_multiprocessing=False, epochs=40,
                    callbacks=[callbacks.ModelCheckpoint(model_name, save_best_only=True),
                               LRTensorBoard(log_dir=args.output),
                               TuneScheduler()])


if not headless:
    summary(model, validation_generator)
