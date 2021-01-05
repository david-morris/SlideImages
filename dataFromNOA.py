#!/bin/env python
### Object for ingesting NOA data
import keras
import numpy as np
import cv2
import os
## helpers
def read_dirs_as_labels(root):
    labels = dict()
    for label in os.listdir(root):
        for file in os.listdir(os.path.join(root, label)):
            labels[file] = label
    return (list(labels.keys()), labels)

class_conversions = {'Photos': 0, 'Diagrame': 1, 'Imaging': 1,
                     'composite': 1, 'VisualisierungenUndModelle': 1}
class_names = ['photos', 'born-digital']
class NoaDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset_root, split, batch_size=32, shuffle=True):
        self.list_IDs, self.labels = read_dirs_as_labels(os.path.join(dataset_root, split))
        self.dataset_root = dataset_root
        self.split = split
        self.dim = (256, 256)
        self.batch_size = batch_size
        self.n_channels = 3
        self.n_classes = 2
        self.shuffle = shuffle
        self.on_epoch_end()

    def filename(self, index):
        return os.path.join(self.dataset_root, self.split, self.labels[self.list_IDs[index]], self.list_IDs[index])

    def image(self, index):
        image = cv2.imread(self.filename(index))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.dim)
        return np.array(image)


    def __len__(self):
        return(int(np.floor(len(self.list_IDs) / self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.classes = [class_conversions[self.labels[self.list_IDs[index]]] for index in self.indexes]
        self.classes = self.classes[:len(self.classes)-(len(self.classes) % self.batch_size)]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = cv2.imread(os.path.join(self.dataset_root, self.split, self.labels[ID], ID))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.dim)
            X[i,] = np.array(image)
            y[i] = class_conversions[self.labels[ID]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
