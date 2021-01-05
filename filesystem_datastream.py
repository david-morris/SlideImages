import keras
import numpy as np
import cv2
import os

multisource_presets = ["maps",       "x_y_plots",          "photos",
                       "piecharts",  "slides",             "structured_diagrams",
                       "tables",     "technical_drawings", "bar_charts"]

crosstesting_presets = ["maps",       "x_y_plots",          "photos",
                        "piecharts",               "structured_diagrams",
                        "tables",     "technical_drawings", "bar_charts"]

docfigure_equivalents = ["Geographic map", "Graph plots", "Natural images",
                         "Pie chart", "Flow chart",
                         "Tables", "Sketches", "Bar plots"]
def find_roots():
    """Function for finding the needed paths.  Customize as needed."""

class FilesystemDatastream(keras.utils.Sequence):
    def _read_dirs_as_labels(self, root):
        labels = dict()
        #for label in os.listdir(root):
        for label in self.class_names:
            for file in os.listdir(os.path.join(root, label)):
                labels[file] = label
        return(list(labels.keys()), labels)

    def __init__(self, dataset_root, batch_size=32, shuffle=True,
                 dim=(256, 256), class_names=multisource_presets, augment=None):
        self.class_names = class_names
        self.list_IDs, self.labels = self._read_dirs_as_labels(os.path.join(dataset_root))
        self.dataset_root = dataset_root
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.n_channels = 3
        self.n_classes = len(class_names)
        self.shuffle = shuffle
        self.class_conversions = {name: i for (i, name) in enumerate(class_names)}
        self.on_epoch_end()

    def filename(self, index):
        return os.path.join(self.dataset_root, self.labels[self.list_IDs[index]], self.list_IDs[index])

    def image(self, index):
        image = cv2.imread(self.filename(index))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.dim)
        return np.array(image)

    def debug(self, index):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(self.filename(index))
        plt.suptitle(self.labels[self.list_IDs[index]])
        plt.imshow(self.image(index))
        plt.show()
        return

    def __len__(self):
        return(int(np.floor(len(self.list_IDs) / self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.classes = [self.class_conversions[self.labels[self.list_IDs[index]]] for index in self.indexes]
        self.classes = self.classes[:len(self.classes)-(len(self.classes) % self.batch_size)]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            path = os.path.join(self.dataset_root, self.labels[ID], ID)
            image = cv2.imread(os.path.join(self.dataset_root, self.labels[ID], ID))
            if image is None:
                print("BAD IMAGE: " + path)
                import pdb; pdb.set_trace()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.dim)
            if self.augment is not None:
                image = self.augment.random_transform(image)
            X[i,] = np.array(image)
            y[i] = self.class_conversions[self.labels[ID]]


        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

class SlideFigureNoSlides(FilesystemDatastream):
    def __init__(self, dataset_root, batch_size=32, shuffle=True,
                 dim=(256,256), augment=None):
        super().__init__(dataset_root, batch_size, shuffle, dim, crosstesting_presets, augment)


class DocFigureEquivalents(FilesystemDatastream):
    def __init__(self, dataset_root, batch_size=32, shuffle=True,
                 dim=(256,256), augment=None):
        super().__init__(dataset_root, batch_size, shuffle, dim, docfigure_equivalents, augment)

class CombinedDatasets(keras.utils.Sequence):
    def __init__(self, docfig_root="/home/morris/datasets/docfigure/docfigure/train",
                 slidefig_root="/home/morris/datasets/multisource_classification/train/",
                 batch_size=32, shuffle=True, dim=(256,256), augment=None):
        self.slidefig = FilesystemDatastream(slidefig_root, int(batch_size / 2), shuffle,
                                             dim,class_names=multisource_presets,augment=augment)
        self.docfig = DocFigureEquivalents(docfig_root, int(batch_size / 2), shuffle, dim, augment)
        self.class_names = multisource_presets

    def __len__(self):
        return len(self.slidefig)

    def __getitem__(self, index):
        sX, sy = self.slidefig.__getitem__(index)
        dX, dy = self.docfig.__getitem__(index)
        # fix the labels after slides
        dy = np.insert(dy, 4, 0.0, axis=1)
        # don't worry about interleaving items within a batch
        return np.concatenate((sX, dX)), np.concatenate((sy, dy))

    def on_epoch_end(self):
        self.slidefig.on_epoch_end()
        self.docfig.on_epoch_end()
