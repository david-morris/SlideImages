import sys
import os
import cv2
import keras
import h5py
from filesystem_datastream import FilesystemDatastream
from keras.models import Sequential,Model,load_model
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.decomposition import PCA
from summary_tools import eric_plot
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--head_height", type=int, default=1)
parser.add_argument("-e", "--embeddings", action="store_true")
args = parser.parse_args()

# multisource_presets = ["imaging",             "linecharts", "maps",
#                        "photos",              "piecharts",  "slides",
#                        "structured_diagrams", "tables",     "technical_drawings",
#                        "scatter_plots", "bar_charts"]

multisource_presets = ["imaging",             "maps",       "x_y_plots",
                       "photos",              "piecharts",  "slides",
                       "structured_diagrams", "tables",     "technical_drawings", "bar_charts"]

reduced_presets = ["imaging",             "linecharts", "maps",
                   "photos",              "piecharts",  "slides",
                   "structured_diagrams", "tables",     "bar_charts"]

# multisource_presets = reduced_presets

train_root = "/home/morris/datasets/multisource_classification/train"
val_root = "/home/morris/datasets/multisource_classification/val"
train_embeddings = list()
train_labels = list()
val_embeddings = list()
val_labels = list()
if args.embeddings:
    model = load_model(args.model, compile=False)
    # remove the head
    for i in range(args.head_height):
        model.layers.pop()
    for label in multisource_presets:
        label_folder = os.path.join(train_root, label)
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.dim, args.dim))
            img_mat = np.array(image, ndmin=4)
            img_embedding = model.predict(img_mat)
            train_labels.append(np.string_(label))
            train_embeddings.append(img_embedding)
    train_embeddings = np.array(train_embeddings)
    for label in multisource_presets:
        label_folder = os.path.join(val_root, label)
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.dim, args.dim))
            img_mat = np.array(image, ndmin=4)
            img_embedding = model.predict(img_mat)
            val_labels.append(np.string_(label))
            val_embeddings.append(img_embedding)
    val_embeddings = np.array(val_embeddings)
    with h5py.File("embeddings_" + args.model, 'w') as hf:
        hf['train_embeddings'] = train_embeddings
        hf['train_labels'] = train_labels
        train_embeddings = np.squeeze(train_embeddings)
        hf['val_embeddings'] = val_embeddings
        hf['val_labels'] = val_labels
        val_embeddings = np.squeeze(val_embeddings)
else:
    with h5py.File("embeddings_" + args.model, 'r') as hf:
        train_embeddings = np.squeeze(hf['train_embeddings'][:])
        train_labels = hf['train_labels'][:]
        val_embeddings = np.squeeze(hf['val_embeddings'][:])
        val_labels = hf['val_labels'][:]


### Begin kNN classification baseline
classes = {np.string_(name): i for (i,name) in enumerate(multisource_presets)}
# Describe our classes in terms of NOA classes to see if the NOA classes are separated better
simplifier = {""}

clf = NearestCentroid()
clf = KNeighborsClassifier(n_neighbors=5)


#y_pred = kmeans.fit_predict(embeddings)
y_true_train = [classes[x] for x in train_labels]
y_true_val = [classes[x] for x in val_labels]
clf.fit(train_embeddings, y_true_train)
y_pred_train = clf.predict(train_embeddings)
y_pred_val = clf.predict(val_embeddings)


#pca = PCA(n_components = 2)
#components = pca.fit_transform(embeddings)


# plt.scatter(components[:,0], components[:,1], c=y_true)
# plt.show()

conf_mat_train = confusion_matrix(y_true_train, y_pred_train)
conf_mat_val = confusion_matrix(y_true_val, y_pred_val)
print(classification_report(y_true_train, y_pred_train, target_names=multisource_presets))
eric_plot(conf_mat_train, multisource_presets, "Train:")
print(classification_report(y_true_val, y_pred_val, target_names=multisource_presets))
eric_plot(conf_mat_val, multisource_presets, "Val:")
