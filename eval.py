#!/bin/env python
import sys
import os
import keras
from filesystem_datastream import FilesystemDatastream,SlideFigureNoSlides,DocFigureEquivalents
from keras.models import Sequential,Model,load_model
from summary_tools import summary, yolomometry
# from pytorch_tools import load_torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--test", action="store_true")
parser.add_argument("--docfigure", action="store_true")
parser.add_argument("--trim", action="store_true")
parser.add_argument("--pytorch", action="store_true")
args = parser.parse_args()

if args.pytorch:
    model = load_torch(args.model)
else:
    model = load_model(args.model, compile=False)

if args.test:
    if args.docfigure:
        dataset_root = "/home/morris/datasets/docfigure/docfigure/new_test/"
        generator = DocFigureEquivalents(dataset_root, batch_size=1, shuffle=False)
    else:
        dataset_root = "/home/morris/datasets/labeled_slidewiki/"
        if args.trim:
            generator = SlideFigureNoSlides(dataset_root, batch_size=1, shuffle=False)
        else:
            generator = FilesystemDatastream(dataset_root, batch_size=1, shuffle=False)
else:
    dataset_root = "/home/morris/datasets/multisource_classification/"
    val_root = os.path.join(dataset_root, "val")
    generator = FilesystemDatastream(val_root,batch_size=1, shuffle=False)


summary(model, generator)
