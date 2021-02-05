# SlideImages
Official code implementation for the ECIR 2020 paper *SlideImages: A Dataset for Educational Image Classification*.

The official repository has moved to [TIBHannover/SlideImages](https://github.com/TIBHannover/SlideImages).

The dataset is available along with trained weights [here](https://data.uni-hannover.de/dataset/slideimages).

# Installation
I used Anaconda as my python environment manager. You can fully recreate my environment with `conda create -n slideimages --file package-list.txt`.  Extract the dataset (note: not Windows-compatible because Windows does not support creating broken symlinks), follow the instructions for sources which need to be downloaded separately, and change dataset root parameters as necessary. 

# Testing
Use `eval.py` to generate statistical summaries.  This opens a window with a color-coded confusion matrix, so you will need to change the `summary` function in `summary_tools.py` if you need to run this in batches or headless.

# Training
Use `train.py` to train weights.  You need to supply a pre-training image --- you can use the supplied weights --- and you should note the flags you can supply to the script.

# Pre-training
If you need different pre-training, you might find it useful to start working with `preTrain.py`, the script I wrote to perform initial pre-training.
