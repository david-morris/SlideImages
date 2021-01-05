from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import random
import matplotlib
import itertools

def show_misclassed(y_true, y_pred, gen, class_names, samples=10):
    misclassed = [(actual, pred, index) for index, (pred, actual) in
                  enumerate(zip(y_pred, y_true)) if pred != actual]
    misclassed = random.sample(misclassed, k=samples)
    for actual, pred, index in misclassed:
        plt.figure()
        plt.suptitle(class_names[actual] + " classed as " + class_names[pred])
        plt.imshow(gen.image(index))

def eric_plot(cm,
              classes,
              title='',
              cmap=plt.cm.Blues,
              normalized=True,
              xlabel='',
              ylabel='',
              draw_rect=False,
              multiplier=1,
              figsize=(12, 12), stats=None):

    # classes = classes.tolist()
    matplotlib.rcParams['font.size'] = 30
    f, ax = plt.subplots(figsize=figsize)
    #plt.rcParams.update({'font.size': 14})
    cm_scaled = np.array(cm, copy=True)
    cm_scaled[np.where(cm == 0)] = -0.1*np.max(cm)
    plt.imshow(cm_scaled, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.ylim(len(classes) - 0.5, -0.5)

    if multiplier == 1:
        fmt = '.0f'
    else:
        fmt = '.0f'

    cm = multiplier * cm
    thresh = 0.80 * np.max(cm)
    if not normalized:
        thresh = 100
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.text(cm.shape[0], -1, "Support", color="black", fontsize=30)

    plt.tight_layout()
    # print('writing file to {}'.format(fout))
    plt.show()

def sb_plot(conf_mat, classes, title=None, cmap="Blues"):
    frame = pd.DataFrame(conf_mat, columns=classes, index=classes)
    frame.index.name = 'Actual'
    frame.columns.name = 'predicted'
    plt.figure()
    plt.suptitle(title)
    sn.set(font_scale=0.6)
    sn.heatmap(frame, cmap=cmap, annot=True, annot_kws={"size": 16})

def plot_conf_mat(conf_mat, classes, cmap=plt.cm.Blues, title=None):
    '''Plot a calculated conf matrix'''
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax

def yolomometry(model, generator, hierarchy):
    n_top = len(hierarchy["top"])
    n_mid = len(hierarchy["mid"])
    n_fine = len(hierarchy["fine"])
    mid_nums = {name: i for (i, name) in enumerate(hierarchy["mid"])}
    top_nums = {name: i for (i, name) in enumerate(hierarchy["top"])}
    y_true_fine = generator.classes
    y_pred_probs = model.predict_generator(generator)

    fine_names = generator.class_names
    y_pred_fine = np.argmax(y_pred_probs[:,n_top+n_mid:], axis=1)

    mid_names = hierarchy["mid"]
    y_true_mid = [mid_nums[hierarchy["fine_mid"][hierarchy["fine"][fine_index]]] for fine_index in y_true_fine]
    y_pred_mid = np.argmax(y_pred_probs[:,n_top:n_top+n_mid], axis=1)

    top_names = hierarchy["top"]
    y_true_top = [top_nums[hierarchy["mid_top"][hierarchy["mid"][mid_index]]] for mid_index in y_true_mid]
    y_pred_top = np.argmax(y_pred_probs[:,:n_top], axis=1)

    # y_true_fine = list(map(lambda x: x+1, y_true_fine))
    # y_true_mid = list(map(lambda x: x+1, y_true_mid))
    # y_true_top = list(map(lambda x: x+1, y_true_top))
    # y_pred_fine = list(map(lambda x: x+1, y_pred_fine))
    # y_pred_mid = list(map(lambda x: x+1, y_pred_mid))
    # y_pred_top = list(map(lambda x: x+1, y_pred_top))
    print("Fine Report:")
    print(classification_report(y_true_fine, y_pred_fine, target_names=hierarchy["fine"]))
    print("Mid Report:")
    print(classification_report(y_true_mid, y_pred_mid, target_names=hierarchy["mid"]))
    print("Top Report:")
    print(classification_report(y_true_top, y_pred_top, target_names=hierarchy["top"]))


def summary(model, generator, keras_model=True):
    if keras_model:
        posteriors = model.predict_generator(generator)
        y_pred = np.argmax(posteriors, axis=1)
        for idx in enumerate(y_pred):
            posteriors[idx] = 0.0
        second = np.argmax(posteriors, axis=1)
        class_names = generator.class_names
        y_true = generator.classes
        toptwo = np.copy(y_pred)
        for i in range(len(y_true)):
            if second[i] == y_true[i]:
                toptwo[i] = second[i]
    else:
        # get the class names (already extracted)
        class_names = generator
        # get labels and predictions (already extracted)
        y_true, y_pred = model

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names = class_names))
    conf_mat = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    np.set_printoptions(precision=2)
    print(normalize(conf_mat, axis=0, norm='l1'))
    if keras_model:
        # show_misclassed(y_true, y_pred, generator, class_names)
        pass
    #eric_plot(conf_mat, class_names, title="Instances")
    #sb_plot(normalize(conf_mat, axis=1, norm='l1'), class_names, title="sensitivity")
    #sb_plot(normalize(conf_mat, axis=0, norm='l1'), class_names, title="precision")
    stats = classification_report(y_true, y_pred, target_names = class_names, output_dict=True)
    eric_plot(normalize(conf_mat, axis=1, norm='l1'), class_names,
              title="% ← classified as ↓", multiplier=100,
              stats=stats)
    print(normalize(conf_mat, axis=1, norm='l1'))
    if keras_model:
        print("Top-Two Accuracy")
        print(classification_report(y_true, toptwo, target_names = class_names))
    return conf_mat
