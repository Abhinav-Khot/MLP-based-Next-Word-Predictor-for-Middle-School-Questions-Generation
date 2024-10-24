import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



def calculate_accuracy(y_true : np.array , y_pred : np.array) -> float:
    assert len(y_true) == len(y_pred)

    return np.mean(y_true == y_pred)

def calculate_recall(y_true : np.array , y_pred : np.array, cls : int) -> float:
    assert len(y_true) == len(y_pred)
    
    z = (y_true == cls)
    z_ = (y_pred == cls)
    return np.sum(z & z_) / np.sum(z)

def calculate_precision(y_true : np.array , y_pred : np.array, cls : int) -> float:
    assert len(y_true) == len(y_pred)
    
    z = (y_true == cls)
    z_ = (y_pred == cls)

    return np.sum(z & z_) / np.sum(z_)

def calculate_f1(y_true : np.array , y_pred : np.array, cls : int) -> float:    
    precision = calculate_precision(y_true, y_pred, cls)
    recall = calculate_recall(y_true, y_pred, cls)

    return 2 * precision * recall / (precision + recall)

def calculate_metrics(y_true : np.array , y_pred : np.array) -> pd.DataFrame:
    classes = np.unique(y_true).astype('uint8')
    metrics = pd.DataFrame(columns = ['class', 'precision', 'recall', 'f1 score'])
    for i in range(len(classes)):
        metrics.loc[i, 'class'] = classes[i]
        metrics.loc[i, 'precision'] = calculate_precision(y_true, y_pred, classes[i])
        metrics.loc[i, 'recall'] = calculate_recall(y_true, y_pred, classes[i])
        metrics.loc[i, 'f1 score'] = calculate_f1(y_true, y_pred, classes[i])
    
    metrics.index += 1
    return metrics

def calculate_and_plot_confusion_matrix(y_true : np.array, y_pred : np.array):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis = 1)[:, np.newaxis]
    labels = np.unique(y_true).astype('uint8')
    cm = pd.DataFrame(cm, columns = labels, index = labels)
    plt.figure(figsize = (10, 7))
    sns.heatmap(cm, annot = True, cmap = 'Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return cm


def pick_outliers(cm : pd.DataFrame):
    cm_ = cm.copy().to_numpy()
    dic = dict()
    for (i,j) in np.ndenumerate(cm_):
        if i[0] != i[1]:
            dic[(i[0], i[1])] = j

    outliers = sorted(dic.items(), key = lambda x: x[1], reverse = True)

    print("The top commonly confused classes are:")
    for i in range(10):
        print(outliers[i][0][0] ," is interpreted to be ", outliers[i][0][1])