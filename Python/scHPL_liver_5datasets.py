"""
Title: Optimizing cell type annotation based on single cell transcriptomes of healthy liver
        utilizing a hierarchical progressive learning method
Author: Youkyung Lim
Affiliation: MSc Bioinformatics and Systems Biology, Vrije Universiteit Amsterdam
Date: 10/02/2022
Usage: $python3 main.py [RESOLUTION: 'high' or 'low'] [FLAT_CLASSIFIER: 'svm' or 'svm_occ']
"""

#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import time as tm
from scHPL import train, predict, update, progressive_learning, utils, evaluate
from sklearn.metrics import zero_one_loss, adjusted_rand_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

PATH_INPUT = 'integrated_data/5/'
PATH_OUTPUT = 'output/5/'
FEATURES = ['2000', '5000'] # number of columns (features, genes) in the integrated datasets
FLAT_CLASSIFIER = ['svm', 'svm_occ'] # linear SVM, or One-class SVM

def ReadDatasets(FEATURES):
    # Read integrated datasets (transposed as columns = features/genes) and their labels
    macparland2018_data = pd.read_csv(PATH_INPUT + FEATURES + '/macparland2018_data.csv', index_col=0, sep=',')
    macparland2018_label = pd.read_csv(PATH_INPUT + FEATURES + '/macparland2018_label.csv', header=0, index_col=None, sep=',')
    macparland2021_data = pd.read_csv(PATH_INPUT + FEATURES + '/macparland2021_data.csv', index_col=0, sep=',')
    macparland2021_label = pd.read_csv(PATH_INPUT + FEATURES + '/macparland2021_label.csv', header=0, index_col=None, sep=',')
    tamburini_data = pd.read_csv(PATH_INPUT + FEATURES + '/tamburini_data.csv', index_col=0, sep=',')
    tamburini_label = pd.read_csv(PATH_INPUT + FEATURES + '/tamburini_label.csv', header=0, index_col=None, sep=',')
    aizarani_data = pd.read_csv(PATH_INPUT + FEATURES + '/aizarani_data.csv', index_col=0, sep=',')
    aizarani_label = pd.read_csv(PATH_INPUT + FEATURES + '/aizarani_label.csv', header=0, index_col=None, sep=',')
    segal_data = pd.read_csv(PATH_INPUT + FEATURES + '/segal_data.csv', index_col=0, sep=',')
    segal_label = pd.read_csv(PATH_INPUT + FEATURES + '/segal_label.csv', header=0, index_col=None, sep=',')

    # Make the labels unique by adding its dataset name
    macparland2018_label = macparland2018_label['x'].map(str) + ' - macparland2018'
    macparland2021_label = macparland2021_label['x'].map(str) + ' - macparland2021'
    tamburini_label = tamburini_label['x'].map(str) + ' - tamburini'
    aizarani_label = aizarani_label['x'].map(str) + ' - aizarani'
    segal_label = segal_label['x'].map(str) + ' - segal'

    # Put a dataset and its label in a tuple, then put all of them in one list
    datasets = [(macparland2018_data, macparland2018_label), (macparland2021_data, macparland2021_label),
                (aizarani_data, aizarani_label), (tamburini_data, tamburini_label), (segal_data, segal_label)]

    return datasets


def SortbyResolution(datasets, RESOLUTION):
    # Sort datasets by the number of unique labels, and the last one is the test set
    if RESOLUTION == "high":
        # high to low : macparland2021 (30) - macparland2018 (20) - aizarani (10) - tamburini (7) - segal (2)
        datasets = sorted(datasets, key=lambda x: x[1].nunique(), reverse=True)
        data_train = datasets[:len(datasets) - 1]
        data_test = datasets[len(datasets) - 1][0]
        y_true = datasets[len(datasets) - 1][1]
        return data_train, data_test, y_true

    elif RESOLUTION == "low":
        # low to high :  segal (2) - tamburini (7) - aizarani (10) - macparland2018 (20) - macparland2021 (30)
        datasets = sorted(datasets, key=lambda x: x[1].nunique())
        data_train = datasets[:len(datasets) - 1]
        data_test = datasets[len(datasets) - 1][0]
        y_true = datasets[len(datasets) - 1][1]
        return data_train, data_test, y_true

    else:
        print("Usage: $python3 scHPL_liver_5datasets.py [RESOLUTION: 'high' or 'low'] [FLAT_CLASSIFIER: 'svm' or 'svm_occ']")

def UniqueLabels(data_train):
    # if identical labels exist in the training data, change the later one to have '_' at the end?
    # put all unique labels in one
    lst_labels = [x[1].unique() for x in data_train]
    # compare to the string with the earlier index
    lst_original = []
    lst_unique = []
    for idx, x in enumerate(lst_labels): # i = a list of unique labels per dataset
        for y in x:
            lst_original.append(y)
            if y not in lst_unique:
                lst_unique.append(y)
            else:
                # if they are identical (or already in the unique list), add '_' to the latter one
                y1 = y + '_'
                lst_unique.append(y1)
                data_train[idx][1].replace(y, y1)
    return data_train

def ParseDataTrain(data_train):
    # Put the ordered training datasets in a list, and their labels in another list
    data = []
    labels = []
    for ele in data_train:
        data.append(ele[0])
        labels.append(ele[1])
    current = labels[0][0] + ' : ' + labels[1][0] + ' : ' + labels[2][0] + ' : ' + labels[3][0] + '\n\n'
    return data, labels, current

def TrainClassifier(data, labels, FLAT_CLASSIFIER):
    # Train a hierarchical classifier according to the recommended setting
    if FLAT_CLASSIFIER == 'svm':
        start_train = tm.time()
        tree = progressive_learning.learn_tree(data, labels, classifier=FLAT_CLASSIFIER,
                                               dimred=False, threshold=0.25)
        training_time = tm.time() - start_train
        return tree, training_time

    elif FLAT_CLASSIFIER == 'svm_occ':
        start_train = tm.time()
        tree = progressive_learning.learn_tree(data, labels, classifier=FLAT_CLASSIFIER,
                                               dimred=True, threshold=0.25)
        training_time = tm.time() - start_train
        return tree, training_time

    else:
        print("Usage: $python3 scHPL_liver_5datasets.py [RESOLUTION: 'high' or 'low'] [FLAT_CLASSIFIER: 'svm' or 'svm_occ']")


def SaveClassifier(tree, FEATURES, file_name):
    # Save the trained classifier
    with open(PATH_OUTPUT + FEATURES + '/tree_' + file_name + '.pkl', 'wb') as f:
        pickle.dump(tree, f)

    # Save the final tree
    with open(PATH_OUTPUT + FEATURES + '/tree_' + file_name + '.txt', 'w', encoding='utf-8') as f:
        for i in tree[0].ascii_art():
            f.writelines('\n'.join(i))

def EvaluateTree(y_true, y_pred, tree):
    # Make a confusion matrix and calculate evaluation metrics
    confmatrix = evaluate.confusion_matrix(y_true, y_pred)
    num_labels = pd.Series(y_pred).nunique()
    num_unclassified = confmatrix['root'].sum() if 'root' in confmatrix.columns else 0
    hf1 = evaluate.hierarchical_F1(y_true, y_pred, tree)
    ari = adjusted_rand_score(y_true, y_pred)
    ratio_mis = zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)  # fraction of misclassifications (float)
    return confmatrix, num_labels, num_unclassified, hf1, ari, ratio_mis

