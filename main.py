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
import time as tm
from scHPL import train, predict, update, progressive_learning, utils, evaluate
import seaborn as sns
import matplotlib.pyplot as plt
from scHPL_liver.Python.scHPL_liver_5datasets import ReadDatasets, SortbyResolution, UniqueLabels, ParseDataTrain, TrainClassifier, SaveClassifier, EvaluateTree


PATH_INPUT = 'integrated_data/5/'
PATH_OUTPUT = 'output/5/'
FEATURES = ['2000', '5000'] # number of columns (features, genes) in the integrated datasets
FLAT_CLASSIFIER = ['svm', 'svm_occ'] # linear SVM, or One-class SVM

def main(argv):
    RESOLUTION = argv[0]
    FLAT_CLASSIFIER = argv[1]

    for i in FEATURES:
        datasets = ReadDatasets(i)
        data_train, data_test, y_true = SortbyResolution(datasets, RESOLUTION)
        file_name = RESOLUTION + '_' + i + '_' + FLAT_CLASSIFIER

        # Train a hierarchical progressive classifier & Save the classifier
        data, labels, current = ParseDataTrain(data_train)
        print('====================================\n', file_name, '\n', current)
        tree, training_time = TrainClassifier(data, labels, FLAT_CLASSIFIER)
        SaveClassifier(tree, i, file_name)
        print('\n', file_name, 'training time:', training_time)

        # Predict labels
        start_pred = tm.time()
        y_pred = predict.predict_labels(data_test, tree)
        pred_time = tm.time()-start_pred
        print('\n', 'Time to make predictions with scHPL:', pred_time, '\n')

        # Evaluate & Save the results
        confmatrix, num_labels, num_unclassified, hf1, ari, ratio_mis = EvaluateTree(y_true, y_pred, tree)
        pd.DataFrame(confmatrix).to_csv(PATH_OUTPUT + i + '/conf_' + file_name + '.csv', sep=',')
        evaluation = pd.DataFrame(
            [[i, FLAT_CLASSIFIER, RESOLUTION, num_labels, num_unclassified, hf1, ari, ratio_mis, training_time, pred_time]],
            columns=['n_feature', 'classifier', 'resolution', 'num_labels', 'num_unclassified',
                     'Hierarchical_F1_score', 'Adjusted_Rand_Index', 'ratio_misclassification',
                     'training_time', 'prediction_time'])
        evaluation.to_csv(PATH_OUTPUT + i + '/eval_' + file_name + '.csv', sep=',')
        fig = plt.figure(figsize=(12,12), dpi=300)
        sns.heatmap(round(confmatrix,2), vmin = 0, vmax = 1, annot=True)
        fig.savefig(PATH_OUTPUT + i + '/eval_' + file_name + '.png', bbox_inches='tight', dpi=fig.dpi)

if __name__ == "__main__":
    main(sys.argv[1:])
