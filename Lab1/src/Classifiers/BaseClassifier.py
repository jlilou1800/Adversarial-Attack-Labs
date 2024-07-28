import io
from math import sqrt

from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score, \
    confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch


class BaseClassifier:
    def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.classifier = None
        self.metric_choose = '*'
        self.metric_choose = metric_choose
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test


    def k_fold_cross_validation(self, n, best_parameters, classifierAlgo):
        print("parameters : ", best_parameters)
        for metric in best_parameters:
            metrics_y_pred = []
            metrics_y_score = []
            self.train(best_parameters[metric]["parameters"], classifierAlgo)
            results_true, results_pred, results_score = self.predict()
            metrics_y_true = results_true
            metrics_y_pred = metrics_y_pred + list(results_pred)
            metrics_y_score = metrics_y_score + list(results_score)

    def train(self, parameters, classifierAlgo):
        self.classifier = classifierAlgo()
        self.classifier.set_params(**parameters)
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self, X=None, Y=None):
        if X is None or Y is None:
            x_test = self.x_test
            y_test = self.y_test
        else:
            x_test = X
            y_test = Y
        y_pred = self.classifier.predict(x_test)
        y_score = self.classifier.predict_proba(x_test)

        return y_test, y_pred, y_score[:, 1]

    def predict2(self, x_test):
        y_test = self.classifier.predict(x_test)
        y_score = self.classifier.predict_proba(x_test)
        return y_test, y_score[:, 1]

    def predict_proba(self, x):
        # x = [x]
        p = self.classifier.predict_proba(x)
        return p

    def score(self, x, y):
        p = self.classifier.score(x, y)
        return p

    def evaluationMetrics(self, y_true, y_pred):
        return {"accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f_measure": f1_score(y_true, y_pred, average='weighted')}


    def confusion_matrix(self, y_test, y_pred):
        nbr_of_char = 26
        n = len(y_test)
        # initialization of confusion matrix
        mat = list()
        for i in range(nbr_of_char):
            tmp = list()
            for j in range(nbr_of_char):
                tmp.append(0)
            mat.append(tmp)
        # computation of confusion matrix
        for i in range(n):
            mat[y_pred[i]-97][y_test[i]-97] += 1
        # return mat
        return None

    def compute_counters(self, matrix):
        nbr_of_char = 26
        self.reset_counters()
        for i in range(nbr_of_char):
            self.TP += matrix[i][i]
        for i in range(nbr_of_char):
            self.FP += sum(matrix[i]) - matrix[i][i]
        tmp = list(map(list, zip(*matrix)))
        for i in range(nbr_of_char):
            self.FN += sum(tmp[i]) - tmp[i][i]

    def reset_counters(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
