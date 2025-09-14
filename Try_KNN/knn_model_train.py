import time
import os
import sys
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helper_fun import *
from credit_fraud_utils_eval import Metrices
from credit_fraud_train import Prepare

RANDOM_STATE = 42
TRAIN_PATH = 'data/split/train.csv'
VAL_PATH = 'data/split/val.csv'
TRAIN_VAL_PATH = 'data/split/trainval.csv'
ZERO_CLASS_WEIGHT = 1 # 0.9, 1
ONE_CLASS_WEIGHT = 1



class KNNClassifier():
    def __init__(self, x_train, t_train, x_val, t_val):
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val

    def try_kmeans(self, neg_samples=400):
        x_train_pos = self.x_train[self.t_train==1]
        t_train_pos = self.t_train[self.t_train==1]
        x_train_neg = self.x_train[self.t_train==0]
        t_train_neg = self.t_train[self.t_train==0]

        k_means_classifer = KMeans(n_clusters=neg_samples)
        k_means_classifer.fit(x_train_neg)
        x_train_neg = k_means_classifer.cluster_centers_
        t_train_neg = np.zeros(len(k_means_classifer.cluster_centers_))

        x_train = np.vstack((x_train_pos, x_train_neg))
        t_train = np.concatenate((t_train_pos, t_train_neg))

        return x_train, t_train

    def try_pca(self, n_components=10):
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        x_train = pca.fit_transform(self.x_train)
        x_val = pca.transform(self.x_val)
        return x_train, x_val

    def apply_knn(self, n_neighbours=100):
        knn = KNeighborsClassifier(n_neighbors=n_neighbours)
        model = knn.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(x_val)
        return model, t_pred, t_pred_prob

    
    def train(self, n_neighbours=100, apply_pca=True, apply_kmeans=False, n_components=10, neg_samples=400):
        x_train,t_train = self.x_train, self.t_train
        if apply_pca:
            x_train, t_train = self.try_pca(n_components=n_components)
        elif apply_kmeans:
            x_train, t_train = self.try_kmeans(neg_samples=neg_samples)

        model, t_pred, t_pred_prob = self.apply_knn(n_neighbours=n_neighbours)
        return model, t_pred, t_pred_prob

        
    def evaluation(self, t_pred, t_pred_prob, best_rec=0.9):
        eval = Metrices(self.t_val, t_pred, t_pred_prob)

        report = eval.report()
        prc, rec, thr = eval.calc_pc()
        best_prc, best_rec, best_thr = eval.best_thresall(prc, rec, thr)
        return report, best_prc, best_rec, best_thr


if __name__ == '__main__':
    prep = Prepare(TRAIN_PATH)
    x_train, t_train, x_val, t_val = prep.prepare_data(2, True, 1, 2, 80, 80, 'smote')

    knn = KNNClassifier(x_train, t_train, x_val, t_val)

    model, t_pred, t_pred_prob = knn.train()
    report, prc, rec, thr = knn.evaluation(t_pred, t_pred_prob)
    print(report)
    print(prc, rec, thr)

