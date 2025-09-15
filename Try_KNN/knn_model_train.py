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



def try_kmeans(x_train, t_train, neg_samples=400):
   x_train_pos = x_train[t_train==1]
   t_train_pos = t_train[t_train==1]
   x_train_neg = x_train[t_train==0]
   t_train_neg = t_train[t_train==0]

   k_means_classifer = KMeans(n_clusters=neg_samples, random_state=RANDOM_STATE)
   k_means_classifer.fit(x_train_neg)
   x_train_neg = k_means_classifer.cluster_centers_
   t_train_neg = np.zeros(len(k_means_classifer.cluster_centers_))

   x_train = np.vstack((x_train_pos, x_train_neg))
   t_train = np.concatenate((t_train_pos, t_train_neg))

   return x_train, t_train

class KNNClassifier():
   def __init__(self, x_train, t_train, x_val, t_val):
      self.x_train = x_train
      self.t_train = t_train
      self.x_val = x_val
      self.t_val = t_val

   def apply_knn(self, x_train, t_train, x_val, n_neighbours=100):
      knn = KNeighborsClassifier(n_neighbors=n_neighbours)
      model = knn.fit(x_train, t_train)
      start = time.time()
      t_pred = model.predict(x_val)
      time_pred = time.time()-start
      t_pred_prob = model.predict_proba(x_val)[:,1]
      return model, t_pred, t_pred_prob, time_pred
   
   def train(self, n_neighbours=100, apply_pca=False, apply_kmeans=False, n_components=10, neg_samples=400):
      x_train,t_train, x_val= self.x_train, self.t_train, self.x_val
      
      pca=None
      if apply_pca:
         pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
         x_train = pca.fit_transform(x_train)
         x_val = pca.transform(x_val)
      elif apply_kmeans:
         x_train, t_train = try_kmeans(x_train, t_train,neg_samples=neg_samples)

      model, t_pred, t_pred_prob, time_pred = self.apply_knn(x_train, t_train, x_val, n_neighbours=n_neighbours)
   
      return model, pca, t_pred, t_pred_prob, time_pred

   def evaluation(self, t_pred, t_pred_prob, best_rec=0.9):
      eval = Metrices(self.t_val, t_pred, t_pred_prob)

      report = eval.report()
      prc, rec, thr = eval.calc_pc()
      best_prc, best_rec, best_thr = eval.best_thresall(prc, rec, thr)
      return report, best_prc, best_rec, best_thr


if __name__ == '__main__':

   prep = Prepare(TRAIN_VAL_PATH)
   x_train, t_train, x_val, t_val = prep.prepare_data(2, True, 1, 2, 80, 80, 'smote')

   knn = KNNClassifier(x_train, t_train, x_val, t_val)
   model, pca, t_pred, t_pred_prob, time_pred = knn.train(n_neighbours=60, apply_pca=False, n_components=4)
   report, prc, rec, thr = knn.evaluation(t_pred, t_pred_prob)

   print(report)
   print(f'Time Taken for Preiction {time_pred}')
   print(prc, rec, thr)
#  save_model_knn((model,pca), thr, 'KNN_test')

