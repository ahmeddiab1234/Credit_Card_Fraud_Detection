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

config = load_config()

RANDOM_STATE = config['random_state']
TRAIN_PATH = config['dataset']['train_path']
VAL_PATH = config['dataset']['val_path']
TRAIN_VAL_PATH = config['dataset']['train_val_path']
ZERO_CLASS_WEIGHT = config['dataset']['zero_weight']
ONE_CLASS_WEIGHT = config['dataset']['one_weight']
preprocess_values = config['preprocessing']


def try_kmeans(x_train, t_train):
   neg_samples = config['model']['knn']['params']['neg_samples']
   
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

   def apply_knn(self, x_train, t_train, x_val):
      n_neighbours = config['model']['knn']['params']['n_neighbours']
      knn = KNeighborsClassifier(n_neighbors=n_neighbours)
      model = knn.fit(x_train, t_train)
      start = time.time()
      t_pred = model.predict(x_val)
      time_pred = time.time()-start
      t_pred_prob = model.predict_proba(x_val)[:,1]
      return model, t_pred, t_pred_prob, time_pred
   
   def train(self):
      apply_pca = config['model']['knn']['params']['apply_pca']
      apply_kmeans = config['model']['knn']['params']['apply_kmeans']
      n_components = config['model']['knn']['params']['n_components']
      neg_samples = config['model']['knn']['params']['neg_samples']

      x_train,t_train, x_val= self.x_train, self.t_train, self.x_val
      
      pca=None
      if apply_pca:
         pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
         x_train = pca.fit_transform(x_train)
         x_val = pca.transform(x_val)
      elif apply_kmeans:
         x_train, t_train = try_kmeans(x_train, t_train,neg_samples=neg_samples)

      model, t_pred, t_pred_prob, time_pred = self.apply_knn(x_train, t_train, x_val)
   
      return model, pca, t_pred, t_pred_prob, time_pred

   def evaluation(self, t_pred, t_pred_prob,):
      best_rec = config['dataset']
      eval = Metrices(self.t_val, t_pred, t_pred_prob)

      report = eval.report()
      prc, rec, thr = eval.calc_pc()
      best_prc, best_rec, best_thr = eval.best_thresall(prc, rec, thr)
      return report, best_prc, best_rec, best_thr



