"""
load test data
apply prparing data
load pickle file 
test the data 
"""

from credit_fraud_utils_data import Processing_Pipeline 
from utils.helper_fun import *
from credit_fraud_utils_eval import Metrices
import os
import pandas
import numpy

if __name__=='__main__':
    test_df = load_df('data/split/val.csv')
    preprocess = Processing_Pipeline()
    test_df_preprocessed = preprocess.apply_preprocessing(test_df, True, False)
    test_df_preprocessed, x,t = load_x_t(test_df_preprocessed)

    x_test,t_test = preprocess.apply_scaling(x, t, None, None, 1, False)
    
    model, threshold, model_name = load_model_val()
    t_pred = model.predict(x_test)
    t_pred_prob = model.predict_proba(x_test)[:,1]

    eval = Metrices(t_test, t_pred, t_pred_prob)
    report = eval.report()
    print(report)

    prc,rec,thr = eval.calc_pc()
    visualize_pr(prc, rec, thr)


