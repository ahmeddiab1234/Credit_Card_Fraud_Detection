

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from credit_fraud_utils_data import Processing_Pipeline 
from utils.helper_fun import *
from credit_fraud_utils_eval import Metrices
import pandas
import numpy
import time

config = load_config()

if __name__=='__main__':
    test_df = load_df(config['dataset']['test_path'])
    preprocess = Processing_Pipeline()

    test_df_preprocessed = preprocess.apply_preprocessing(test_df, 
        remove_dublicate=config['preprocessing']['remove_dublicates'],
        remove_outlier=config['preprocessing']['remove_outlier'],
        change_time=config['preprocessing']['change_time']
        )
    
    test_df_preprocessed, x,t = load_x_t(test_df_preprocessed)

    x_test,t_test = preprocess.apply_scaling(x, t,None, None,
        config['preprocessing']['scaler_option'], False)

    (model,pca), threshold, model_name = load_model_knn()
    if pca is not None:
        x_test = pca.transform(x_test)
    start = time.time()
    t_pred = model.predict(x_test)
    seconds = time.time()-start
    t_pred_prob = model.predict_proba(x_test)[:,1]

    eval = Metrices(t_test, t_pred, t_pred_prob)
    report = eval.report()
    print(report)
    print(seconds)

    prc,rec,thr = eval.calc_pc()
    # visualize_pr(prc, rec, thr)

