
from credit_fraud_utils_data import Processing_Pipeline 
from utils.helper_fun import *
from credit_fraud_utils_eval import Metrices
import os
import pandas
import numpy

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
        
    model, threshold, model_name = load_model()
    t_pred = model.predict(x_test)
    t_pred_prob = model.predict_proba(x_test)[:,1]

    eval = Metrices(t_test, t_pred, t_pred_prob)
    report = eval.report()
    print(report)

    prc,rec,thr = eval.calc_pc()
    # visualize_pr(prc, rec, thr)


