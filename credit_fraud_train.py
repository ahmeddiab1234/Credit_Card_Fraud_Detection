"""
1 - try data as it is 
2 - oversameple , undersample, over&under sample

try:
    - Logistic regression and tune hyperparamters
    - random forest classifier
    - voting classifier
    - xgboost, light boost, cat boost

pkl file
    - model
    - threshall
    - model name

"""

from utils.helper_fun import *
from credit_fraud_utils_data import Processing_Pipeline
from credit_fraud_utils_eval import Metrices
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from collections import Counter
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
TRAIN_PATH = 'data/split/train.csv'
VAL_PATH = 'data/split/val.csv'
TRAIN_VAL_PATH = 'data/split/trainval.csv'
# ZERO_CLASS_WEIGHT = 0.8 # 0.9, 1
ZERO_CLASS_WEIGHT = 1 # 0.9, 1
ONE_CLASS_WEIGHT = 1


class Prepare():
    def __init__(self):
        self.df = load_df(TRAIN_VAL_PATH)

    def prepare_data(self, data_choice=1, is_scaling=True, scaler_op=1, sample_op=1, ov_sample_factor=20, un_sample_factor=20, over_strategy='smote'):
        preprocess = Processing_Pipeline()
        self.df_transformed = preprocess.apply_preprocessing(self.df, True, False)
        self.df, self.x, self.t = load_x_t(self.df_transformed)
        self.x_train, self.x_val, self.t_train, self.t_val = split_data(self.x, self.t, 0.2)

        self.x_train_scaled,self.t_train_scaled, self.x_val_scaled, self.t_val_scaled = \
                preprocess.apply_scaling(self.x_train, 
                self.t_train, self.x_val, self.t_val, scaler_op)


        if data_choice==1:
            if is_scaling:
                return self.x_train_scaled, self.t_train_scaled, self.x_val_scaled, self.t_val_scaled
            else:
                return self.x_train, self.t_train, self.x_val, self.t_val
            
        else:
            self.x_train_sampled, self.t_train_sampled = preprocess.apply_sampling(
                        self.x_train_scaled, self.t_train_scaled, sample_op, un_sample_factor, ov_sample_factor, over_strategy)
            
            self.x_val_sampled, self.t_val_sampled = preprocess.apply_sampling(
                        self.x_val_scaled, self.t_val_scaled, sample_op, un_sample_factor, ov_sample_factor, over_strategy)
            
            return self.x_train_sampled, self.t_train_sampled, self.x_val_sampled, self.t_val_sampled



class Train():
    def __init__(self, x_train, t_train, x_val, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train = t_train
        self.t_val = t_val

    def logistic_regression(self, solver='sag', fit_intercept=True, max_iter=10000):
        model = LogisticRegression(solver=solver, fit_intercept=fit_intercept, max_iter=max_iter, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)

        return t_pred, t_pred_prob

    def random_forest(self, max_depth=5, n_estimators=25):
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)
        
        return t_pred, t_pred_prob

    def voting_classifier(self, solver='sag', fit_intercept=True, max_iter=10000, max_depth=5,n_estimators=25):
        log_reg = LogisticRegression(solver='sag', fit_intercept=True, max_iter=10000, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        ran_for = RandomForestClassifier(max_depth=5, n_estimators=25, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        voting = VotingClassifier(
            estimators=[('lr',log_reg), ('ran', ran_for)],
            voting='hard'
        )
        voting.fit(self.x_train, self.t_train)
        t_pred = voting.predict(self.x_val)
        # t_pred_prob = voting.predict_proba(self.x_val)
        return t_pred, None
    
    def xgboost(self, max_depth=5, n_estimators=100, lr=0.01):
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, 
                random_state=RANDOM_STATE, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)
        return t_pred, t_pred_prob
    
    def light_boast(self, n_estimators=100, lr=0.1, max_depth=-1):
        model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                learning_rate=lr, random_state=RANDOM_STATE, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)
        return t_pred, t_pred_prob
    
    def cat_boast(self, iterations=100, depth=5, lr=0.1):
        model = CatBoostClassifier(iterations=iterations, depth=depth, 
                learning_rate=lr, random_state=RANDOM_STATE, class_weight={0:ZERO_CLASS_WEIGHT, 1:ONE_CLASS_WEIGHT})
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)
        return t_pred, t_pred_prob


class Eval():
    def __init__(self, t_val, t_pred, t_pred_prop=None):
        self.t_val=t_val
        self.t_pred=t_pred
        self.t_pred_prop = t_pred_prop
        self.metrics = Metrices(self.t_val, self.t_pred, self.t_pred_prop)

    def report_(self):
        return self.metrics.report()
    
    def eval_metrices_(self):
        return self.metrics.clac_eval_metrices()
    
    def best_threshall(self, precision, recall, threshall, target='precision', target_pr=0.9, target_re=0.9):
        return Metrices.best_thresall(self, 
                precision, recall, threshall, target=target, target_precision=target_pr, target_recall=target_re)
    
    def calc_pc_(self):
        return self.metrics.calc_pc()

    def visualize_pr_(slef, precision, recall, threshall, versus_threshall=True):
        visualize_pr(precision, recall, threshall, versus_threshall)


if __name__=='__main__':
    prep = Prepare()
    x_train, t_train, x_val, t_val = prep.prepare_data(1, True, 1)
    # print(x_train.shape) # (17648, 30)
    # print(t_train.shape) # (17648,)
    # print(x_val.shape) # (4412, 30)
    # print(t_val.shape) # (4412,)
    # print(Counter(t_train))
    # print(Counter(t_val))

    # cnt = 2
    
    for depth in [3,5,10]:
        # for iteration: 
            train = Train(x_train, t_train, x_val, t_val)
            t_pred, t_pred_prob = train.cat_boast(depth=depth)

            eval = Eval(t_val, t_pred, t_pred_prob)
            print(f'depth {depth}')
            print(eval.report_())
            # print(eval.eval_metrices_())

    pass



"""
data as it is :
    logistic regression:
        solver: sag, fit-intercept=True, max-iterations=10000
        f1-score=88%

    random forest:
        max-depth,n-estimators : {[9,100],[9,50], [9,20], [6,50], [6,20], [5,100]}
        f1-score=88%

    xgboost:
        max-depth: 3, lr=0.2, n-estimator:100
        f1-score=77%
    
    light-boost:
        n-estimator:500, lr=0.05, max-depth:3
        f1-score: 79%

    cat-boost:
        depth: 

"""
