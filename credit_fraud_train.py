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

RANDOM_STATE = 42
TRAIN_PATH = 'data/split/train.csv'
VAL_PATH = 'data/split/val.csv'
TRAIN_VAL_PATH = 'data/split/trainval.csv'


class Prepare():
    def __init__(self):
        self.df = load_df(TRAIN_VAL_PATH)
        self.x, self.t = load_x_t(self.df)
        self.x_train, self.x_val, self.t_train, self.t_val = split_data(self.x, self.t, 0.2)

    def prepare_data(self, data_choice=1, is_scaling=True, scaler_op=1, sample_op=1, ov_sample_factor=20, un_sample_factor=20, over_strategy='smote'):
        self.x_train_preprocessed = Processing_Pipeline.apply_preprocessing(self, self.x_train)
        self.x_val_preprocessed = Processing_Pipeline.apply_preprocessing(self, self.x_val)

        self.x_train_scaled,self.t_train_scaled, self.x_val_scaled, self.t_val_scaled = \
                Processing_Pipeline.apply_scaling(self, self.x_train_preprocessed, 
                                                self.t_train, self.x_val_preprocessed, self.t_val, scaler_op)


        if data_choice==1:
            if is_scaling:
                return self.x_train_scaled, self.t_train_scaled, self.x_val_scaled, self.t_val_scaled
            else:
                return self.x_train, self.t_train, self.x_val, self.t_val
            
        else:
            self.x_train_sampled, self.t_train_sampled = Processing_Pipeline.apply_sampling(self, 
                        self.x_train_scaled, self.t_train_scaled, sample_op, un_sample_factor, ov_sample_factor, over_strategy)
            
            self.x_val_sampled, self.t_val_sampled = Processing_Pipeline.apply_sampling(self, 
                        self.x_val_scaled, self.t_val_scaled, sample_op, un_sample_factor, ov_sample_factor, over_strategy)
            
            return self.x_train_sampled, self.t_train_sampled, self.x_val_sampled, self.t_val_sampled



class Train():
    def __init__(self, x_train, t_train, x_val, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train = t_train
        self.t_val = t_val

    def logistic_regression(self, solver='sag', fit_intercept=True, max_iter=10000):
        model = LogisticRegression(solver=solver, fit_intercept=fit_intercept, max_iter=max_iter)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)

        return t_pred, t_pred_prob

    def random_forest(self, max_depth=5, n_estimators=50):
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)
        
        return model, t_pred, t_pred_prob

    def voting_classifier(self, ):
        log_reg = LogisticRegression(solver='sag', fit_intercept=True, max_iter=10000)
        ran_for = RandomForestClassifier(max_depth=5, n_estimators=50)
        voting = VotingClassifier(
            estimators=[('lr',log_reg), ('ran', ran_for)],
            voting='hard'
        )
        voting.fit(self.x_train, self.t_train)
        t_pred = voting.predict(self.x_val)
        t_pred_prob = voting.predict_proba(self.x_val)
        
        return t_pred, t_pred_prob
    
    def xgboost(self, ):
        pass

    def light_boast(self, ):
        pass

    def cat_boast(self, ):
        pass

    def eval():
        pass



class eval():
    def __init__(self, t_val, t_pred, t_pred_prop):
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

    pass


