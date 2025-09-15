
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


config = load_config()

RANDOM_STATE = config['random_state']
TRAIN_PATH = config['dataset']['train_path']
VAL_PATH = config['dataset']['val_path']
TRAIN_VAL_PATH = config['dataset']['train_val_path']
ZERO_CLASS_WEIGHT = config['dataset']['zero_weight']
ONE_CLASS_WEIGHT = config['dataset']['one_weight']
preprocess_values = config['preprocessing']



class Prepare():
    def __init__(self, path=TRAIN_VAL_PATH):
        self.df = load_df(path)

    def prepare_data(self, preprocess_values=preprocess_values):
        preprocess = Processing_Pipeline()
        self.df_transformed = preprocess.apply_preprocessing(self.df)
        self.df, self.x, self.t = load_x_t(self.df_transformed)
        self.x_train, self.x_val, self.t_train, self.t_val = split_data(self.x, self.t, 0.2)

        self.x_train_scaled,self.t_train_scaled, self.x_val_scaled, self.t_val_scaled = \
                preprocess.apply_scaling(self.x_train, 
                self.t_train, self.x_val, self.t_val, True)


        if preprocess_values['data_choice']==1:
            if preprocess_values['is_scaling']:
                return self.x_train_scaled, self.t_train_scaled, self.x_val_scaled, self.t_val_scaled
            else:
                return self.x_train, self.t_train, self.x_val, self.t_val
            
        else:
            self.x_train_sampled, self.t_train_sampled = preprocess.apply_sampling(
                        self.x_train_scaled, self.t_train_scaled)
            
            self.x_val_sampled, self.t_val_sampled = preprocess.apply_sampling(
                        self.x_val_scaled, self.t_val_scaled)
            
            return self.x_train_sampled, self.t_train_sampled, self.x_val_sampled, self.t_val_sampled



class Train():
    def __init__(self, x_train, t_train, x_val, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train = t_train
        self.t_val = t_val

    def logistic_regression(self):
        model_name = 'logistic_regression'
        params = config['model'][model_name]['params']
        model = LogisticRegression(**params)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)[:,1]

        return model, t_pred, t_pred_prob

    def random_forest(self):
        model_name = 'random_forest_params'
        params = config['model'][model_name]['params']
        model = RandomForestClassifier(**params)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)[:,1]
        
        return model, t_pred, t_pred_prob

    def voting_classifier(self):
        model_name = 'voting_classifier'
        params1 = config['model'][model_name]['params']['model1']
        params2 = config['model'][model_name]['params']['model2']
        voting_type = config[model_name]['params']['voting']
        log_reg = LogisticRegression(**params1)
        ran_for = RandomForestClassifier(**params2)
        voting = VotingClassifier(
            estimators=[('lr',log_reg), ('ran', ran_for)],
            voting=voting_type
        )
        voting.fit(self.x_train, self.t_train)
        t_pred = voting.predict(self.x_val)
        return voting, t_pred, None
    
    def xgboost(self):
        model_name = 'xgboost'
        params = config['model'][model_name]['params']
        model = XGBClassifier(**params)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)[:,1]
        return model, t_pred, t_pred_prob
    
    def light_boast(self):
        model_name = 'light_boost'
        params = config['model'][model_name]['params']
        model = LGBMClassifier(**params)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)[:,1]
        return model, t_pred, t_pred_prob
    
    def cat_boast(self):
        model_name = 'cat_boost'
        params = config['model'][model_name]['params']
        model = CatBoostClassifier(**params)
        model.fit(self.x_train, self.t_train)
        t_pred = model.predict(self.x_val)
        t_pred_prob = model.predict_proba(self.x_val)[:,1]
        return model, t_pred, t_pred_prob


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
    
    def best_threshall(self, precision, recall, threshall):
        return Metrices.best_thresall(self, 
                precision, recall, threshall)
    
    def calc_pc_(self):
        return self.metrics.calc_pc()

    def visualize_pr_(slef, precision, recall, threshall, versus_threshall=True):
        visualize_pr(precision, recall, threshall, versus_threshall)


if __name__=='__main__':
    prep = Prepare()

    x_train, t_train, x_val, t_val = prep.prepare_data()

    train = Train(x_train, t_train, x_val, t_val)
    model_name = 'RandomForest'
    
    if model_name=='RandomForest':
        model, t_pred, t_pred_prob = train.random_forest()
    elif model_name=='LogisticRegression':
        model, t_pred, t_pred_prob = train.logistic_regression()
    if model_name=='VotingClassifier':
        model, t_pred, t_pred_prob = train.voting_classifier()
    elif model_name=='XgBoost':
        model, t_pred, t_pred_prob = train.xgboost()
    elif model_name=='LightBoost':
        model, t_pred, t_pred_prob = train.light_boast()
    elif model_name=='CatBoost':
        model, t_pred, t_pred_prob = train.cat_boast()

    eval = Eval(t_val, t_pred, t_pred_prob)
    print(eval.report_())

    precision,recall,threshold = eval.calc_pc_()
    b_thr, prc, rec = eval.best_threshall(precision, recall, threshold) 
    print(b_thr, prc, rec)

    # save_model(model, b_thr, 'Random_Forest')
    

