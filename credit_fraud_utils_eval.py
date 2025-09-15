import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, roc_auc_score, classification_report
from utils.helper_fun import visualize_pr, load_df, load_x_t, split_data
from sklearn.linear_model import LogisticRegression
from utils.helper_fun import load_config


config = load_config()

class Metrices:
    def __init__(self, t_gt, t_pr, t_pr_prob):
        self.t_gt=t_gt
        self.t_pr = t_pr
        self.t_pr_prob = t_pr_prob

    def report(self):
        report = classification_report(self.t_gt, self.t_pr)
        return report
    
    def calc_pc(self):
        precision, recall, threshall = precision_recall_curve(self.t_gt, self.t_pr_prob)
        precision, recall = precision[:-1], recall[:-1]
        return precision, recall, threshall

    def clac_eval_metrices(self):
        f1score = f1_score(self.t_gt, self.t_pr)
        
        roc_auc = roc_auc_score(self.t_gt, self.t_pr_prob)
        accuracy = accuracy_score(self.t_gt, self.t_pr)

        return f1score, roc_auc, accuracy
    
    def best_thresall(self, precision, recall, threshall):
        target = config['dataset']['eval_target']
        target_precision = config['dataset']['target_prc']
        target_recall = config['dataset']['target_rec'] 
        if target=='precision':
            best_thres_idx = np.argmax(precision>=target_precision)
            best_thres = threshall[best_thres_idx]
            return best_thres, target_precision, recall[best_thres_idx]

        else:
            best_thres_idx = np.argmin(recall>=target_recall)
            best_thres = threshall[best_thres_idx]
            return best_thres, precision[best_thres_idx], target_recall
        
    def visualize_pr_(slef, precision, recall, threshall, versus_threshall=True):
        visualize_pr(precision, recall, threshall, versus_threshall)
    

