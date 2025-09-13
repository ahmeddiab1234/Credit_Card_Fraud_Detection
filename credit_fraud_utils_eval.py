import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, roc_auc_score, classification_report
from utils.helper_fun import visualize_pr, load_df, load_x_t, split_data
from sklearn.linear_model import LogisticRegression

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
    
    def best_thresall(self, precision, recall, threshall, target='precision', target_precision=0.9, target_recall=0.9):
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
    


if __name__=='__main__':
    df= load_df('data/split/train.csv')
    df,x,t=load_x_t(df)
    x_train, x_val, t_train, t_val = split_data(x,t)

    model = LogisticRegression(max_iter=10000,random_state=42)
    model.fit(x_train, t_train)
    t_pred = model.predict(x_val)
    t_pred_prob = model.predict_proba(x_val)[:, 1]
    metrc = Metrices(t_val, t_pred, t_pred_prob)
    pr,re,thr = metrc.calc_pc()
    metrc.visualize_pr_(pr,re,thr)


