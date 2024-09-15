import numpy as np 
from sklearn.metrics import average_precision_score, auc, roc_curve, f1_score, roc_auc_score

import pdb

class APMetrics():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        
        self.data_len = 0 
        self.prev_ap = 0


    def update(self, labels_true, pred_scores):
        #batch_size = len(labels_true)
        self.curr_ap = 0
        self.curr_auc = 0
        
        self.curr_ap = average_precision_score(labels_true, pred_scores)
        self.curr_auc = roc_auc_score(labels_true, pred_scores)
        
        # self.prev_ap = (self.prev_ap*self.data_len + self.curr_ap)/(self.data_len + 1)
        # self.prev_auc = (self.prev_auc*self.data_len + self.curr_auc)/(self.data_len + 1)
        # self.data_len +=1

    


    def get_results(self):
        return self.curr_ap, self.curr_auc

    def compute_fpr(self, labels_true, pred_scores):
        fpr, tpr, th = roc_curve(labels_true, pred_scores)
        idx = (np.abs(tpr - 0.95)).argmin()
        fpr95 = fpr[idx]

        return fpr95, tpr[idx]


    def reset(self):
        self.data_len = 0
        self.prev_ap = 0

