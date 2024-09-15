import numpy as np 
import torch 

from scipy.spatial import distance 
from sklearn.metrics import jaccard_score
from tabulate import tabulate
import sys 
import os 
from sklearn.preprocessing import normalize

import pdb 

def compute_excl(feats, preds, targets, in_dist_classes):

    num_classes = len(in_dist_classes)
    mask = np.array(preds) == np.array(targets)   
    feats = np.array(feats) 
    feats_list = []
    encoding_list = [] 
    feats_norm_list = [] 
    excl_mat = np.zeros((num_classes,num_classes))
    print(len(np.unique(targets)), num_classes)
    for i in range(0, num_classes):
        label_mask = np.where(np.array(targets) == i, 1, 0)
        full_mask = mask*label_mask
        # full_mask = label_mask
        full_mask_idx = np.array(full_mask.astype(bool).nonzero()[0]).astype(int)  
        feats_curr = feats[full_mask_idx]
        
        if len(feats_curr) == 0: 
            print(i)
            # continue
            feats_mean = []
            encoding = []
            feats_norm = np.zeros((1,512))
        else: 
          
            feats_mean = np.mean(feats_curr, axis = 0)
            encoding = np.where(feats_mean > 0, 1, 0)  ## set this to 0.05 or 1 only sc cifar 10n, that has unncecearity small activations 
            # feats_mean = np.where(feats_mean > 1,feats_mean,0)
            feats_norm = normalize(np.expand_dims(feats_mean, axis = 0), axis = 1)
        
        feats_list.append(feats_mean)
        encoding_list.append(encoding)
        feats_norm_list.append(feats_norm)
   
    feats_norm_array = np.asarray(feats_norm_list)
    feats_norm_array = feats_norm_array[:,0,:]
    cosine_mat = 1- np.matmul(feats_norm_array,np.transpose(feats_norm_array))
    print("MEAN ID activations :", np.mean(np.linalg.norm(feats, axis = 1)))
    mean_act = np.mean(np.linalg.norm(feats, axis = 1))
    for i in range(0, num_classes-1):
        encoding_1 = encoding_list[i]
        if len(encoding_1) == 0:
            excl_mat[i,:] = 0
            continue
        for j in range(i+1, num_classes):
            encoding_2 = encoding_list[j]
            if len(encoding_2) == 0:
                continue
            metric_TT = np.sum(np.logical_and(encoding_1, encoding_2)*1)
            metric_TF = np.sum(np.logical_xor(encoding_1, encoding_2)*1)
            metric_FF = np.sum(np.logical_or(encoding_1, encoding_2)*1)            
           
            exclsive_metric = metric_TF/len(encoding_1)
            # exclsive_metric = metric_TF/(metric_TF + metric_TT)

            excl_mat[i,j] = exclsive_metric
            excl_mat[j,i] = exclsive_metric
  
    return excl_mat, cosine_mat, mean_act

    

            
            

