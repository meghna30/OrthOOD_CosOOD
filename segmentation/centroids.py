import numpy as np  
import torch 

import sys 
import os 


def compute_centroids(pan_feats, preds, targets):

    num_classes = 19
    feats = np.moveaxis(pan_feats, 1,3)
    mask = preds == targets    
    feats_list = []
    encoding_list = [] 
    nums = np.ones(19) 
    for i in range(0, num_classes):
        label_mask = np.where(targets == i, 1, 0)
        
        full_mask = mask*label_mask
        full_mask_idx = full_mask.astype(bool)
        feats_curr = feats[full_mask_idx]
        if len(feats_curr) == 0: 
            feats_mean = []
            nums[i] = 0
           
        else: 
            feats_mean = np.mean(feats_curr, axis = 0)
        
        feats_list.append(feats_mean)
    return feats_list, nums