import numpy as np 
import torch 

from scipy.spatial import distance 
from sklearn.metrics import jaccard_score
from tabulate import tabulate
import sys 
import os 
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import pdb 

def compute_excl_ood(feats,targets_ood, targets_id,feats_id, in_dist_classes, model):

    num_classes = len(in_dist_classes)
    feats_ood = np.array(feats) 
    feats_list_id = []
    encoding_list_id = [] 
    feats_norm_list_id = [] 
    feats_list_ood = []
    encoding_list_ood = [] 
    feats_norm_list_ood = [] 
    weights = model.classifier[len(model.classifier)-1].weight.detach().cpu().numpy()
    weights_norm = weights/np.expand_dims(np.linalg.norm(weights, axis= 1), axis=1)
    cosine_mat = np.zeros((num_classes,num_classes))
    excl_mat = np.zeros((num_classes,num_classes))
    print(np.unique(targets_id), num_classes)
    for i in range(0, num_classes):
        label_mask = np.where(np.array(targets_id) == i, 1, 0)
        full_mask = label_mask
      
        # full_mask = mask
        full_mask_idx = np.array(full_mask.astype(bool).nonzero()[0]).astype(int)  
        feats_curr = np.array(feats_id)[full_mask_idx]
        
       
        feats_mean = np.mean(feats_curr, axis = 0)
        encoding = np.where(feats_mean > 0, 1, 0)  
        feats_norm = normalize(np.expand_dims(feats_mean, axis = 0), axis = 1)
        
        feats_list_id.append(feats_mean)
        encoding_list_id.append(encoding)
        feats_norm_list_id.append(feats_norm)
   
    feats_norm_array_id = np.asarray(feats_norm_list_id)
    feats_norm_array_id = feats_norm_array_id[:,0,:]
    # pdb.set_trace()
    print("MEAN ID activations :", np.mean(np.linalg.norm(feats_id, axis = 1)))

    for i in range(0, num_classes):
            label_mask = np.where(np.array(targets_ood) == i, 1, 0)
            full_mask = label_mask
        
            # full_mask = mask
            full_mask_idx = np.array(full_mask.astype(bool).nonzero()[0]).astype(int)  
            feats_curr = feats_ood[full_mask_idx]
            
            # cosine_ =  np.mean(np.matmul(normalize(feats_curr),np.swapaxes(weights_norm, 0, 1)), axis = 0)
            feats_mean = np.mean(feats_curr, axis = 0)
            encoding = np.where(feats_mean > 0, 1, 0)  
            feats_norm = normalize(np.expand_dims(feats_mean, axis = 0), axis = 1)
            cosine_ =  np.mean(np.matmul(np.transpose(np.expand_dims(encoding, axis=1)),np.swapaxes(weights_norm, 0, 1)), axis = 0)
            cosine_mat[i,:] = cosine_
            
            feats_list_ood.append(feats_mean)
            encoding_list_ood.append(encoding)
            feats_norm_list_ood.append(feats_norm)
    
    feats_norm_array_ood = np.asarray(encoding_list_ood)
    # feats_norm_array_ood = feats_norm_array_ood[:,0,:]
    print("MEAN OOD activations :", np.mean(np.linalg.norm(feats_ood, axis = 1)))
    weights_norm = weights/np.expand_dims(np.linalg.norm(weights, axis= 1), axis=1)
    # feats_norm = feats_norm_array_ood/np.expand_dims(np.linalg.norm(feats_norm_array_ood, axis = 1),axis=1)
    # cosine_mat = np.matmul(feats_norm_array_id,np.transpose(feats_norm_array_ood))
    cosine_mat = np.matmul(feats_norm_array_ood,np.swapaxes(weights_norm, 0, 1))
    print(np.mean(cosine_mat), np.std(cosine_mat))

    # cosine_mat = (cosine_similarity(feats_norm_array_ood, weights))
    for i in range(0,num_classes):
        encoding_1 = encoding_list_id[i] 
      
        # if len(encoding_1) == 0:
        #     excl_mat[i,:] = 0
        #     continue
        for j in range(0,num_classes ):
            encoding_2 = encoding_list_ood[j]
            # if len(encoding_2) == 0:
            #     continue
            metric_TT = np.sum(np.logical_and(encoding_1, encoding_2)*1)
            metric_TF = np.sum(np.logical_xor(encoding_1, encoding_2)*1)
            metric_FF = np.sum(np.logical_or(encoding_1, encoding_2)*1)            
           
            exclsive_metric = metric_TF/len(encoding_1)
            # exclsive_metric = (metric_TF/(metric_TF + metric_TT))

            excl_mat[i,j] = exclsive_metric
          
  
    return excl_mat, cosine_mat

    

            