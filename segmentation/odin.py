import os 
import random 
import argparse 
import numpy as np 
import time 

from torch.utils import data 
from datasets import LostAndFound, FishyScapes, FishyScapesLF, RaodAnomaly
from utils import ext_transforms as et 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.visualizer import Visualizer

from baseline.src.model_utils import load_network

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pdb
from uncertainty.ap_metrics import APMetrics
import numpy.ma as ma

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_data_root", type = str, default='/mnt/kostas-graid/datasets/meghnag/data/LostFound')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size (default: 4)')

    parser.add_argument("--ood_dataset", type=str, default='lostandfound')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    return parser

def get_dataset(opts):
    print("getting data")
    val_transform = et.ExtCompose([
            # et.ExtResize(512,),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    if opts.ood_dataset == 'lostandfound':
        ood_dst = LostAndFound(split='test', root = opts.ood_data_root,transform=val_transform )
    elif opts.ood_dataset == 'fishyscapes':
        ood_dst = FishyScapes(root = opts.ood_data_root, transform= val_transform)
    elif opts.ood_dataset == 'fishyscapes_lf':
                    ood_dst = FishyScapesLF(root = opts.ood_data_root, transform= val_transform)
    elif opts.ood_dataset == 'roadanomaly':
                    ood_dst = RaodAnomaly(root = opts.ood_data_root, transform= val_transform)

    else: 
        print("incorrect baseline")
    return ood_dst



def main():
    opts = get_argparser().parse_args()
    T = 1000
    epsilon = 0.0038 
    # T = 3.0
    # epsilon = 0.0001
    softmax_ = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device) 
    model = load_network(model_name="DeepLabV3+_WideResNet38", num_classes=19,
                               ckpt_path=None, train = False, cosine_sim = False)
    
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"]) # model_state
    
    model.to(device)
    
    model.eval()
   
    testset = get_dataset(opts)
    test_loader = data.DataLoader(
            testset, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    print("data set length :", len(testset))

    y_true_list = []
    sfm_list = []
  
    ap_metrics = APMetrics(2)
    img_idx = 0
    time_avg = 0
    for (images, labels) in test_loader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        img_idx +=1
        images.requires_grad = True 

        preds,_ = model(images)

        start_time = time.time()

        # preds_T = preds/T
        preds_T = preds        
        pred_T_sfm = torch.log(softmax_(preds_T))

        loss = criterion(pred_T_sfm, labels)

        model.zero_grad()
        loss.backward()
        images_grad = images.grad.data 
        # images_grad =  (torch.ge(images.grad.data, 0))
        # images_grad = (images_grad.float() - 0.5) * 2
        
        # # images_grad[0][0] = (images_grad[0][0] )/(63.0/255.0)
        # # images_grad[0][1] = (images_grad[0][1] )/(62.1/255.0)
        # # images_grad[0][2] = (images_grad[0][2])/(66.7/255.0)
        
        sign_data_grad = (-images_grad).sign()

        # image_pert = torch.add(images.data, images_grad, alpha = -epsilon)

        image_pert = images - epsilon*sign_data_grad
        
        with torch.no_grad():
            new_preds,_ = model(image_pert)
        
        new_preds = new_preds/T

        outputs_sf = softmax_(new_preds)
        end_time = time.time()
        print(f'\t time elapsed: {(end_time - start_time)*1000}')
        time_avg+=(end_time - start_time)*1000
        if img_idx == 100:
            print("avg time :", time_avg/100)
            img_idx = 0
            time_avg = 0
        
        outputs_sf = outputs_sf.cpu().numpy()
        
        outputs = 1 - np.max(outputs_sf, axis = 1)
        
        # with torch.no_grad():
        #     preds = model(images)
        # outputs_sf = softmax_(preds).cpu().numpy()
        # outputs = 1- np.max(outputs_sf, axis = 1)
        targets = labels.cpu().numpy()
        for i in range(len(images)):

            output = outputs[i]
            target = targets[i]

            if opts.ood_dataset == 'lostandfound':
                    # mask = np.where(target == 2, 1,0).astype(bool)
                labels_true = np.where(target == 2, 1, 0)
            else:
                labels_true = target

            # labels_true = np.where(target == 2, 1, 0)
                
            mask_1 = np.where(target == 255, 0, 1).astype(bool) 

            op_masked = ma.masked_array(output, mask_1)
            sfm = op_masked.data[op_masked.mask]

            y_true = ma.masked_array(labels_true, mask_1)
            y_true = y_true.data[y_true.mask]

            if np.count_nonzero(y_true) == 0:
                continue
            y_true_list +=list(y_true)
            sfm_list+=list(sfm)
        
    ap_metrics.update(y_true_list, sfm_list)
    print(ap_metrics.get_results())
    fpr95, tpr = ap_metrics.compute_fpr(y_true_list, sfm_list)
    print("FPR-95 :", fpr95, tpr)

if __name__=='__main__':
    main()





