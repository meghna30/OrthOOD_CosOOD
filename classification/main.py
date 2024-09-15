import torch
import torch.nn as nn
import numpy as np
import time
import pdb
import data_loader
from train import Trainer
import random

from arguments import get_args, print_args
import run_utils, data_utils
import json
import argparse
import os
import matplotlib.pyplot as plt 

load_args = False


if load_args:
    print("Loading arguments")
    parser =argparse.ArgumentParser()
    args = parser.parse_args()
    args = get_args()
    with open(args.save_path+'/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    args.batch_size = 1
    args.train = False
    args.load_checkpoint = True
    print_args(args)
     ## this is to make sure you set the arguement for OOD detection

else:
    args = get_args()
    print_args(args)
    if args.train:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        with open(args.save_path+'/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)



data_path = '../../shels/data' ## this might vary for you

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()


torch.manual_seed(args.random_seed)

np.random.seed(args.random_seed)
random.seed(args.random_seed)


if args.experiment == "ood_detect":
    print(args.multiple_dataset)

    file_path = args.save_path+'/outfile.txt'
    f = open(file_path, 'w')

    print("OOD detection across datasets")
    output_dim = args.total_tasks ## this if for dataet1 -ID dataset
    trainloader,valloader, testloader,classes,oodloader = data_utils.mutliple_dataset_loader(data_path, args.dataset1, args.dataset2, args.batch_size)
    print("In dist clsasses", classes)
    # classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    # print("OOD dist classes", classes_ood)
    # classes_idx_OOD = np.arange(0,len(classes_ood))
    classes_idx_ID = np.arange(0,len(classes))
    trainer = Trainer(output_dim,device, args)

    run_utils.run(trainer, args, trainloader, valloader, testloader,oodloader, [], [],classes_idx_ID,0)
    if args.load_checkpoint:
        test_acc, ood_ap, ood_auc, ood_fpr95, excl_mat, cosine_mat,_,_ = run_utils.do_ood_eval(trainer, testloader, oodloader,[], classes_idx_ID, args.save_path, args.save_acts, classes)
        print("TEST_Acc:", test_acc)
        print("TEST_Acc:", test_acc, file = f)
        print("OOD Metrics (AUPR, AUC, FPR95) :", ood_ap, ood_auc, ood_fpr95)
        print("Exclusivity :", np.mean(excl_mat), np.std(excl_mat))
        print("Cosine Distance :", np.mean(cosine_mat), np.std(cosine_mat))
        print("OOD Metrics (AUPR, AUC, FPR95) :", ood_ap, ood_auc, ood_fpr95, file = f)
        print("Exclusivity :", np.mean(excl_mat), np.std(excl_mat), file=f)
        print("Cosine Distance :", np.mean(cosine_mat), np.std(cosine_mat), file=f)
    f.close()

