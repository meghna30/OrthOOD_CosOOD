import torch
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as data
import torch.optim as optim
from torchvision import models
from excl_metric import compute_excl
from train import Trainer

from excl_metric_ood import compute_excl_ood

import pdb


def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.constant_(m.bias, 0)

def run(trainer,args, trainloader,valloader, testloader, oodloader, classes_idx_OOD, classes,classes_idx_ID, exp_no):

    if args.load_checkpoint:
        checkpoint = torch.load(args.save_path+'/model_{}.pt'.format(exp_no))
        trainer.model.load_state_dict(checkpoint)

    else:
        trainer.model.apply(weights_init_)


    if args.train:
        prev_loss = 1e30
        prev_loss_g = 1e30
        for z in range(0,1):
            print("TRAINING")
            if args.dataset1 == 'cifar10_pret':
                alexnet = models.vgg16(pretrained=True)
                output_dim = args.ID_tasks
                alexnet.classifier[3] = nn.Linear(4096,1024)
                alexnet.classifier[6] = nn.Linear(1024, output_dim)
                alexnet_dict = alexnet.state_dict()
                model_dict = trainer.model.state_dict()
                pretrained_dict = {k : v for k,v in alexnet_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                trainer.model.load_state_dict(model_dict)
        
            # scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size = 10, gamma = 0.5) ## works for logits norm with lr - 0.001
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=200)

            for epoch in range(args.epochs):
                start_time = time.time()
                train_loss, train_acc = trainer.optimize(trainloader,classes_idx_OOD, classes_idx_ID)
                end_time = time.time()
                print(f'\t EPOCH: {epoch+1:.0f} | time elapsed: {end_time - start_time:.3f}')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                loss, acc,_,_ ,_,_= trainer.evaluate(valloader,classes_idx_OOD,classes_idx_ID)
                print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')
                # ap, auc, fpr95,_ = trainer.evaluate_ood(oodloader)
                # print("OOD Metrics (AUPR, AUC, FPR95) :", ap, auc, fpr95)
                if args.dataset1 =='cifar10':
                    scheduler.step()
                if loss < prev_loss:
                    prev_loss = loss
                    train_loss, train_acc= trainer.optimize(valloader,classes_idx_OOD, classes_idx_ID)
                    torch.save(trainer.model.state_dict(), args.save_path+'/model_{}.pt'.format(exp_no))

            
        print("TESTING")
        loss, acc, labels_list, feats_list, preds_list,_ = trainer.evaluate(testloader,classes_idx_OOD,classes_idx_ID)
        print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')
        ap, auc, fpr95,_ = trainer.evaluate_ood(oodloader)
        print("OOD Metrics (AUPR, AUC, FPR95) :", ap, auc, fpr95)

def do_ood_eval(trainer,testloader, oodloader,classes_idx_OOD, classes_idx_ID, save_path, save_acts, class_names):

    print("ID Test Accuracy")
    loss, acc, labels_list, feats_list, preds_list, sf_list_id = trainer.evaluate(testloader,classes_idx_OOD,classes_idx_ID, excl_metric = True)
    _, _, labels_list_ood,feats_list_ood, _, sf_list_ood  = trainer.evaluate(oodloader,classes_idx_OOD,classes_idx_ID, excl_metric = True)
    # excl_mat = 0
    # cosine_mat = 0
   
    excl_mat, cosine_mat, mean_act = compute_excl(feats_list, preds_list, labels_list, classes_idx_ID)
   

    print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')
    print("OOD Evaluation")

    ap, auc, fpr95, mean_act_ood = trainer.evaluate_ood(oodloader)
    # ap, auc, fpr95, mean_act_ood = trainer.evaluate_odin(oodloader)
    print(ap, auc, fpr95, mean_act, mean_act_ood)
    return acc, ap, auc, fpr95, excl_mat,cosine_mat , mean_act, mean_act_ood
