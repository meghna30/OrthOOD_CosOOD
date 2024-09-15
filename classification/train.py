import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from model import Model
from model_cosine_gtsrb import GTSRB
from model_cosine_svhn import SVHN
from model_cosine1 import MNIST
from model_vgg16_cosine import CIFAR10
from model_cosine_cifar import CIFAR

from scipy.stats import entropy
from ap_metrics import APMetrics
import pdb
from torch.autograd import Variable

class Trainer:
    def __init__(self,output_dim,device, args):

        self.output_dim = output_dim
        self.device = device
        self.learning_rate = args.lr
        self.orthogonal_loss = args.opl
        self.mu = args.mu
        if args.dataset1 == 'mnist':
            self.model = MNIST(self.output_dim, args.cosine_sim, args.baseline, not args.train)
            # self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = 0.9, weight_decay = 0.0001)
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
            no_layers = 8
            ll_layer_idx = 2
        elif args.dataset1 == 'fmnist':
            self.model = MNIST(self.output_dim, args.cosine_sim, args.baseline, not args.train)
            # self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = 0.9, weight_decay = 0.0001)
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
            no_layers = 8
            ll_layer_idx = 2
        elif args.dataset1 == 'svhn':
            self.model = SVHN(self.output_dim, args.cosine_sim, args.baseline, not args.train)
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
            no_layers = 9
            ll_layer_idx = 4
        elif args.dataset1 == 'gtsrb':
            self.model = GTSRB(self.output_dim, args.cosine_sim, args.baseline, not args.train)
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
            no_layers  = 9
            ll_layer_idx = 4
        elif args.dataset1 == 'cifar10':
            self.model = CIFAR10(self.output_dim, args.cosine_sim, args.baseline, not args.train)
            # self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = 0.9, weight_decay =5e-4)
            # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=1e-6)
            no_layers =  8 #16
            ll_layer_idx = 2 #6


        # self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = 0.9, weight_decay = 0.0001)



        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        # if args.dataset1 =='cifar10':
            # usef lr = 0.001

        # else:
        #     self.optimizer = optim.Adam(self.model.parameters(),lr = self.learning_rate)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.learning_rate, weight_decay = 1e-6)
        #self.optimizer_g = optim.Adam(self.model_gating.parameters(), lr = self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCELoss()
        #if not args.baseline:
        #self.criterion = nn.NLLLoss()
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)



    def calculate_accuracy(self,y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)

        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc, top_pred.squeeze(1)

    def oploss(self, feats, y, mu):
        """
        feats: shape (B,D)
        labels: shape (B,1)
        """

        feats = feats
        labels = y.unsqueeze(1)
        feats = F.normalize(feats, p=2, dim=1)
        mask = torch.eq(labels, labels.t())
        mask_pos = mask.fill_diagonal_(0)
        mask_neg = torch.logical_not(mask)

        dot_prod = torch.matmul(feats, feats.t())
        pos_total = (mask * dot_prod).sum()
        neg_total = torch.abs(mask_neg*dot_prod).sum()
        pos_mean = pos_total / (mask_pos.sum() + 1e-6)
        neg_mean = neg_total / (mask_neg.sum() + 1e-6)

        # loss = mu*(0.01*(1.0 - pos_mean) + neg_mean)
        loss = mu * neg_mean
        # loss = mu*(-1 * torch.mean(F.softmax(dot_prod, dim=1) * F.log_softmax(dot_prod, dim=1)))
        return loss




    def optimize(self,data,ood_class, in_dist_classes):#,mask_c, mask_f, nodes, nodes_f, w, w_f):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for (x,y) in data:
            x = x.to(self.device)
            y = y.to(self.device)

            if len(x) == 1:
                continue

            self.optimizer.zero_grad()
            layers , y_pred = self.model(x)

            if len(ood_class) == 1:
                y = torch.where(y>ood_class[0], y-1,y)

            elif len(ood_class) > 1:
                y_new = y.clone()
                for j in range(0,len(in_dist_classes)):

                    y_new = torch.where(y == in_dist_classes[j], j, y_new)
                y = y_new.clone()

            if self.orthogonal_loss:
                feats = layers[len(layers)-2]

            loss = self.criterion(y_pred,y)
            if self.orthogonal_loss:
                feats = layers[len(layers)-2]
                loss += self.oploss(feats, y, self.mu)

            acc, _ = self.calculate_accuracy(y_pred,y)

            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # print(loss.item())
        return epoch_loss/len(data), epoch_acc/len(data)



    def evaluate(self,data,ood_class,in_dist_classes, excl_metric = False):

        epoch_loss = 0
        epoch_acc = 0
        y_list = np.ones((self.output_dim))

        labels_list = []
        preds_list = []
        feats_list = []
        act_list  = []
        y_true_list = []
        ent_list = []
        self.model.eval()

        with torch.no_grad():

            for (x, y) in data:

                x = x.to(self.device)
                y = y.to(self.device)

                layers ,y_pred = self.model(x)

                if len(ood_class) == 1:
                    y = torch.where(y>ood_class[0], y-1,y)


                elif len(ood_class) > 1:
                    y_new = y.clone()
                    for j in range(0,len(in_dist_classes)):
                        y_new = torch.where(y == in_dist_classes[j], j, y_new)
                    y = y_new.clone()
               
                loss = self.criterion(y_pred, y)
                acc, top_pred = self.calculate_accuracy(y_pred, y)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if excl_metric:
                    labels_list += list(y.cpu().numpy())
                    preds_list += list(top_pred.cpu().numpy())
                    feats_list += list(layers[len(layers)-2].cpu().numpy())
                    softmax_map = nn.Softmax(dim=1)(y_pred).cpu().numpy()
                    # sfmax = np.max(softmax_map, axis = 1 )
                    sfmax = entropy(softmax_map, axis = 1)
                    ent_list +=list(sfmax)


        return epoch_loss/len(data), epoch_acc/len(data), labels_list, feats_list, preds_list, ent_list

    def evaluate_ood(self, data):
        """
        data is already formatted to be 1 for OOD classes and 0 ID classes
        """
        ap_metrics = APMetrics(2)
        epoch_loss = 0
        epoch_acc = 0
        y_true_list = []
        ent_list = []
        feats_list = []
        self.model.eval()
        with torch.no_grad():
            for (x,y) in data:
                x = x.to(self.device)
                y = y.to(self.device)
                # y = torch.where(y>364,0,1)
                layers, y_pred = self.model(x)

                softmax_map = nn.Softmax(dim=1)(y_pred).cpu().numpy()

                entropy_map = entropy(softmax_map, axis = 1)
                # entropy_map = 1-np.max(softmax_map, axis = 1 )
                y_true = y.cpu().numpy()
                y_true_list +=list(y_true)
                ent_list +=list(entropy_map)
                feats_list += list(layers[len(layers)-2].cpu().numpy())
      
        ap_metrics.update(y_true_list, ent_list)
        fpr95, tpr = ap_metrics.compute_fpr(y_true_list,ent_list )
        feats = np.array(feats_list)
        mean_act = np.mean(np.linalg.norm(feats, axis = 1))
        # print("FPR-95 :", fpr95, tpr)
        # print(ap_metrics.get_results())
        curr_ap, curr_auc = ap_metrics.get_results()

        return curr_ap, curr_auc, fpr95, mean_act

    def evaluate_odin_val(self,data,ood_class,in_dist_classes, excl_metric = False):

        epoch_loss = 0
        epoch_acc = 0
        y_list = np.ones((self.output_dim))
        T = 1000
        # epsilon = 0.0038
        # epsilon = 0.0014 ## for cifar10
        epsilon = 0.0004 ## everything else 
        labels_list = []
        preds_list = []
        feats_list = []
        act_list  = []
        y_true_list = []
        ent_list = []
        self.model.eval()

    

        for (x, y) in data:

            x = x.to(self.device)
            y = y.to(self.device)

            x.requires_grad = True 
        
            layers, y_pred = self.model(x)

            
            nnOutputs = y_pred - torch.max(y_pred, dim=1, keepdim=True)[0]
            nnOutputs = nn.Softmax(dim=1)(nnOutputs)
        
            labels = torch.max(nnOutputs,1)[1]
            # labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
            preds_T = y_pred 
            preds_T_sfm = torch.log(nn.Softmax(dim=1)(preds_T))
            loss = self.criterion(preds_T_sfm, labels)

            self.model.zero_grad()
            loss.backward()
            x_grad = x.grad.data 

            sign_data_gard = (-x_grad).sign()
            x_pert = x - epsilon*sign_data_gard

            with torch.no_grad():
                _, new_preds= self.model(x_pert)
            
            new_preds = new_preds/T 
            acc, top_pred = self.calculate_accuracy(new_preds, y)
            epoch_loss += 0
            epoch_acc += acc.item()

            if excl_metric:
                labels_list += list(y.cpu().numpy())
                preds_list += list(top_pred.cpu().numpy())
                feats_list += list(layers[len(layers)-2].detach().cpu().numpy())
                softmax_map = nn.Softmax(dim=1)(new_preds).detach().cpu().numpy()
                # sfmax = np.max(softmax_map, axis = 1 )
                sfmax = entropy(softmax_map, axis = 1)
                ent_list +=list(sfmax)


        return epoch_loss/len(data), epoch_acc/len(data), labels_list, feats_list, preds_list, ent_list

    


