import numpy as np
import data_loader
import torch.utils.data as data

import pdb


def single_dataset_loader(dataset, data_path, batch_size, ood_class_idx):
    ## load dataset1
    if dataset == "cifar10":
        trainloader,valloader, testloader,classes, ood_testset, testset = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx,0, False)
    elif dataset == "fmnist":
        trainloader,valloader, testloader,classes, ood_testset, testset = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx, 0, False)
        classes[0] = 'top'
    elif dataset == "mnist":
        trainloader,valloader, testloader,classes, ood_testset, testset = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx,0, False)     
    elif dataset =="svhn":
        trainloader,valloader, testloader,classes, ood_testset, testset = data_loader.data_loader_SVHN(data_path, batch_size, ood_class_idx,0, False)
    elif dataset == "gtsrb":
        trainloader,valloader, testloader,classes, ood_testset, testset = data_loader.data_loader_GTSRB(data_path, batch_size, ood_class_idx,0, False)
 
 
    else:
        print("Invalid dataset ")
        exit()
    oodloader = data.DataLoader(ood_testset, batch_size = batch_size, shuffle = True, num_workers = 2)
    return trainloader,valloader, testloader, classes, oodloader

def mutliple_dataset_loader(data_path, dataset1, dataset2, batch_size):
    ood_class_idx = []

        ## load dataset1
    if dataset1 == "cifar10":
        trainloader,valloader, testloader,classes, _, testset = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx, 1, False)
        testset.targets[:] = [0]*len(testset)
    elif dataset1 == "fmnist":
        trainloader,valloader, testloader,classes, _, testset = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx, 1, False)
        testset.targets[:] = 0
    elif dataset1 == "mnist":
        trainloader,valloader, testloader,classes, _, testset  = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx, 1, False)
        testset.targets[:] = 0
    elif dataset1 == "gtsrb":
        trainloader,valloader, testloader,classes, _, testset = data_loader.data_loader_GTSRB(data_path, batch_size, ood_class_idx, 1, False)
        testset.targets[:] = 0
    elif dataset1 == "svhn":
        trainloader,valloader, testloader,classes, _, testset = data_loader.data_loader_SVHN(data_path, batch_size, ood_class_idx, 1, False)
        testset.labels[:] =0

 
    else:
        print("Invalid dataset 1")
        exit()

    if dataset2 == "cifar10":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx, 1, True)
    elif dataset2 == "fmnist":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx, 1, True)
    elif dataset2 == "mnist":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx,1, True )
    elif dataset2 == "svhn":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_SVHN(data_path, batch_size, ood_class_idx, 1, True)
    elif dataset2 == "gtsrb":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_GTSRB(data_path, batch_size, ood_class_idx, 1, True)
    elif dataset2 == "lsun":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_LSUN(data_path, batch_size, ood_class_idx, 1, True)
    elif dataset2 == "places":
        _,_, _,_, ood_testset, _ = data_loader.data_loader_Places(data_path, batch_size, ood_class_idx, 1, True)
        

    else:
        print("Invalid dataset ")
        exit()

   
    combined_ood_set = data.ConcatDataset([testset, ood_testset])
    oodloader = data.DataLoader(combined_ood_set, batch_size = batch_size, shuffle = True, num_workers = 2)

    # for plotting activations 
    # oodloader = data.DataLoader(ood_testset, batch_size = batch_size, shuffle = True, num_workers = 2)
   
    return trainloader, valloader, testloader, classes, oodloader
