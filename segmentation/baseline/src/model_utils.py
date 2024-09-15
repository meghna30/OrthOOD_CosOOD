import numpy as np
# import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from baseline.src.model.deepv3 import DeepWV3Plus
# from src.model.DualGCNNet import DualSeg_res50


def load_network(model_name, num_classes, ckpt_path=None, train=False, cosine_sim = False):
    network = None
    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)
    
    if model_name == "DeepLabV3+_WideResNet38":
        network = nn.DataParallel(DeepWV3Plus(cosine_sim,num_classes))
    elif model_name == "DualGCNNet_res50":
        network = DualSeg_res50(num_classes)
    else:
        print("\nModel is not known")
        exit()

    if ckpt_path is not None:
        if not cosine_sim:
            network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
        else:
            network_dict = network.state_dict()
            checkpoint_dict = torch.load(ckpt_path)['state_dict']
            pretrained_dict = {k : v for k,v in checkpoint_dict.items() if k in network_dict}
            network_dict.update(pretrained_dict)
            network.load_state_dict(network_dict)
    network = network.cuda()
    if train:
        print("... ok")
        return network.train()
    else:
        print("... ok")
        return network.eval()