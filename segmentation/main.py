"""
Add source
"""

from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, LostAndFound, FishyScapes, FishyScapesLF, RaodAnomaly
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from uncertainty.metrics import Umetrics
from excl_metric import compute_excl
from centroids import compute_centroids
## baseline import
from baseline.src.model_utils import load_network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.visualizer import Visualizer
from feature_visualization import visualize 

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2

import pdb
from uncertainty.ap_metrics import APMetrics
import numpy.ma as ma


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--ood_data_root", type = str, default='./datasets/data/LostFound')
    parser.add_argument("--meta_save_path", type=str, default='./datasets/data/meta_data/train/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--out_file", type=str, default='out')
    


    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--ood_seg", action='store_true', default=False)
    parser.add_argument("--save_umetrics", action='store_true', default=False)
    parser.add_argument("--save_feats", action='store_true', default = False)
    parser.add_argument("--compute_exclusivity", action='store_true', default = False)
    parser.add_argument("--compute_centroids", action='store_true', default = False)

    parser.add_argument("--visualize_features", action='store_true', default = False)

    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=40e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--sigma", type=float, default=0.06,
                        help="entropy regularization control param")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Orthogonal projection loss control param")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--cent_path", default=None, type=str,
                        help="load centroids")
    
    parser.add_argument("--model_tag", default='none', type=str, help="ood class tag to identify models")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--baseline", action='store_true', default = False)

    parser.add_argument("--save_softmax_only", action='store_true', default = False)
    parser.add_argument("--meta_train_data", action ='store_true', default = False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'nll_loss'], help="loss type (default: False)")
    parser.add_argument("--entropy_reg", action='store_true', default = False)
    parser.add_argument("--orthogonal_loss", action='store_true', default=False)
    parser.add_argument("--cosine_sim", action='store_true', default=False)
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # ood options
    parser.add_argument("--ood_train_data",action ='store_true', default=False)
    parser.add_argument("--ood_train_len", type=int, default=500)
    parser.add_argument("--ood_classes", type=str,  nargs = '+', default = None,
                        help = "specify OOD class names")
    parser.add_argument("--ood_id", type=int, nargs = '+', default = None,
                        help = "specify OOD class ids")
    parser.add_argument("--ood_data", action='store_true',default = False, help ='Set to to True if loading different ood dataset')
    parser.add_argument("--ood_dataset", type=str, default='lostandfound')

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')


    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    ID_classes = [255, 1, 2,3,4,5,6,7,8,9,10,12]

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize(512,),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_noise_transform = et.ExtCompose([
            # et.ExtResize(512,512),
            et.GaussianBlurforOOD(35,20, ID_classes),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])


        val_transform = et.ExtCompose([
            # et.ExtResize(512,),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])


        print("OOD classes : {} || OOD IDs : {}".format(opts.ood_classes, opts.ood_id))

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform, get_ood = False, ood_classes = opts.ood_classes, ood_id = opts.ood_id)
        if opts.ood_train_data:
            train_noise_dst = Cityscapes(root=opts.data_root,
                                split='train', transform=train_noise_transform, get_ood = False, ood_classes = opts.ood_classes, ood_id = opts.ood_id)

        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform,get_ood = False, ood_classes = opts.ood_classes, ood_id = opts.ood_id)

        ood_dst = []

        if opts.test_only:
            if not opts.ood_data:
                train_dst_ood = Cityscapes(root=opts.data_root,
                                   split='train', transform=val_transform, get_ood = True, ood_classes = opts.ood_classes, ood_id = opts.ood_id)
                val_dst_ood = Cityscapes(root=opts.data_root,
                                 split='val', transform=val_transform, get_ood = True, ood_classes = opts.ood_classes, ood_id = opts.ood_id)

                ood_dst = torch.utils.data.ConcatDataset([train_dst_ood, val_dst_ood])
            else:
                if opts.ood_dataset == 'lostandfound':
                    ood_dst = LostAndFound(split='test', root = opts.ood_data_root,transform=val_transform )
                elif opts.ood_dataset == 'fishyscapes':
                    ood_dst = FishyScapes(root = opts.ood_data_root, transform= val_transform)
                elif opts.ood_dataset == 'fishyscapes_lf':
                    ood_dst = FishyScapesLF(root = opts.ood_data_root, transform= val_transform)
                elif opts.ood_dataset == 'roadanomaly':
                    ood_dst = RaodAnomaly(root = opts.ood_data_root, transform= val_transform)



        if opts.ood_train_data:
            add_train_len = opts.ood_train_len
            train_len = int(len(train_dst) - add_train_len)
            #
            train_add , _ = torch.utils.data.random_split(train_noise_dst,[add_train_len, train_len], generator=torch.Generator().manual_seed(42))
            train_dst = torch.utils.data.ConcatDataset([train_dst, train_add])

        if opts.meta_train_data:
            train_len = int(len(train_dst) - 500)
            val_dst,_ = torch.utils.data.random_split(train_dst,[500, train_len])

        print("Train ID length", len(train_dst))
        print("Val ID length", len(val_dst))
        print("OOD Data length", len(ood_dst))

    return train_dst, val_dst, ood_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    img_id = 0
    ID_classes = [1, 2,3,4,5,6,7,8,9,10,12]
    #ood_image_gen = od.GaussianBlur(kernel_size = 15, min_max = (1,10), ID_classes = ID_classes)

    ## open save file 
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Created output folder")
    filename = 'output/'+opts.out_file+'.txt'
    save_path = 'output/'+opts.out_file+'.npz'
    f = open(filename, 'w')
    with torch.no_grad():
        ap_metrics = APMetrics(2)
        gaussian_blur = transforms.GaussianBlur(25,(1,10))
        y_true_list = []
        ent_list = []
        excl_mat_avg = 0
        total_nums = np.zeros(19)
        total_nums_exc = np.zeros((19,1))
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs, pan_feats = model(images)
            pan_feats = pan_feats.cpu().numpy()
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            if opts.visualize_features:
                visualize(pan_feats, labels.cpu().numpy())
            
            umterics_obj = Umetrics(opts.num_classes, outputs, preds, images, labels, save_id = i, compute_metrics = True, save_metrics = opts.save_umetrics, save_softmax_only = opts.save_softmax_only, meta_save_path=opts.meta_save_path)


            entropy_map = umterics_obj.entropy_map
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0], entropy_map[0]))

            if opts.compute_exclusivity:
                # excl_mat_curr = compute_excl(pan_feats, preds, targets)               
                # excl_mat_avg = (excl_mat_avg*i + excl_mat_curr)/(i+1)
               
                excl_mat_curr, cosine_mat_curr, curr_nums = compute_excl(pan_feats, preds, targets)
                if i == 0:
                    excl_mat_avg = excl_mat_curr
                    cosine_mat_avg = cosine_mat_curr
                    total_nums_exc = curr_nums

                else:
                    excl_mat_avg = (excl_mat_avg*total_nums_exc + excl_mat_curr)/(total_nums_exc + curr_nums)
                    cosine_mat_avg = (cosine_mat_avg*total_nums_exc + cosine_mat_curr)/(total_nums_exc + curr_nums)
                    
                
                total_nums_exc = total_nums_exc + curr_nums

            if opts.compute_centroids: 
                centroid_mat_curr, curr_nums = compute_centroids(pan_feats, preds, targets)
                if i == 0: 
                    centroid_mat_avg = centroid_mat_curr
                    total_nums = curr_nums
                else:
                    for j in range(0, 19):
                        
                        if len(centroid_mat_curr[j]) > 0:
                            
                            if len(centroid_mat_avg[j]) == 0:
                                centroid_mat_avg[j] = centroid_mat_curr[j]
                            else:
                                centroid_mat_avg[j] = (centroid_mat_avg[j]*total_nums[j] + centroid_mat_curr[j])/(total_nums[j]+1)
                
                total_nums = total_nums + curr_nums

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

                    entropy = (entropy_map[i]*255).astype(np.uint8)/255.0
                   
                    # entropy_map = 1 - umterics_obj.softmax_map_max  ## comment this for entropy computation
                    if opts.ood_dataset == 'lostandfound':
                        mask = np.where(target == 2, 1,0).astype(bool) #lf
                    else:
                        mask = target.astype(bool)

                    mask_1 = np.where(target == 255, 0, 1).astype(bool)

                    ent_masked = ma.masked_array(entropy_map[i], mask_1)
                    ent = ent_masked.data[ent_masked.mask]

                    if opts.ood_dataset == 'lostandfound':
                        labels_true = np.where(target == 2, 1, 0) #lf
                    else:
                        labels_true = target
                    y_true = ma.masked_array(labels_true, mask_1)
                    y_true = y_true.data[y_true.mask]
                   

                    # if len(y_true)>0 and len(ent)>0:
                    if np.count_nonzero(y_true) != 0 :
                        y_true_list +=list(y_true)
                        ent_list+=list(ent)

                        ap_metrics.update(y_true, ent)
                        # print(ap_metrics.get_results())
                    # #
                    # target = loader.dataset.decode_target(target, ood=True).astype(np.uint8)
                    # pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    # Image.fromarray(image).save('results/%d_image.png' % img_id)
                    # # Image.fromarray(target).save('results/%d_target.png' % img_id)
                    # # Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    # Image.fromarray(entropy*mask_1).save('results/%d_entropy.png' % img_id)

                    # fig = plt.figure()
                    # plt.imshow(mask)
                    # # plt.axis('off')
                    # # plt.imshow(pred, alpha=0.7)
                    # ax = plt.gca()
                    # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    # plt.close()
                    # img_id += 1
        
        
        if opts.compute_exclusivity:
            # print("Avg Exclusivity :", excl_mat_avg, file=f)
            print(np.shape(excl_mat_avg))
            print("MEAN Exclusivity :", np.mean(excl_mat_avg))
            print("MEAN Cosine :", np.mean(cosine_mat_avg))
            # np.savez(save_path, excl_mat = excl_mat_avg, cosine_mat = cosine_mat_avg)
        
        if opts.compute_centroids:
            print("saving centroids")
            centroids = []
            for j in range(0, 19):    
                if len(centroid_mat_avg[j]) == 0:
                    centroid_mat_avg[j] = np.zeros((256))
                
                centroids.append(centroid_mat_avg[j].astype(np.float32))
            np.savez(save_path, centroids = centroids)
                
        if opts.save_val_results:
            ap_metrics.update(y_true_list, ent_list)

            fpr95, tpr = ap_metrics.compute_fpr(y_true_list, ent_list)
            print("FPR-95 :", fpr95, tpr)
            print(ap_metrics.get_results())
        score = metrics.get_results()
    return score, ret_samples


def entropy_loss(scores, masks, gt_mask):

    probs = F.softmax(scores,1) + 1e-8
    max_probs = torch.max(probs, dim = 1)[0]
    entropy_i = -probs * torch.log(probs)
    entropy = torch.sum(entropy_i, 1)
    #entropy = entropy*masks*gt_mask*max_probs
    entropy = entropy*masks
    return entropy.mean()



def opl_loss(centroids, feats, labels, mu):
    """
    Orthogonal projection loss 
    """
   
    cents = torch.tensor(np.array(centroids)).float().to('cuda')
    feats = torch.permute(feats,(0,2,3,1))
    labels= labels.unsqueeze(1)
    
    labels = F.interpolate(labels.float(), size=feats.shape[1:3], mode = 'nearest').long().squeeze(1)

    feats = torch.reshape(feats,(-1,256))
    labels = torch.reshape(labels,(-1,1))
   
    indices_ = torch.logical_not(torch.eq(labels[:,0], 255)).nonzero()
    
    labels = labels[indices_,:].squeeze(1)[:,0]
    feats = feats[indices_,:].squeeze(1)  

    feats = F.normalize(feats, p=2, dim=1)
    cents = F.normalize(cents, p = 2, dim=1)
    
  
    dot_prod = torch.matmul(feats, cents.t()) ## NX19
    loss = mu*(-1 * torch.mean(F.softmax(dot_prod, dim=1) * F.log_softmax(dot_prod, dim=1)))

    return loss 



def main():
    opts = get_argparser().parse_args()


    if opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19


    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst, ood_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.

    if opts.ood_seg:
        val_loader = data.DataLoader(
            ood_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (opts.dataset, len(train_dst), len(val_dst)))
    else:
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (opts.dataset, len(train_dst), len(val_dst)))



    # Set up model (all models are 'constructed at network.modeling)
    if not opts.baseline:
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    else:
        # model = load_network(model_name="DeepLabV3+_WideResNet38", num_classes=19,
                                # ckpt_path='checkpoints/cityscapes_best.pth', train=True)
        model = load_network(model_name="DeepLabV3+_WideResNet38", num_classes=19,
                               ckpt_path=None, train=False, cosine_sim = opts.cosine_sim)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    if not opts.baseline:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1*opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        # optim.Adam(model.parameters(), lr=params.learning_rate)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        # criterion = nn.CrossEntropyLoss(reduction='mean')
    elif opts.loss_type =='nll_loss':
        criterion = nn.NLLLoss(ignore_index = 255, reduction='mean')
        prob = nn.LogSoftmax(dim = 1)
    
    if opts.orthogonal_loss:
        centroids = np.load(opts.cent_path, allow_pickle = True)['centroids']

   
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        # print(checkpoint["cur_itrs"])
        model.load_state_dict(checkpoint["model_state"]) # model_state
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9) ## reinitialising scheduler 
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            cur_itrs = 0
            # best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images


    if opts.test_only:
        model.eval()
        if opts.compute_centroids:
            val_score, ret_samples = validate(
                opts=opts, model=model, loader=train_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        else:
            val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return
    interval_loss = 0

    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels_mask = torch.where(labels == 255, 1, 0)

            optimizer.zero_grad()
            outputs, feats = model(images)
            pred = outputs.max(dim=1)[1]
            gt_mask = (pred != labels)
            if opts.loss_type == 'nll_loss':
                log_outputs = prob(outputs)
                loss = criterion(log_outputs, labels)
            else:
                loss = criterion(outputs, labels)
            if opts.entropy_reg:
                loss -= opts.sigma*entropy_loss(outputs, labels_mask, gt_mask) ## was 0.01
            if opts.orthogonal_loss:
                loss += opl_loss(centroids, feats, labels, opts.mu)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_%s.pth' %
                          (opts.model, opts.dataset, opts.model_tag))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_%s.pth' %
                              (opts.model, opts.dataset, opts.model_tag))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl, entropy_map) in enumerate(ret_samples):

                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)

                        colormap = plt.get_cmap('inferno')
                        #heatmap = (entropy_map*255).astype(np.uint8)

                        heatmap = (colormap(entropy_map)*255).astype(np.uint8)[:,:,:3]
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).transpose(2,0,1)

                        concat_img = np.concatenate((img, target, lbl, heatmap), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
            # if cur_itrs >= 15000:
                return


if __name__ == '__main__':
    main()
