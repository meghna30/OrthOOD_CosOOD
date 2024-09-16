

# 1. Preliminaries 
    - Install requirements.txt to setup the python environment

    - DownLoad Cityscapes, Lost & Found and Road Anomaly datasets. 

    - Train the base segmentation

    -  Evaluate for ID accuracy 

        python3 main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.01  --crop_size 768 --batch_size 4 --output_stride 16 
        --data_root /data/cityscapes --test_only --ckpt checkpoints/cityscapes_best.pth --baseline  --save_val_results 


# 2. OrthOOD 

## Compute the per class centroids using the train set.

    python3 main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --model_tag /test --test_only --ckpt checkpoints/cityscapes_best.pth  --compute_centroids --out_file best_cents --baseline

Note - Make sure the train set is used in the validation function, see line 532 in main.py 

## Train OrthOOD  

    python3 main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.01  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --model_tag model_orth --random_seed 5  --total_itrs 10000 --ckpt checkpoints/cityscapes_best.pth --baseline --cent_path output/best_cents.npz --orthogonal_loss --mu 250.0 

    mu: 250

## Evaluate OrthOOD  
OOD: Lost & Found 

    python3 main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.01  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --test_only --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_model_orth.pth --baseline  --save_val_results --ood_seg --ood_data --ood_dataset lostandfound --ood_data_root /data/LostFound


# 3. CosOOD 

## Train CosOOD 

    python3 main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.01  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --model_tag model_cosine --random_seed 6 --total_itrs 10000 --ckpt checkpoints/cityscapes_best.pth --cosine_sim  --baseline
    
## Evalaute CosOOD 
OOD: Lost & Found

Comment line 310 in baseline/src/model/deepv3.py

    python3 main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.01  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --model_tag /test --test_only --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_model_cosine.pth --baseline --cosine_sim --save_val_results --ood_seg --ood_data --ood_dataset lostandfound --ood_data_root /data/LostFound



