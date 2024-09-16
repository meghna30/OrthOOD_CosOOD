
# 1. Preliminaries 
  - Install requirements.txt to setup the python environment
  - Pytorch datasets for MNIST, FMNIST, SVHN, Places365 and CIFAR10
  - GTSRB can be downloaded here https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

    Make sure to set the data_path variable in main.py to the data folder path

# 3. OrthOOD 
  ID: FMNIST, OOD: MNIST

## Train OrthOOD 

    python main.py --dataset1 fmnist --dataset2 mnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 20 --train True --save_path test/orth --opl True --mu 1.0

## Evaluate OrthOOD 

    python main.py --dataset1 fmnist --dataset2 mnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 20 --load_checkpoint True --save_path test/orth --opl True --mu 1.0

## Hyperpameter mu 
  CIFAR10: 0.5, FMNIST: 10, GTSRB: 0.5, SVHN: 0.25

# 4. CosOOD 
  ID: FMNIST, OOD: MNIST

## Train CosOOD 
    
    python main.py --dataset1 fmnist --dataset2 mnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 20 --train True --save_path test/cosine_sim --cosine_sim True 

## Evaluate CosOOD 

    python main.py --dataset1 fmnist --dataset2 mnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 20 --load_checkpoint True --save_path test/cosine_sim --cosine_sim True 




    









