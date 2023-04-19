#!/bin/bash
screen_name=$1
gpu_id=$2
dataset_name=$3

cd /home/yfliu/data/data_from_58/code/PET_unlearn

screen -r "${screen_name}" -X stuff "CUDA_VISIBLE_DEVICES=${gpu_id} \
python train_single_task.py --sample_small --dataset_name ${dataset_name} \
-t \"io.last_model='./ckpts/${dataset_name}/seed_666/${dataset_name}_small_last.pt'\" \
-t \"io.best_model='./ckpts/${dataset_name}/seed_666/${dataset_name}_small_best.pt'\" \
-t \"optim.lr=3e-5\"\n"