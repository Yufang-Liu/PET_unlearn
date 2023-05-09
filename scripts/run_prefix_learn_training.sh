#!/bin/bash
screen_name=$1
gpu_id=$2
dataset_name=$3

cd /home/yfliu/data/data_from_58/code/PET_unlearn

screen -r "${screen_name}" -X stuff "CUDA_VISIBLE_DEVICES=${gpu_id} \
python train_single_task.py --sample_small --add_prefix --dataset_name ${dataset_name} \
--fix_classifier --learning_type learn \
-t \"io.last_prefix_model='./ckpts/${dataset_name}/seed_666/${dataset_name}_prefix_learn_small_last'\" \
-t \"io.best_prefix_model='./ckpts/${dataset_name}/seed_666/${dataset_name}_prefix_learn_small_best'\" \n"
