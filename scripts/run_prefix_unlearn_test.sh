#!/bin/bash
screen_name=$1
gpu_id=$2
unlearn_task=$3

cd /home/yfliu/data/data_from_58/code/PET_unlearn

screen -r "${screen_name}" -X stuff "CUDA_VISIBLE_DEVICES=${gpu_id} \
python train_multi_task.py --sample_small \
--add_prefix  --prefix_dir './ckpts/${unlearn_task}/seed_666/${unlearn_task}_prefix_small_last' \
--test_only --unlearn_dataset_name ${unlearn_task} \
--load_from_pretrained './ckpts/seed_666/multi_task_best.pt' \n"
