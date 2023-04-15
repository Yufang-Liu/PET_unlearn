#!/bin/bash
screen_name=$1
gpu_id=$2

cd /home/yfliu/data/data_from_58/code/PET_unlearn

screen -r ${screen_name} -X stuff "CUDA_VISIBLE_DEVICES=${gpu_id} \
python train_single_task.py --sample_small --add_prefix \
--fix_classifier --load_from_pretrained './ckpts/seed_666/dbpedia_small_best.pt' \n"