#!/bin/bash
screen_name=$1
gpu_id=$2

cd /home/yfliu/data/data_from_58/code/PET_unlearn

screen -r ${screen_name} -X stuff "CUDA_VISIBLE_DEVICES=${gpu_id} \
python train_single_task.py --sample_small -t \"io.last_model='./ckpts/seed_666/dbpedia_small_last.pt'\"\
 -t \"io.best_model='./ckpts/seed_666/dbpedia_small_best.pt'\" -t \"optim.lr=3e-5\"\n"