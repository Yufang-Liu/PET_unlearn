#!/bin/bash
screen_name=$1
gpu_id=$2
dataset_name=$3

cd /home/yfliu/data/data_from_58/code/PET_unlearn

screen -r "${screen_name}" -X stuff "CUDA_VISIBLE_DEVICES=${gpu_id} \
python train_multi_task.py --sample_small --unlearn_dataset ${dataset_name} \
-t \"io.last_model='./ckpts/seed_666/multi_task_small_finetune_last.pt'\" \
-t \"io.best_model='./ckpts/seed_666/multi_task_small_finetune_best.pt'\" \
-t \"optim.lr=3e-5\" --finetune -t \"trainer.max_epochs=2\" \
--add_prefix --prefix_dir ./ckpts/${dataset_name}/seed_666/${dataset_name}_prefix_small_last \
--load_from_pretrained ./ckpts/seed_666/multi_task_small_best.pt \n"