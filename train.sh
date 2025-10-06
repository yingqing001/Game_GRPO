#!/bin/bash

torchrun --nproc_per_node=4 train.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --num_groups_per_gpu 1 \
    --num_per_group 4 \
    --minibatch_size 4 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --max_steps 1000 \
    --out_dir "./output/checkpoints" \
    --wandb_project "game-2048" \
    --wandb_run_name "test" 