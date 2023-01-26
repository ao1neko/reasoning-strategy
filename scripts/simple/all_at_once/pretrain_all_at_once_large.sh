#!/bin/bash

project_dir=$1
save_model_dir=$2



CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=all_at_once \
        --train_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_all_at_once.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_all_at_once.jsonl"  \
        --test_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_all_at_once.jsonl"  \
        --model_name=t5 \
        --load_model_dir="t5-large" \
        --train=true \
        --seed="42" \
        --predict=false \
        --train_epochs=30 \
        --eval_steps=100  \
        --save_steps=100 \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/t5_large/all_at_once" \
        --run_dir="42/pretrain/t5_large/all_at_once" \
        --batch_size=32 \
        --learning_rate=0.0001 \
        --load_tokenizer="t5-large"